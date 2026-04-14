"""LLM-as-judge evaluation using a local Ollama model.

The judge scores a (prompt, response) pair on four dimensions and
flags potential hallucinations.  llama3:8b is the recommended judge
because it's fast — reserve larger models for generation.
"""

import json
import re
from ollama_client import generate

# Weights must sum to 1.0
_WEIGHTS = {
    "relevance":    0.35,
    "accuracy":     0.35,
    "completeness": 0.20,
    "conciseness":  0.10,
}

_JUDGE_PROMPT = """You are a strict, impartial evaluator for chatbot responses.

ORIGINAL PROMPT:
{prompt}

CHATBOT RESPONSE:
{response}

Score the response on each dimension from 1 to 10, then decide if there is a hallucination.

RUBRIC:
  10 — perfect / exceeds expectations
  7-9 — good, minor issues only
  5-6 — acceptable but noticeably lacking
  3-4 — poor, major gaps or errors
  1-2 — completely wrong or unhelpful

DEFINITIONS:
  relevance     — does the response directly address what was asked?
  accuracy      — is the information factually correct?
  completeness  — does it cover everything the prompt asked for?
  conciseness   — is it free from unnecessary padding or repetition?
  hallucination — did the model state something factually incorrect or fabricated?

Return ONLY a valid JSON object — no explanation, no markdown fences:
{{"relevance": <int>, "accuracy": <int>, "completeness": <int>, "conciseness": <int>, "hallucination_detected": <true|false>}}"""


def evaluate(prompt: str, response: str, judge_model: str) -> dict:
    """Return evaluation dict with individual scores and a composite quality_score."""
    judge_input = _JUDGE_PROMPT.format(prompt=prompt, response=response)

    try:
        result = generate(judge_input, judge_model)
        raw = result["response"]
    except Exception as exc:
        return {**_default_scores(), "error": str(exc)}

    # Pull out the first {...} block (handles extra prose if the model disobeys)
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not match:
        return _default_scores()

    try:
        ev = json.loads(match.group())
    except json.JSONDecodeError:
        return _default_scores()

    # Sanitise scores
    for k in _WEIGHTS:
        try:
            ev[k] = max(1.0, min(10.0, float(ev[k])))
        except (KeyError, TypeError, ValueError):
            ev[k] = 5.0

    ev["hallucination_detected"] = bool(ev.get("hallucination_detected", False))

    # Weighted composite
    raw_score = sum(ev[k] * _WEIGHTS[k] for k in _WEIGHTS)
    if ev["hallucination_detected"]:
        raw_score *= 0.3  # heavy penalty

    ev["quality_score"] = round(raw_score, 2)
    return ev


_IMPROVE_PROMPT = """You are an expert prompt engineer. A user submitted a prompt to a chatbot and it was scored by an evaluator.

ORIGINAL PROMPT:
{prompt}

CHATBOT RESPONSE:
{response}

SCORES (out of 10):
  Relevance:    {relevance}
  Accuracy:     {accuracy}
  Completeness: {completeness}
  Conciseness:  {conciseness}
  Hallucination detected: {hallucination}
  Overall quality: {quality}

Analyse why the scores are what they are, then give 3 to 5 specific, actionable tips to improve the prompt so the chatbot produces a better response next time.

Focus only on weaknesses (scores below 8). If a dimension scored 8 or above, skip it.
Keep each tip concrete — bad: "be more specific", good: "add the target audience so the model knows the depth of explanation required".

Also provide one rewritten version of the prompt that applies all your tips.

Format your response exactly like this:

TIPS:
1. <tip>
2. <tip>
3. <tip>

IMPROVED PROMPT:
<rewritten prompt>"""


def suggest_improvements(
    prompt: str,
    response: str,
    evaluation: dict,
    judge_model: str,
) -> dict:
    """Return improvement tips and a rewritten prompt.

    Returns a dict with:
        tips             list[str]  — actionable improvement tips
        improved_prompt  str        — rewritten version of the prompt
        raw              str        — full model output (for debugging)
    """
    coach_input = _IMPROVE_PROMPT.format(
        prompt=prompt,
        response=response,
        relevance=evaluation.get("relevance", "?"),
        accuracy=evaluation.get("accuracy", "?"),
        completeness=evaluation.get("completeness", "?"),
        conciseness=evaluation.get("conciseness", "?"),
        hallucination="YES" if evaluation.get("hallucination_detected") else "No",
        quality=evaluation.get("quality_score", "?"),
    )

    try:
        result = generate(coach_input, judge_model)
        raw = result["response"]
    except Exception as exc:
        return {"tips": [], "improved_prompt": "", "raw": str(exc)}

    # Parse TIPS section
    tips: list[str] = []
    tips_match = re.search(r"TIPS:\s*\n(.*?)(?:\n\s*\n|IMPROVED PROMPT:)", raw, re.DOTALL)
    if tips_match:
        for line in tips_match.group(1).strip().splitlines():
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if line:
                tips.append(line)

    # Parse IMPROVED PROMPT section
    improved = ""
    improved_match = re.search(r"IMPROVED PROMPT:\s*\n(.*)", raw, re.DOTALL)
    if improved_match:
        improved = improved_match.group(1).strip()

    return {"tips": tips, "improved_prompt": improved, "raw": raw}


def _default_scores() -> dict:
    return {
        "relevance":             5.0,
        "accuracy":              5.0,
        "completeness":          5.0,
        "conciseness":           5.0,
        "hallucination_detected": False,
        "quality_score":         5.0,
    }
