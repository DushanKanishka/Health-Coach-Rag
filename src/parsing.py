import json
import re


def parse_user_question_to_features(user_question: str, llm):
    """
    Use the LLM to extract numeric lifestyle features from a free-text question.
    Returns a dict with keys needed for ML and rules.
    """
    extraction_prompt = f"""
You are an information extraction assistant.

From the following user's message, extract the following fields if they are mentioned.
If a field is not mentioned, set it to null.

Fields (JSON keys) you must output:
- avg_steps_7d (float, average daily steps over last 7 days)
- avg_sleep_7d (float, average sleep hours over last 7 days)
- avg_stress_7d (float, average stress level 1-10)
- water_glasses (int, glasses of water per day)
- calories_intake (int, approximate daily kcal)
- resting_heart_rate (int, beats per minute)
- age (int, years)
- bmi (float)
- steps (int, today's steps if mentioned)
- sleep_hours (float, today's sleep if mentioned)
- stress_level (int, today's stress 1-10 if mentioned)
- fatigue_score (int, today's fatigue 1-10 if mentioned)

User message:
\"\"\"{user_question}\"\"\"

Return ONLY a valid JSON object, no explanation and no extra text.
"""

    response = llm.invoke(extraction_prompt)
    raw = getattr(response, "content", str(response)).strip()

    # Strip <think>...</think> reasoning blocks some models emit in-content
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Clean markdown fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # Fall back to the first {...} JSON object if there's stray surrounding text
    if not raw.startswith("{"):
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            raw = match.group(0)

    data = json.loads(raw)
    return data
