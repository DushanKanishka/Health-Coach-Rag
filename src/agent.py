import re

import numpy as np

from src.parsing import parse_user_question_to_features
from src.retrieval import retrieve_relevant_chunks

INFO_FIELDS = [
    "avg_steps_7d", "avg_sleep_7d", "avg_stress_7d", "water_glasses",
    "calories_intake", "resting_heart_rate", "age", "bmi",
    "steps", "sleep_hours", "stress_level", "fatigue_score",
]

_LIST_MARKER = r"(?:[-*]|\d+\.)\s"


def _tighten_lists(text: str) -> str:
    """Collapse blank lines between consecutive markdown list items."""
    return re.sub(
        rf"(\n[ \t]*{_LIST_MARKER}[^\n]*)\n{{2,}}(?=[ \t]*{_LIST_MARKER})",
        r"\1\n",
        text,
    )


def health_coach_agent_free_text(
    user_question: str,
    model,
    llm,
    kb_index,
    model_emb,
    feature_cols,
    prompt_template: str,
    no_data_prompt_template: str,
    top_k: int = 3,
):
    """
    1) Parse features from text (LLM extraction)
    2) Compute fatigue risk with ML if enough info
    3) Build simple recommendations from habits
    4) Retrieve guideline chunks (RAG)
    5) Ask LLM to combine everything into one answer
    """
    # 1) Extract structured features
    feats = parse_user_question_to_features(user_question, llm)

    # If no health data was mentioned at all, ask for it instead of dumping
    # generic guidelines based on an unrelated RAG retrieval.
    if not any(feats.get(f) is not None for f in INFO_FIELDS):
        ask_prompt = no_data_prompt_template.format(user_question=user_question)
        response = llm.invoke(ask_prompt)
        answer = _tighten_lists(getattr(response, "content", str(response)))
        return answer, kb_index.iloc[0:0], None, [], feats

    # 2) Try to compute ML fatigue risk
    can_run_ml = all(feats.get(col) is not None for col in feature_cols)
    ml_prob = None
    if can_run_ml:
        X_user = np.array([[feats[col] for col in feature_cols]])
        ml_prob = float(model.predict_proba(X_user)[0][1])  # probability high fatigue

    # 3) Rule-based recommendations using extracted features
    recs = []

    if feats.get("avg_steps_7d") is not None and feats["avg_steps_7d"] < 6000:
        recs.append(
            "Your average steps are below 6,000. Gradually increase daily steps by 1,000–2,000 over the next week."
        )

    if feats.get("avg_sleep_7d") is not None and feats["avg_sleep_7d"] < 7:
        recs.append(
            "Your average sleep is under 7 hours. Aim for a more regular sleep schedule and try to reach 7–9 hours."
        )

    if feats.get("avg_stress_7d") is not None and feats["avg_stress_7d"] >= 7:
        recs.append(
            "Your stress levels are high. Consider daily relaxation breaks, breathing exercises, or light walks."
        )

    if feats.get("water_glasses") is not None and feats["water_glasses"] < 6:
        recs.append(
            "Your water intake seems low. Aim for at least 6–8 glasses of water per day, unless advised otherwise."
        )

    if feats.get("resting_heart_rate") is not None and feats["resting_heart_rate"] > 80:
        recs.append(
            "Your resting heart rate is on the higher side. Increasing regular activity and managing stress may help."
        )

    if not recs:
        recs.append(
            "Your habits are partly on track. Focus on keeping consistency in activity, sleep, hydration and stress."
        )

    # 4) Build user_context for RAG retrieval
    user_context = {k: v for k, v in feats.items() if k in [
        "age", "bmi", "avg_steps_7d", "avg_sleep_7d", "avg_stress_7d",
        "water_glasses", "calories_intake", "resting_heart_rate"
    ]}
    user_context["predicted_high_fatigue_prob"] = ml_prob

    # 5) Retrieve RAG chunks
    top_chunks, scores = retrieve_relevant_chunks(
        user_question,
        kb_index,
        model_emb,
        user_context=user_context,
        top_k=top_k
    )
    context_text = "\n\n".join(top_chunks["text"].tolist())

    # 6) Build final prompt
    if ml_prob is None:
        ml_risk_text = "not computed (not enough numeric data provided)"
    else:
        if ml_prob >= 0.7:
            lvl = "HIGH"
        elif ml_prob >= 0.4:
            lvl = "MODERATE"
        else:
            lvl = "LOW"
        ml_risk_text = f"{ml_prob:.2f} ({lvl} fatigue risk)"

    recs_text = "\n".join(f"- {r}" for r in recs)

    prompt = prompt_template.format(
        ml_risk_text=ml_risk_text,
        recs_text=recs_text,
        context_text=context_text,
        user_question=user_question,
        user_context=user_context,
    )

    response = llm.invoke(prompt)
    final_answer = _tighten_lists(getattr(response, "content", str(response)))

    return final_answer, top_chunks, ml_prob, recs, feats
