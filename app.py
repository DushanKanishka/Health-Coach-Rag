import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq

from src.retrieval import load_kb_index, load_embedding_model
from src.models import train_rf_model_and_return
from src.agent import health_coach_agent_free_text


# =========================
# STREAMLIT BASIC CONFIG + CUSTOM THEME
# =========================

st.set_page_config(
    page_title="Digital Health Coach",
    page_icon="🩺",
    layout="wide",
)

# Simple custom CSS for nicer UI
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f9fafb 0%, #ecfeff 40%, #f5f3ff 100%);
    }

    /* Main title */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }

    /* Chat bubbles */
    .chat-bubble {
        padding: 0.75rem 1rem;
        border-radius: 14px;
        margin-bottom: 0.5rem;
        white-space: pre-wrap;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .chat-user {
        background: #0f766e;
        color: #ecfeff;
    }
    .chat-bot {
        background: #f3f4f6;
        color: #111827;
        border: 1px solid #e5e7eb;
    }

    /* Metrics card style */
    .metric-card {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background: rgba(255,255,255,0.85);
        border: 1px solid #e5e7eb;
        margin-bottom: 0.75rem;
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# HEADER
# =========================

st.markdown(
    "<div class='main-title'>🩺 Digital Health Coach – Intelligent Chatbot</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Hybrid system: ML-based fatigue risk + rule-based coaching + RAG over WHO & healthy lifestyle guidelines, all wrapped in an LLM agent.</div>",
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("### ℹ️ How this works")
    st.write(
        """
        This app combines:
        - ✅ **Random Forest** fatigue risk prediction
        - ✅ **Rule-based** lifestyle recommendations
        - ✅ **RAG** over WHO / healthy diet / physical activity PDFs
        - ✅ **Groq LLM** to generate a final answer

        **How to use:**
        - Describe your habits (sleep, steps, diet, stress, water, etc.)
        - Ask what you should improve
        - The agent analyses your text and responds like a coach.
        """
    )
    st.markdown("---")


# =========================
# CONFIG / LLM (Groq)
# =========================

try:
    import config
except ValueError as e:
    st.error(str(e))
    st.stop()

llm = ChatGroq(
    groq_api_key=config.GROQ_API_KEY,
    model_name=config.GROQ_MODEL_NAME,
    temperature=config.GROQ_TEMPERATURE,
)

SYSTEM_PROMPT_TEMPLATE = config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
NO_DATA_PROMPT_TEMPLATE = config.NO_DATA_PROMPT_PATH.read_text(encoding="utf-8")


# =========================
# CACHED LOADERS
# =========================

@st.cache_resource
def _load_kb_index():
    return load_kb_index(config.KB_INDEX_PATH)


@st.cache_resource
def _load_embedding_model():
    return load_embedding_model(config.EMBEDDING_MODEL_NAME)


@st.cache_resource
def _train_rf_model_and_return():
    return train_rf_model_and_return(config.DATA_PATH, config.FEATURE_COLS)


kb_index = _load_kb_index()
model_emb = _load_embedding_model()
rf_model = _train_rf_model_and_return()


# =========================
# CHATBOT UI – 2 COLUMNS
# =========================

left_col, right_col = st.columns([2.2, 1.3])

with left_col:
    st.markdown("### 💬 Chat with your Digital Health Coach")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm your digital health coach. "
                    "Tell me about your daily habits (sleep, steps, stress, diet, water, etc.) "
                    "and ask what you can improve."
                ),
            }
        ]

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            bubble_class = "chat-user" if msg["role"] == "user" else "chat-bot"
            st.markdown(
                f"<div class='chat-bubble {bubble_class}'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

    user_input = st.chat_input("Type your question about your health habits...")

    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(
                f"<div class='chat-bubble chat-user'>{user_input}</div>",
                unsafe_allow_html=True,
            )

        # Generate health coach response
        with st.chat_message("assistant"):
            with st.spinner("Thinking (ML + RAG + LLM)..."):
                try:
                    answer, used_chunks, ml_prob, recs, feats = health_coach_agent_free_text(
                        user_question=user_input,
                        model=rf_model,
                        llm=llm,
                        kb_index=kb_index,
                        model_emb=model_emb,
                        feature_cols=config.FEATURE_COLS,
                        prompt_template=SYSTEM_PROMPT_TEMPLATE,
                        no_data_prompt_template=NO_DATA_PROMPT_TEMPLATE,
                        top_k=4
                    )
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")
                    assistant_text = "Sorry, something went wrong while processing your request."
                    ml_prob = None
                    recs = []
                    feats = {}
                    used_chunks = pd.DataFrame()
                else:
                    # Build a nice answer text with risk info
                    if not recs:
                        # No health data was extracted at all — `answer` is already a
                        # conversational request for more details, nothing to append.
                        assistant_text = answer
                    elif ml_prob is None:
                        risk_text = (
                            "I couldn't compute a precise fatigue risk because not enough "
                            "numeric data was provided."
                        )
                        assistant_text = answer + "\n\n---\n\n" + risk_text
                    else:
                        if ml_prob >= 0.7:
                            lvl = "HIGH"
                        elif ml_prob >= 0.4:
                            lvl = "MODERATE"
                        else:
                            lvl = "LOW"
                        risk_text = f"My estimate of your high-fatigue risk is **{ml_prob:.2f}** ({lvl})."
                        assistant_text = answer + "\n\n---\n\n" + risk_text

                st.markdown(
                    f"<div class='chat-bubble chat-bot'>{assistant_text}</div>",
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )

                # Store latest details in session for right column
                st.session_state["last_ml_prob"] = ml_prob
                st.session_state["last_recs"] = recs
                st.session_state["last_feats"] = feats
                st.session_state["last_chunks"] = used_chunks


with right_col:
    st.markdown("### 📊 Insight Panel")

    ml_prob = st.session_state.get("last_ml_prob", None)
    recs = st.session_state.get("last_recs", [])
    feats = st.session_state.get("last_feats", {})
    used_chunks = st.session_state.get("last_chunks", pd.DataFrame())

    # Fatigue risk card
    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Fatigue Risk</div>", unsafe_allow_html=True)
        if ml_prob is None:
            st.markdown(
                "<div class='metric-value'>N/A</div>"
                "<div class='metric-sub'>Not enough numeric data</div>",
                unsafe_allow_html=True,
            )
        else:
            if ml_prob >= 0.7:
                lvl = "HIGH"
                color = "#b91c1c"
            elif ml_prob >= 0.4:
                lvl = "MODERATE"
                color = "#b45309"
            else:
                lvl = "LOW"
                color = "#15803d"
            st.markdown(
                f"<div class='metric-value' style='color:{color}'>{ml_prob:.2f} ({lvl})</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='metric-sub'>Model: Random Forest on 7-day trends</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Recommendations
    st.markdown("#### ✅ Key Recommendations")
    if recs:
        for r in recs:
            st.markdown(f"- {r}")
    else:
        st.markdown("_Ask a question to see personalized suggestions._")

    # Extracted features
    st.markdown("#### 🔍 Extracted Features")
    if feats:
        st.json(feats)
    else:
        st.caption("No features extracted yet. Ask something like:\n\n> I sleep 5–6 hours, walk 4,000 steps, drink 3 glasses of water...")

    # RAG Sources
    st.markdown("#### 📚 Guideline Chunks (RAG)")
    if isinstance(used_chunks, pd.DataFrame) and not used_chunks.empty:
        for i, row in used_chunks.iterrows():
            with st.expander(f"Source: {row['source']}"):
                st.write(row["text"])
    else:
        st.caption("Sources will appear here after your first question.")
