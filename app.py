"""
Prompt-based Agent — Streamlit Community Cloud entry point.

Env vars are loaded from (in priority order):
  1. Streamlit secrets  (st.secrets)
  2. .env file          (python-dotenv)
  3. Real environment variables
"""

import os
import uuid
import streamlit as st

# ── 1. Load configuration before importing the agent ────────────────────────

def _bootstrap_env() -> None:
    """Populate os.environ from Streamlit secrets or .env, whichever is present."""
    # Streamlit secrets (flat keys only – nested tables are ignored here)
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass

    # .env file (local development)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)   # don't override what Streamlit already set
    except ImportError:
        pass


_bootstrap_env()

# ── 2. Import agent (env vars must be set first) ─────────────────────────────

from PromptBasedAgent import graph  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

# ── 3. Helpers ────────────────────────────────────────────────────────────────

def make_thread_id(seed: str) -> str:
    """Return a deterministic UUID5 from *seed* (patient_id or session key)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


def run_graph(messages: list, user_name: str, user_id: str, thread_id: str) -> str:
    """Invoke the LangGraph agent and return the last AI message content."""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_name": user_name or None,
            "user_id": user_id or None,
        }
    }
    result = graph.invoke({"messages": messages}, config=config)
    last = result["messages"][-1]
    # Handle both AIMessage objects and plain dicts
    if hasattr(last, "content"):
        return last.content
    return str(last.get("content", last))


# ── 4. Streamlit UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title="Prompt-based Agent", page_icon="🎯")
st.title("🎯 Prompt-based Agent")

# ── Sidebar: identity / config ────────────────────────────────────────────────
with st.sidebar:
    st.header("Session settings")
    user_name = st.text_input("Your name", placeholder="e.g. Alice")
    user_id   = st.text_input("Patient ID", placeholder="e.g. 42")

    # Thread ID is derived from patient_id when provided, else from a random
    # per-browser-session key so conversations don't bleed into each other.
    if "session_seed" not in st.session_state:
        st.session_state.session_seed = str(uuid.uuid4())

    seed      = user_id.strip() if user_id.strip() else st.session_state.session_seed
    thread_id = make_thread_id(seed)

    st.caption(f"Thread ID (UUID5): `{thread_id}`")

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# ── Conversation state ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": ..., "content": ...}

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── New user input ────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message…")

if user_input:
    # Show and store user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Build LangChain message list for the graph
    lc_messages = []
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        else:
            lc_messages.append(AIMessage(content=m["content"]))

    # Invoke graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = run_graph(lc_messages, user_name, user_id, thread_id)
            except Exception as exc:
                response = f"⚠️ Error: {exc}"
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})