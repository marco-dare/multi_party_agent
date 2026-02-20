# Goal Reflection Agent

A LangGraph-powered conversational agent that runs in **ConversationalCare** and on **Streamlit Community Cloud** from the same codebase.

---

## Architecture

```
app.py                      ← Streamlit entry point
PromoptBasedAgent.py        ← LangGraph agent 
prompts/
  agent.prompt    ← System prompt
requirements.txt
.env.example                ← Copy → .env for local dev
```

### Thread IDs (UUID5)

Each conversation is keyed by a **deterministic UUID5** derived from the patient ID (when provided) or a random per-browser-session seed.  
`uuid.uuid5(uuid.NAMESPACE_DNS, seed)`.

---

## Local development

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env — add at minimum OPENAI_API_KEY

# 3. Run
streamlit run app.py
```

---

## Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select `app.py`.
3. Open **App settings → Secrets** and paste:

```toml
OPENAI_API_KEY = "sk-..."
```

The app reads `st.secrets` first, then falls back to `.env`.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
