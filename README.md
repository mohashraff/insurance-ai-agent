# Insurance Fraud AI Agent

A local, privacy-preserving AI agent that detects and explains insurance claim fraud — combining a deterministic ML model with a local LLM agent for orchestration and explanation.

## The idea

Most "AI fraud detection" demos either let an LLM guess at fraud with no real model behind it, or use a model with no explanation of *why* it flagged something. Neither is good enough when the decision matters.

**Here, the ML model is the only thing that ever makes a fraud decision.** The LLM agent never overrides it — it only explains the model's output and answers contextual questions. This keeps every decision deterministic and auditable.

## Architecture

```
User question
     │
     ▼
Regex router
     │
     ├── Pure fraud lookup → Random Forest model + SHAP explainer
     │     (instant, deterministic, no LLM)
     │
     └── Needs context ("why", "typology", etc.)
              │
              ▼
         smolagents CodeAgent (local Qwen2.5-Coder via Ollama)
              │
              ├── predict_fraud tool → same RF model + SHAP
              └── search_documents tool → FAISS RAG over an uploaded policy PDF
```

## Stack

`Python` · `scikit-learn` (Random Forest) · `SHAP` · `smolagents` · `Ollama` (`qwen2.5-coder`, `nomic-embed-text`) · `LangChain` · `FAISS` · `Streamlit`

Fraud model trained on a healthcare insurance claims dataset from Kaggle.

## Key challenges solved

- **Small local LLMs (1.5B–3B) hallucinate fake tool calls** instead of using real ones — 7B was the reliability floor for agentic tool-calling on this hardware.
- **Streamlit + threaded agent execution**: smolagents runs tools in a separate thread without access to `st.session_state` — fixed by snapshotting state into a plain variable before invoking the agent.
- **Explainability**: wrapped predictions with a SHAP `TreeExplainer`, aggregating one-hot-encoded features back to original claim fields for human-readable reasons.
- **Deterministic-vs-generative routing**: a regex router distinguishes "give me the model's answer" from "explain/contextualize it," so the LLM never quietly substitutes its own judgment for the trained classifier.


## Running it locally

This app depends on a local Ollama instance for the LLM and embeddings, so it's designed to run locally rather than on a cloud host.

1. Install [Ollama](https://ollama.com), then pull the models:
   ```
   ollama pull qwen2.5-coder
   ollama pull nomic-embed-text
   ```
2. Clone this repo and install dependencies:
   ```
   git clone <[your-repo-url](https://github.com/mohashraff/insurance-ai-agent.git)>
   cd insurance-ai-agent
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```
4. In the browser tab that opens, upload a claims CSV/XLSX (must include a `Claim_ID` column and the core fields the model expects) and, optionally, a PDF for the knowledge base. Then ask questions in the chat, e.g.:
   - `"is claim 233 fraud?"`
   - `"is claim 233 fraud and why?"`
   - `"does claim 233 match any known fraud typology?"`

## Status

Working local prototype, built as a learning project in hybrid ML/LLM system design and explainable AI.

## Disclaimer

Educational/demo project. Not intended for production use without a validated dataset, feature review, and compliance sign-off.

## Kaggle dataset used to train the model
[Dataset](https://www.kaggle.com/datasets/tejalaveti2306/health-insurance-claims-data-for-fraud-detection/data)
