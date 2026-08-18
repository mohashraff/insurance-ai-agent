import streamlit as st
import pandas as pd
import re

from io import BytesIO
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from smolagents import CodeAgent, LiteLLMModel

from tools.claim_tools import make_claim_tools
from tools.rag_tool import search_documents, set_vectorstore


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Insurance Fraud AI Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Insurance Fraud AI Agent")

st.write(
    "Upload claims data and/or a PDF knowledge base, "
    "then ask questions using the chat."
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "claims_df" not in st.session_state:
    st.session_state.claims_df = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ==========================================================
# CREATE PDF VECTORSTORE
# ==========================================================

@st.cache_resource
def create_vectorstore(file_bytes):

    pdf_file = BytesIO(file_bytes)
    reader = PdfReader(pdf_file)

    documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if not page_text:
            continue

        chunks = splitter.split_text(page_text)

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_number}
                )
            )

    if not documents:
        raise ValueError("No readable text was found in the PDF.")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    vectorstore = FAISS.from_documents(documents, embedding=embeddings)

    return vectorstore, len(reader.pages), len(documents)


# ==========================================================
# UPLOAD SECTION
# ==========================================================

left, right = st.columns(2)

with left:

    st.subheader("1. Claims Dataset")

    claims_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="claims_upload"
    )

    if claims_file is not None:

        try:

            if claims_file.name.lower().endswith(".csv"):
                df = pd.read_csv(claims_file)
            else:
                df = pd.read_excel(claims_file)

            st.session_state.claims_df = df

            st.success(
                f"Loaded {len(df):,} rows and {len(df.columns)} columns."
            )

            with st.expander("Preview claims dataset"):
                st.dataframe(df.head(20), use_container_width=True)
                st.write("**Detected columns:**")
                st.write(list(df.columns))

        except Exception as e:
            st.error(f"Could not load claims file: {e}")


with right:

    st.subheader("2. Knowledge PDF")

    pdf_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_upload"
    )

    if pdf_file is not None:

        try:

            file_bytes = pdf_file.getvalue()

            with st.spinner("Processing PDF..."):
                vectorstore, page_count, chunk_count = create_vectorstore(file_bytes)

            set_vectorstore(vectorstore)

            st.session_state.pdf_loaded = True
            st.session_state.pdf_name = pdf_file.name

            st.success(f"{pdf_file.name} ready")
            st.write(f"Pages: **{page_count}**")
            st.write(f"Chunks: **{chunk_count}**")

        except Exception as e:
            st.session_state.pdf_loaded = False
            st.error(f"Could not process PDF: {e}")


# ==========================================================
# STATUS
# ==========================================================

st.divider()

status_col1, status_col2 = st.columns(2)

with status_col1:
    if st.session_state.claims_df is not None:
        st.success("Claims dataset available")
    else:
        st.info("No claims dataset loaded")

with status_col2:
    if st.session_state.pdf_loaded:
        st.success(f"PDF available: {st.session_state.pdf_name}")
    else:
        st.info("No PDF knowledge base loaded")


# ==========================================================
# TOOLS (shared with test_agent.py)
# ==========================================================
# NOTE: smolagents executes tool code in a separate thread pool
# that does not have access to Streamlit's ScriptRunContext, so
# st.session_state cannot be read lazily inside the tool closure.
# Capture the current dataframe into a plain variable instead —
# this is safe because Streamlit reruns the whole script on every
# interaction, so this value is always fresh for the current run.

current_claims_df = st.session_state.claims_df

get_claim_by_id, predict_fraud = make_claim_tools(
    lambda: current_claims_df
)
# ==========================================================
# LOCAL OLLAMA MODEL
# ==========================================================

llm = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:latest",
    api_base="http://localhost:11434"
)


# ==========================================================
# AGENT
# ==========================================================

agent = CodeAgent(
    tools=[get_claim_by_id, predict_fraud, search_documents],
    model=llm,
    max_steps=4
)


# ==========================================================
# CHAT
# ==========================================================

st.divider()
st.subheader("Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about a claim or the uploaded PDF...")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing..."):

            try:

                fraud_words = [
                    "fraud", "fraudulent", "suspicious",
                    "high risk", "risk", "flagged"
                ]

                context_words = [
                    "typology", "typologies", "policy", "procedure", "process",
                    "why", "match", "known", "pattern", "explain", "mean", "means",
                    "should", "next", "handle", "review process", "escalat"
                ]

                is_fraud_question = any(
                    word in question.lower() for word in fraud_words
                )

                needs_context = any(
                    word in question.lower() for word in context_words
                )

                claim_match = re.search(
                    r"\bclaim\s*(?:id\s*)?#?\s*(\d+)\b",
                    question.lower()
                )

                # ------------------------------------------------
                # DIRECT ML ROUTE
                # Only for clean, single-purpose fraud lookups.
                # The Random Forest remains the source of truth.
                # Qwen cannot override its fraud decision.
                # ------------------------------------------------

                if is_fraud_question and claim_match and not needs_context:

                    claim_id = int(claim_match.group(1))

                    prediction = predict_fraud(claim_id=claim_id)

                    if "error" in prediction:

                        answer = (
                            f"Could not analyze claim {claim_id}.\n\n"
                            f"{prediction['error']}"
                        )

                    else:

                        probability = prediction["fraud_percentage"]
                        threshold = prediction["review_threshold"] * 100
                        flagged = prediction["flagged"]

                        reasons_text = "\n".join(
                            f"  - **{r['feature']}**: {r['direction']} "
                            f"(impact {r['impact']})"
                            for r in prediction["top_reasons"]
                        )

                        if flagged:
                            answer = (
                                f"⚠️ **Claim {claim_id} is flagged for "
                                f"possible fraud.**\n\n"
                                f"- Fraud probability: **{probability}%**\n"
                                f"- Review threshold: **{threshold:.0f}%**\n"
                                f"- Model decision: **FLAGGED FOR REVIEW**\n"
                                f"- Model features used: "
                                f"**{prediction['features_used']}**\n"
                                f"- Optional features imputed: "
                                f"**{prediction['features_imputed']}**\n\n"
                                f"**Top contributing factors:**\n{reasons_text}"
                            )
                        else:
                            answer = (
                                f"✅ **Claim {claim_id} is not currently "
                                f"flagged for fraud.**\n\n"
                                f"- Fraud probability: **{probability}%**\n"
                                f"- Review threshold: **{threshold:.0f}%**\n"
                                f"- Model decision: **NOT FLAGGED**\n"
                                f"- Model features used: "
                                f"**{prediction['features_used']}**\n"
                                f"- Optional features imputed: "
                                f"**{prediction['features_imputed']}**\n\n"
                                f"**Top factors reducing risk:**\n{reasons_text}"
                            )

                # ------------------------------------------------
                # EVERYTHING ELSE → AGENT
                # ------------------------------------------------

                else:

                    if (
                        st.session_state.claims_df is None
                        and not st.session_state.pdf_loaded
                    ):
                        answer = "Upload a claims CSV/XLSX or a PDF first."
                    else:
                        result = agent.run(question)
                        answer = str(result)

                st.markdown(answer)

            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})