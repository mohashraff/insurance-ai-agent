import pandas as pd

from smolagents import CodeAgent, LiteLLMModel

from tools.claim_tools import make_claim_tools


# ==========================================================
# GLOBAL DATASET
# ==========================================================

claims_df = None


# ==========================================================
# FILE LOADER
# ==========================================================

def load_claims(file_path):
    """
    Loads either CSV or XLSX claims data.
    """

    if file_path.lower().endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.lower().endswith(".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Only .csv and .xlsx files are supported.")


# ==========================================================
# LOAD TEST DATA
# ==========================================================

claims_df = load_claims("claims.csv")


# ==========================================================
# TOOLS (shared with app.py)
# ==========================================================

get_claim_by_id, predict_fraud = make_claim_tools(lambda: claims_df)


# ==========================================================
# LOCAL QWEN MODEL
# ==========================================================

llm = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:latest",
    api_base="http://localhost:11434"
)


# ==========================================================
# AGENT
# ==========================================================

agent = CodeAgent(
    tools=[get_claim_by_id, predict_fraud],
    model=llm,
    max_steps=4
)


# ==========================================================
# TEST QUESTION
# ==========================================================

if __name__ == "__main__":

    question = (
        "Is claim 233 fraudulent? "
        "Use the trained fraud model and explain why."
    )

    result = agent.run(question)

    print("\nFINAL ANSWER:")
    print(result)