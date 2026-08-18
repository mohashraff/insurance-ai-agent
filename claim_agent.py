import pandas as pd

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    tool
)

from tools.fraud_tool import (
    predict_claim_row,
    CORE_REQUIRED_COLUMNS
)


# ==========================================================
# GLOBAL DATASET
# ==========================================================

claims_df = None


# ==========================================================
# CLAIM LOOKUP TOOL
# ==========================================================

@tool
def get_claim_by_id(claim_id: int) -> str:
    """
    Retrieves the complete information for one insurance claim.

    Args:
        claim_id: The Claim_ID to search for.

    Returns:
        All available information for that claim.
    """

    global claims_df


    if claims_df is None:
        return "No claims dataset is loaded."


    if "Claim_ID" not in claims_df.columns:

        return (
            "The uploaded dataset does not "
            "contain a Claim_ID column."
        )


    result = claims_df[
        claims_df["Claim_ID"] == claim_id
    ]


    if result.empty:

        return (
            f"Claim {claim_id} "
            "was not found."
        )


    claim = result.iloc[0]


    return claim.to_json()


# ==========================================================
# FRAUD PREDICTION TOOL
# ==========================================================

@tool
def predict_fraud(claim_id: int) -> dict:
    """
    Runs the trained insurance fraud machine-learning model
    on a specific claim from the loaded dataset.

    Use this tool when the user asks whether a claim is
    fraudulent, suspicious, high risk, or asks for its
    fraud probability.

    Args:
        claim_id: The Claim_ID of the claim to analyze.

    Returns:
        A dictionary containing fraud probability,
        model decision, threshold, and feature availability.
    """

    global claims_df

    if claims_df is None:
        return {
            "error": "No claims dataset is loaded."
        }

    if "Claim_ID" not in claims_df.columns:
        return {
            "error": "The uploaded dataset does not contain Claim_ID."
        }

    result = claims_df[
        claims_df["Claim_ID"] == claim_id
    ]

    if result.empty:
        return {
            "error": f"Claim {claim_id} was not found."
        }

    claim = result.iloc[0]

    try:
        prediction = predict_claim_row(claim)

    except Exception as e:
        return {
            "error": str(e),
            "claim_id": claim_id
        }

    return {
        "claim_id": claim_id,
        "fraud_probability": prediction["fraud_probability"],
        "fraud_percentage": prediction["fraud_percentage"],
        "review_threshold": prediction["threshold"],
        "flagged": prediction["flagged"],
        "decision": (
            "FLAGGED FOR REVIEW"
            if prediction["flagged"]
            else "NOT FLAGGED"
        ),
        "features_used": prediction["number_features_used"],
        "features_imputed": prediction["number_features_imputed"],
        "core_required_features": CORE_REQUIRED_COLUMNS,
    }


# ==========================================================
# FILE LOADER
# ==========================================================

def load_claims(file_path):
    """
    Loads either CSV or XLSX claims data.
    """

    if file_path.lower().endswith(
        ".csv"
    ):

        return pd.read_csv(
            file_path
        )


    elif file_path.lower().endswith(
        ".xlsx"
    ):

        return pd.read_excel(
            file_path
        )


    else:

        raise ValueError(
            "Only .csv and .xlsx "
            "files are supported."
        )


# ==========================================================
# LOCAL QWEN MODEL
# ==========================================================

llm = LiteLLMModel(

    model_id=(
        "ollama/"
        "qwen2.5-coder:latest"
    ),

    api_base=(
        "http://localhost:11434"
    )
)


# ==========================================================
# AGENT
# ==========================================================

agent = CodeAgent(

    tools=[
        get_claim_by_id,
        predict_fraud
    ],

    model=llm,

    max_steps=4
)


# ==========================================================
# LOAD TEST DATA
# ==========================================================

claims_df = load_claims(
    "claims.csv"
)


# ==========================================================
# TEST QUESTION
# ==========================================================

question = (
    "Is claim 233 fraudulent? "
    "Use the trained fraud model."
)


result = agent.run(
    question
)


print(
    "\nFINAL ANSWER:"
)

print(result)