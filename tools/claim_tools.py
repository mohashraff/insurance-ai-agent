import pandas as pd
from smolagents import tool

from tools.fraud_tool import (
    predict_claim_row,
    CORE_REQUIRED_COLUMNS
)


def make_claim_tools(get_df):
    """
    Builds get_claim_by_id and predict_fraud tools bound to
    whatever dataframe source you give it.

    get_df: a zero-arg callable that returns the current claims
    DataFrame. Pass `lambda: st.session_state.claims_df` in
    Streamlit, or `lambda: claims_df` for a plain global.
    """

    @tool
    def get_claim_by_id(claim_id: int) -> dict:
        """
        Retrieves all available information for a specific
        insurance claim from the loaded claims dataset.

        Use this tool when the user asks to view, inspect,
        describe, or retrieve a particular claim.

        Args:
            claim_id: Claim_ID to retrieve.

        Returns:
            A dictionary containing all available claim data.
        """

        df = get_df()

        if df is None:
            return {"error": "No claims dataset is loaded."}

        if "Claim_ID" not in df.columns:
            return {"error": "The dataset does not contain Claim_ID."}

        result = df[df["Claim_ID"] == claim_id]

        if result.empty:
            return {"error": f"Claim {claim_id} was not found."}

        claim = result.iloc[0]

        output = {}

        for key, value in claim.items():

            if pd.isna(value):
                output[key] = None
            elif hasattr(value, "item"):
                try:
                    output[key] = value.item()
                except Exception:
                    output[key] = str(value)
            else:
                output[key] = value

        return output

    @tool
    def predict_fraud(claim_id: int) -> dict:
        """
        Runs the trained machine-learning fraud model on a
        specific claim and explains the top contributing factors.

        Use this tool whenever the user asks whether a claim
        is fraudulent, suspicious, high-risk, flagged, or asks
        for its fraud probability or the reasons behind it.

        Args:
            claim_id: Claim_ID to analyze.

        Returns:
            Fraud probability, review threshold, model decision,
            top contributing reasons, and feature availability info.
        """

        df = get_df()

        if df is None:
            return {"error": "No claims dataset is loaded."}

        if "Claim_ID" not in df.columns:
            return {"error": "The dataset does not contain Claim_ID."}

        result = df[df["Claim_ID"] == claim_id]

        if result.empty:
            return {"error": f"Claim {claim_id} was not found."}

        claim = result.iloc[0]

        try:
            prediction = predict_claim_row(claim)
        except Exception as e:
            return {"error": str(e), "claim_id": claim_id}

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
            "top_reasons": prediction["top_reasons"],
        }

    return get_claim_by_id, predict_fraud