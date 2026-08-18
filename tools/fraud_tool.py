import os
import joblib
import pandas as pd
import shap


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


# ==========================================================
# LOAD TRAINED OBJECTS
# ==========================================================

model = joblib.load(
    os.path.join(MODEL_DIR, "fraud_model.pkl")
)

encoder = joblib.load(
    os.path.join(MODEL_DIR, "encoder.pkl")
)

categorical_columns = list(
    joblib.load(
        os.path.join(MODEL_DIR, "categorical_columns.pkl")
    )
)

non_categorical_columns = list(
    joblib.load(
        os.path.join(MODEL_DIR, "non_categorical_columns.pkl")
    )
)

model_features = list(
    joblib.load(
        os.path.join(MODEL_DIR, "model_features (1).pkl")
    )
)


# ==========================================================
# SHAP EXPLAINER
# ==========================================================

explainer = shap.TreeExplainer(model)


# ==========================================================
# SETTINGS
# ==========================================================

THRESHOLD = 0.27

CORE_REQUIRED_COLUMNS = [
    "Claim_Amount",
    "Provider_Type",
    "Provider_Specialty",
    "Provider_Patient_Distance_Miles",
]


# ==========================================================
# HELPER: DEFAULT VALUE
# ==========================================================

def get_default_value(column):
    """
    Returns a safe fallback value for an optional feature.
    """

    if column in categorical_columns:

        column_index = categorical_columns.index(column)
        known_categories = encoder.categories_[column_index]

        if len(known_categories) > 0:
            return known_categories[0]

        return ""

    return 0


# ==========================================================
# PREPARE CLAIM
# ==========================================================

def prepare_claim_for_model(claim):
    """
    Converts one claim row into the exact feature format
    required by the trained Random Forest.
    """

    claim = claim.copy()

    # ------------------------------------------------------
    # 1. CHECK FOUR ESSENTIAL FEATURES
    # ------------------------------------------------------

    missing_core = []

    for column in CORE_REQUIRED_COLUMNS:

        if column not in claim.index:
            missing_core.append(column)
        elif pd.isna(claim[column]):
            missing_core.append(column)

    if missing_core:
        raise ValueError(
            "Cannot make a fraud prediction because "
            "essential data is missing: "
            + ", ".join(missing_core)
        )

    # ------------------------------------------------------
    # 2. FEATURE ENGINEERING FROM DATES WHEN AVAILABLE
    # ------------------------------------------------------

    if (
        "Days_Service_to_Claim" not in claim.index
        or pd.isna(claim.get("Days_Service_to_Claim"))
    ):
        if (
            "Claim_Date" in claim.index
            and "Service_Date" in claim.index
            and pd.notna(claim["Claim_Date"])
            and pd.notna(claim["Service_Date"])
        ):
            claim_date = pd.to_datetime(claim["Claim_Date"])
            service_date = pd.to_datetime(claim["Service_Date"])
            claim["Days_Service_to_Claim"] = (claim_date - service_date).days

    if (
        "Days_Until_Policy_Expiration" not in claim.index
        or pd.isna(claim.get("Days_Until_Policy_Expiration"))
    ):
        if (
            "Policy_Expiration_Date" in claim.index
            and "Claim_Date" in claim.index
            and pd.notna(claim["Policy_Expiration_Date"])
            and pd.notna(claim["Claim_Date"])
        ):
            expiration_date = pd.to_datetime(claim["Policy_Expiration_Date"])
            claim_date = pd.to_datetime(claim["Claim_Date"])
            claim["Days_Until_Policy_Expiration"] = (
                expiration_date - claim_date
            ).days

    if (
        "Claim_Month" not in claim.index
        or pd.isna(claim.get("Claim_Month"))
    ):
        if "Claim_Date" in claim.index and pd.notna(claim["Claim_Date"]):
            claim["Claim_Month"] = pd.to_datetime(claim["Claim_Date"]).month

    # ------------------------------------------------------
    # 3. FIND ALL RAW INPUT FEATURES EXPECTED
    # ------------------------------------------------------

    required_model_inputs = categorical_columns + non_categorical_columns

    # ------------------------------------------------------
    # 4. BUILD INPUT
    # ------------------------------------------------------

    input_values = {}
    used_features = []
    imputed_features = []

    for column in required_model_inputs:

        if column in claim.index and pd.notna(claim[column]):
            input_values[column] = claim[column]
            used_features.append(column)
        else:
            input_values[column] = get_default_value(column)
            imputed_features.append(column)

    input_data = pd.DataFrame([input_values])

    # ------------------------------------------------------
    # 5. FIX DATA TYPES
    # ------------------------------------------------------

    if "Procedure_Code" in input_data.columns:
        input_data["Procedure_Code"] = input_data["Procedure_Code"].astype(str)

    if "Claim_Submitted_Late" in input_data.columns:
        input_data["Claim_Submitted_Late"] = input_data[
            "Claim_Submitted_Late"
        ].astype(int)

    # ------------------------------------------------------
    # 6. ONE-HOT ENCODE CATEGORICAL FEATURES
    # ------------------------------------------------------

    encoded_values = encoder.transform(input_data[categorical_columns])

    if hasattr(encoded_values, "toarray"):
        encoded_values = encoded_values.toarray()

    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    encoded_df = pd.DataFrame(
        encoded_values,
        columns=encoded_columns,
        index=input_data.index
    )

    # ------------------------------------------------------
    # 7. KEEP NON-CATEGORICAL FEATURES
    # ------------------------------------------------------

    numeric_df = input_data[non_categorical_columns].copy()

    # ------------------------------------------------------
    # 8. COMBINE
    # ------------------------------------------------------

    final_input = pd.concat([numeric_df, encoded_df], axis=1)

    # ------------------------------------------------------
    # 9. EXACT SAME FEATURE ORDER AS TRAINING
    # ------------------------------------------------------

    final_input = final_input.reindex(columns=model_features, fill_value=0)

    return final_input, used_features, imputed_features


# ==========================================================
# EXPLAINABILITY
# ==========================================================

def get_fraud_reasons(final_input, top_n=5):
    """
    Returns the top features driving the fraud prediction,
    aggregated back to original claim fields (not one-hot columns).
    """

    raw_shap = explainer.shap_values(final_input)

    if isinstance(raw_shap, list):
        class_values = raw_shap[1][0]
    elif raw_shap.ndim == 3:
        class_values = raw_shap[0, :, 1]
    else:
        class_values = raw_shap[0]

    contributions = {}

    for col, val in zip(final_input.columns, class_values):

        matched_original = None

        for cat_col in categorical_columns:
            if col.startswith(cat_col + "_"):
                matched_original = cat_col
                break

        key = matched_original or col
        contributions[key] = contributions.get(key, 0.0) + float(val)

    ranked = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    return [
        {
            "feature": name,
            "impact": round(val, 4),
            "direction": "increases fraud risk" if val > 0 else "decreases fraud risk"
        }
        for name, val in ranked
    ]


# ==========================================================
# PREDICT ONE CLAIM
# ==========================================================

def predict_claim_row(claim):
    """
    Predicts fraud probability for one pandas Series claim.
    """

    final_input, used_features, imputed_features = prepare_claim_for_model(claim)

    fraud_probability = float(model.predict_proba(final_input)[0][1])
    flagged = fraud_probability >= THRESHOLD

    reasons = get_fraud_reasons(final_input)

    return {
        "fraud_probability": fraud_probability,
        "fraud_percentage": round(fraud_probability * 100, 2),
        "threshold": THRESHOLD,
        "flagged": bool(flagged),
        "features_used": used_features,
        "features_imputed": imputed_features,
        "number_features_used": len(used_features),
        "number_features_imputed": len(imputed_features),
        "top_reasons": reasons,
    }