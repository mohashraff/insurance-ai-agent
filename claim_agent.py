import pandas as pd
from smolagents import CodeAgent, LiteLLMModel, tool

claims_df = None


@tool
def get_claim_by_id(claim_id: int) -> str:
    """
    Finds a claim in the currently loaded claims dataset.

    Args:
        claim_id: The ID of the insurance claim to find.

    Returns:
        The claim information as text.
    """

    global claims_df

    if claims_df is None:
        return "No claims dataset is loaded."

    if "Claim_ID" not in claims_df.columns:
        return "The dataset does not contain a Claim_ID column."

    result = claims_df[claims_df["Claim_ID"] == claim_id]

    if result.empty:
        return f"Claim {claim_id} was not found."

    return result.iloc[0].to_json()


model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:latest",
    api_base="http://localhost:11434"
)


agent = CodeAgent(
    tools=[get_claim_by_id],
    model=model,
    max_steps=4
)

file_path = "synthetic_health_claims.csv"   # or claims.xlsx

if file_path.endswith(".csv"):
    claims_df = pd.read_csv(file_path)

elif file_path.endswith(".xlsx"):
    claims_df = pd.read_excel(file_path)

else:
    raise ValueError("Unsupported file type. Use CSV or XLSX.")
result = agent.run(
    "Show me the information for claim 233."
)

print("\nFINAL ANSWER:")
print(result)