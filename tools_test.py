from smolagents import CodeAgent, LiteLLMModel, tool


@tool
def claim_risk_score(claim_amount: float, previous_claims: int) -> str:
    """
    Calculates a simple example insurance claim risk score.

    Args:
        claim_amount: The amount of the insurance claim.
        previous_claims: Number of previous claims made by the patient.

    Returns:
        A text description of the estimated risk.
    """

    score = 0

    if claim_amount > 10000:
        score += 2

    if previous_claims > 5:
        score += 2

    if score >= 4:
        return "High risk"

    elif score >= 2:
        return "Medium risk"

    return "Low risk"


model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:latest",
    api_base="http://localhost:11434",
)


agent = CodeAgent(
    tools=[claim_risk_score],
    model=model,
    max_steps=4,
)


result = agent.run(
    """
    A patient submitted an insurance claim for 18,000 EGP
    and has made 7 previous claims.

    Determine the claim risk using the available tool.
    """
)

print("\nFINAL ANSWER:")
print(result)