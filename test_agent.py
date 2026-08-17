from smolagents import CodeAgent, LiteLLMModel


model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:latest",
    api_base="http://localhost:11434",
)


agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=3,
)


result = agent.run(
    "What is 37 multiplied by 19? Explain briefly."
)

print("\nFINAL ANSWER:")
print(result)