from mlx_lm import load, generate

MODEL_PATH = "qwen-medical-4bit"
ADAPTER_PATH = "medical_adapter_v2"

SYSTEM_PROMPT = """
You are a helpful medical assistant.
Provide accurate and educational medical information.
"""

model, tokenizer = load(
    MODEL_PATH,
    adapter_path=ADAPTER_PATH,
)

while True:
    question = input("\nYou: ")

    if question.lower() in ["exit", "quit"]:
        break

    prompt = f"""
System: {SYSTEM_PROMPT}

User: {question}

Assistant:
"""

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=300,
        verbose=False,
    )

    print("\nAssistant:")
    print(response)