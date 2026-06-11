from mlx_lm import load, generate

MODEL_PATH = "qwen-medical-4bit"
ADAPTER_PATH = "medical_adapter_v2"

SYSTEM_PROMPT = """
            "content": "You are a helpful medical assistant. Give your answers in a structured JSON format with the following schema: { \"<condition_name>\": { \"definition\": \"<string>\", \"types\": [ { \"name\": \"<string>\", \"description\": \"<string>\" } ], \"causes\": [\"<string>\"], \"symptoms\": [\"<string>\"], \"diagnosis\": [\"<string>\"], \"treatment_options\": [\"<string>\"] } }"      },

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