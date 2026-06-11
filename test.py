import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "medical_qwen_lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype="auto",
)


model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model = model.to("mps")

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical assistant. Give your answers in a structured JSON format with the following schema: { \"<condition_name>\": { \"definition\": \"<string>\", \"types\": [ { \"name\": \"<string>\", \"description\": \"<string>\" } ], \"causes\": [\"<string>\"], \"symptoms\": [\"<string>\"], \"diagnosis\": [\"<string>\"], \"treatment_options\": [\"<string>\"] } }"      },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    inputs = {k: v.to("mps") for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

    response = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nAssistant:")
    print(response)