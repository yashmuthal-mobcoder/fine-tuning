from mlx_lm import load

model, tokenizer = load("qwen-medical")

print("Model loaded successfully")
print(type(model))