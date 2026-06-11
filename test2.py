import time
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "medical_qwen_lora"

eval_dataset = load_from_disk("data/eval_dataset")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
).to("mps")

model.eval()

total_loss = 0.0
count = 0

start_time = time.time()

print(f"Evaluating {len(eval_dataset)} samples...\n")

with torch.no_grad():
    for idx, sample in enumerate(eval_dataset):

        inputs = {
            "input_ids": torch.tensor(
                [sample["input_ids"]],
                device="mps"
            ),
            "attention_mask": torch.tensor(
                [sample["attention_mask"]],
                device="mps"
            ),
            "labels": torch.tensor(
                [sample["labels"]],
                device="mps"
            ),
        }

        outputs = model(**inputs)

        loss = outputs.loss.item()

        total_loss += loss
        count += 1

        if count % 50 == 0:
            avg_so_far = total_loss / count

            elapsed = time.time() - start_time

            print(
                f"[{count}/{len(eval_dataset)}] "
                f"Current Loss: {loss:.4f} | "
                f"Running Avg: {avg_so_far:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

avg_loss = total_loss / count
    
print("\n" + "=" * 50)
print("FINE-TUNED MODEL RESULTS")
print("=" * 50)
print(f"Samples Evaluated: {count}")
print(f"Final Eval Loss: {avg_loss:.4f}")   