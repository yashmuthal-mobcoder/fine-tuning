from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import torch
from transformers import DefaultDataCollator

from peft import (
    LoraConfig,
    get_peft_model,
)



MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"



train_dataset = load_from_disk("data/train_dataset")
eval_dataset = load_from_disk("data/eval_dataset")

train_dataset = train_dataset.select(range(500))
eval_dataset= eval_dataset.select(range(100))

print("Train:", len(train_dataset))
print("Eval:", len(eval_dataset))


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True,
)



lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)



model = get_peft_model(
    model,
    lora_config,
)

model.print_trainable_parameters()


data_collator = DefaultDataCollator()


training_args = TrainingArguments(
    output_dir="medical-qwen-lora",

    num_train_epochs=1,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=1e-4,
    weight_decay=0.01,

    logging_steps=10,

    eval_strategy="steps",
    eval_steps=1000,

    save_strategy="steps",
    save_steps=1000,

    save_total_limit=2,

    report_to="none",
    max_grad_norm=1.0,
    
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)


trainer.train()


model.save_pretrained("medical_qwen_lora")
tokenizer.save_pretrained("medical_qwen_lora")

print("Training Complete")