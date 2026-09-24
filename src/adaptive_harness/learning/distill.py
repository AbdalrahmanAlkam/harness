"""Export verified traces and a local LoRA training scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from adaptive_harness.learning.harvester import harvest_verified_traces


MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

TRAIN_SCRIPT = '''"""Train a local adapter from verified harness examples.

Install optional packages: pip install transformers datasets peft accelerate torch
Run from this directory: python train_lora.py
"""
import json
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

root = Path(__file__).resolve().parent
cfg = json.loads((root / "lora_config.json").read_text())
dataset = load_dataset("json", data_files=str(root / "instruction_dataset.jsonl"), split="train")
tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def encode(row):
    text = ("Instruction: " + row["instruction"] + "\\nContext: " + row["input"] +
            "\\nResponse: " + row["output"] + tokenizer.eos_token)
    return tokenizer(text, truncation=True, max_length=cfg["max_length"])

tokenized = dataset.map(encode, remove_columns=dataset.column_names)
model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])
model = get_peft_model(model, LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"],
    lora_dropout=cfg["dropout"], task_type="CAUSAL_LM"))
trainer = Trainer(model=model, train_dataset=tokenized,
    args=TrainingArguments(output_dir=str(root / "adapter"), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"], logging_steps=10, save_strategy="epoch",
        report_to="none"),
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
trainer.train()
model.save_pretrained(root / "adapter")
tokenizer.save_pretrained(root / "adapter")
'''


def export_distillation(db_path: str | Path, output_dir: str | Path,
                        model: str = "qwen2.5-0.5b") -> int:
    if model not in MODELS:
        raise ValueError(f"Choose one of: {', '.join(MODELS)}")
    examples = harvest_verified_traces(db_path)
    if not examples:
        raise ValueError("No verified developer traces are available for distillation")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "instruction_dataset.jsonl"
    with dataset.open("w", encoding="utf-8") as stream:
        for item in examples:
            record = {"instruction": item["user_prompt"],
                      "input": "Complete the task with tools and report only verified results.",
                      "output": json.dumps({"trajectory_steps": item["trajectory_steps"],
                                            "final_solution": item["final_solution"]}, ensure_ascii=False)}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    config = {"base_model": MODELS[model], "rank": 8, "alpha": 16,
              "dropout": 0.05, "max_length": 2048, "epochs": 3, "learning_rate": 0.0002}
    (root / "lora_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (root / "train_lora.py").write_text(TRAIN_SCRIPT, encoding="utf-8")
    return len(examples)
