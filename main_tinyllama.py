from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from config import get_tokenizer, compute_metrics
from utils import load_dataset

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = get_tokenizer(model_name)
dataset = load_dataset("dataset/train.csv")

def preprocess_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(preprocess_function).remove_columns(["text", "label"]).rename_column("label_id", "labels")
tokenized.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.config.pad_token_id = tokenizer.pad_token_id

args = TrainingArguments(
    output_dir="results_tinyllama",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    num_train_epochs=2,
    learning_rate=3e-5,
    logging_dir="logs_tinyllama",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
