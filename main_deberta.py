from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from config import get_tokenizer, compute_metrics
from utils import load_dataset

model_name = "microsoft/deberta-v3-base"
tokenizer = get_tokenizer(model_name)
dataset = load_dataset("dataset/train.csv")

def preprocess_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(preprocess_function).remove_columns(["text", "label"]).rename_column("label_id", "labels")
tokenized.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

args = TrainingArguments(
    output_dir="results_deberta",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="logs_deberta",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()
trainer.evaluate()
