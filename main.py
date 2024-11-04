from utils import set_seed, get_device
from multiprocessing import cpu_count
from dataset import EliWillWatts, ScrappedWatts
from datasets import concatenate_datasets
from watts import Watts
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


SEED = 14
set_seed(SEED)

print("Loading dataset...")
ds1 = EliWillWatts().ds
ds2 = ScrappedWatts().ds
train_ds = concatenate_datasets([ds1["train"], ds2["train"]]).shuffle(SEED)
val_ds = concatenate_datasets([ds1["validation"], ds2["validation"]]).shuffle(SEED)


# model_name = "openai-community/gpt2"
model_name = "distilbert/distilgpt2"
print("Loading model", model_name)

device = get_device()
watts = Watts(model_name, device)
tokenizer = watts.tokenizer

# utils


def tokenize_function(tokenizer, example):
    example["text"] = [
        line for line in example["text"] if len(line) > 0 and not line.isspace()
    ]  # batched
    tokens = tokenizer(
        example["text"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    ids = tokens["input_ids"]

    return {
        "input_ids": ids[:, :-1].numpy(),
        "labels": ids[:, 1:].numpy(),
        "attention_mask": tokens["attention_mask"][:, 1:].numpy(),
    }


# Configs

EPOCHS = 2
BATCH_SIZE = 2
LR = 5e-5  # very small lr since we're fine-tuning
GRAD_ACC = 8  # effectively making batch_size=(GRAD_ACC * BATCH_SIZE)


# Training

print("Tokenizing dataset...")
train_ds = train_ds.map(
    lambda x: tokenize_function(tokenizer, x),
    batched=True,
    num_proc=int(0.75 * cpu_count()),
    remove_columns=["text"],
)
val_ds = val_ds.map(
    lambda x: tokenize_function(tokenizer, x),
    batched=True,
    num_proc=int(0.75 * cpu_count()),
    remove_columns=["text"],
)

training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_dir="logs",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_steps=1000,
    output_dir="checkpoints",
    overwrite_output_dir=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=watts.model,
    args=training_args,
    data_collator=collator,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=lambda pred: {"loss": pred.loss},
)
print("Training...")
trainer.train()

prompt = "Write a short story in the style of Alan Watts: "
print(watts.generate_text(prompt, max_output_length=200))
