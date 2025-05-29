from unsloth import FastLanguageModel
import torch

# model setup (unchanged)
max_seq_length = 2048
dtype            = None
load_in_4bit     = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = "unsloth/Phi-4",
    max_seq_length   = max_seq_length,
    dtype            = dtype,
    load_in_4bit     = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                         = 128,
    target_modules           = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",
                                "embed_tokens", "lm_head"],
    lora_alpha               = 32,
    lora_dropout             = 0,
    bias                      = "none",
    use_gradient_checkpointing = "unsloth",
    random_state             = 3407,
    use_rslora               = True,
    loftq_config             = None,
)

# ——— Load from your JSON file instead ———
from datasets import load_dataset

# point this to wherever you’ve saved data.json
dataset = load_dataset(
    "json",
    data_files = { "train": r"C:\Users\camer\Downloads\vuln\data.json" },
    split      = "train"
)

# drop any empty or bogus examples
dataset = dataset.filter(
    lambda ex: ex["instruction"].strip() and ex["output"].strip()
)

# attach EOS and format as a single "text" field for LM training
EOS_TOKEN = tokenizer.eos_token

def format_for_sft(examples):
    out = []
    for ins, outp in zip(examples["instruction"], examples["output"]):
        txt = (
            "### Instruction:\n" + ins.strip() + "\n"
            "### Response:\n"  + outp.strip() + EOS_TOKEN
        )
        out.append(txt)
    return { "text": out }

dataset = dataset.map(
    format_for_sft,
    batched        = True,
    remove_columns = [c for c in dataset.column_names if c not in ("instruction", "output")]
)

print(f"Number of SFT examples: {len(dataset)}")

# ——— trainer setup (unchanged) ———
from transformers import DataCollatorForLanguageModeling
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm       = False,
)

trainer = UnslothTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = dataset,
    data_collator      = data_collator,
    dataset_text_field = "text",
    max_seq_length     = max_seq_length,
    dataset_num_proc   = 2,
    args = UnslothTrainingArguments(
        per_device_train_batch_size   = 2,
        gradient_accumulation_steps   = 8,
        num_train_epochs              = 4,
        warmup_steps                  = 10,
        learning_rate                 = 5e-5,
        embedding_learning_rate       = 1e-5,
        fp16                          = not is_bfloat16_supported(),
        bf16                          = is_bfloat16_supported(),
        logging_steps                 = 1,
        optim                         = "adamw_8bit",
        weight_decay                  = 0.01,
        lr_scheduler_type             = "linear",
        seed                          = 3407,
        output_dir                    = "outputs",
        report_to                     = "none",
        save_strategy                 = "steps",
        save_steps                    = 50,
        save_total_limit              = 3,
    ),
)

trainer_stats = trainer.train()
