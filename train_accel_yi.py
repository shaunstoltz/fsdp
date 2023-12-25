import transformers
import functools
import torch
from datasets import load_from_disk

from datetime import datetime

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig, ShardingStrategy, CPUOffload, MixedPrecision

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        LlamaDecoderLayer,
    },
)


fsdp_plugin = FullyShardedDataParallelPlugin(
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
    mixed_precision_policy=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
    ),
    backward_prefetch=None,
    param_init_fn=None,
    #state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    #optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)#

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig

base_model_id = "01-ai/Yi-6B"

model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_eos_token=True,
    add_bos_token=True, 
)

model = accelerator.prepare_model(model)

if torch.cuda.device_count() > 1: # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True


lm_datasets = load_from_disk("maths_prompt")

eval_dataset = lm_datasets["validation"]
train_dataset = lm_datasets["train"]

project = "finetune"
base_model_name = "yi34b"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=eval_dataset,
    eval_dataset=train_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=1,
        #gradient_checkpointing=True,
        gradient_accumulation_steps=16,
        max_steps=1000,
        learning_rate=2.5e-5, 
        logging_steps=25,
        fp16=True, 
        #optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=50,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        #report_to="wandb",           # Comment this out if you don't want to use weights & baises
        #run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
trainer.save_model() 