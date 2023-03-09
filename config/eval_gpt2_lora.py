# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'resume'
from functools import partial

import torch
from minlora import LoRAParametrization
fast_attention = False

use_lora = True
learning_rate = 1e-3 # use a higher LR for LoRA
lora_dropout_p = 0.0
rank=4
lora_alpha = 64
lora_config = {
    torch.nn.Embedding: {
        "weight": partial(LoRAParametrization.from_embedding, rank=rank, lora_alpha=lora_alpha),
    },
    torch.nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=lora_alpha),
    },
}