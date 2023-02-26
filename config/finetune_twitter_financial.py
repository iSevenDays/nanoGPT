import time

out_dir = 'out-twitter_financial'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'twitter_financial'
wandb_run_name = 'test4'

dataset = 'twitter_financial'
init_from = 'resume' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 256 tokens = 8,192 tokens/iter
# shakespeare has 419,979 tokens, so 1 epoch ~= 51.266 iters
batch_size = 1
block_size = 256

gradient_accumulation_steps = 32
max_iters = 5000

# finetune at constant LR
learning_rate = 1e-6
decay_lr = True
warmup_iters = 200
lr_decay_iters = max_iters 
min_lr = learning_rate/10

device = 'mps'
compile=False
