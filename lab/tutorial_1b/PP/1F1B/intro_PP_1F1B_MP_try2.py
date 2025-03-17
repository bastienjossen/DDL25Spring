import os
import torch
import torch.distributed as dist
from torch.optim import Adam
from sys import argv

from simplellm.llama import LLamaFirstStage, LLamaStage, LLamaLastStage
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.losses import causalLLMLoss

# Adjust to 6 total ranks
rank = int(argv[1])
world_size = 6
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)

# 2 pipelines, each has 3 stages
pipeline_id = rank // 3  # 0 or 1
stage_id = rank % 3      # 0, 1, or 2

dmodel = 288
num_heads = 6
seq_l = 256
batch_size = 3
num_microbatches = 3
n_layers = 6 // 3  # 2 layers per stage for the pipeline
device = "cpu"

# Create stage-based data-parallel groups:
if stage_id == 0:
    dp_group = dist.new_group([0, 3])
elif stage_id == 1:
    dp_group = dist.new_group([1, 4])
elif stage_id == 2:
    dp_group = dist.new_group([2, 5])

# Instantiate model and dataset
if stage_id == 0:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l, skip=pipeline_id * 3000)
    data_iter = iter(ds)
elif stage_id == 1:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads, device=device,
                     n_layers=n_layers, ctx_size=seq_l)
elif stage_id == 2:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l, skip=pipeline_id * 3000)
    data_iter = iter(ds)

optimizer = Adam(net.parameters(), lr=8e-4)

for iteration in range(50):
    optimizer.zero_grad()

    # -----------------------
    # FORWARD PIPELINE
    # -----------------------
    if stage_id == 0:
        full_batch = next(data_iter).to(device)
        embedded = net.embed(full_batch)
        # Chunk into microbatches
        microbatches = torch.chunk(embedded, num_microbatches, dim=0)
        # Send each microbatch to stage 1
        for micro in microbatches:
            dist.send(micro.cpu(), dst=rank + 1)

    elif stage_id == 1:
        stored_activations = []
        microbatches_in = []
        for _ in range(num_microbatches):
            recv_micro = torch.empty((batch_size // num_microbatches, seq_l, dmodel))
            dist.recv(recv_micro, src=rank - 1)  # from stage 0
            microbatches_in.append(recv_micro)
        for micro in microbatches_in:
            micro = micro.to(device)
            micro.requires_grad_()
            out = net(micro)
            stored_activations.append((micro, out))
            dist.send(out.cpu(), dst=rank + 1)  # to stage 2

    elif stage_id == 2:
        microbatches_in = []
        for _ in range(num_microbatches):
            recv_micro = torch.empty((batch_size // num_microbatches, seq_l, dmodel))
            dist.recv(recv_micro, src=rank - 1)  # from stage 1
            microbatches_in.append(recv_micro)
        # Retrieve target batch
        full_target = next(data_iter).to(device)
        target_microbatches = torch.chunk(full_target, num_microbatches, dim=0)
        # Compute loss & backprop for each microbatch
        for micro, target_micro in zip(microbatches_in, target_microbatches):
            micro = micro.to(device)
            micro.requires_grad_()
            logits = net(micro)
            loss = causalLLMLoss(logits, target_micro, tokenizer.vocab_size) / num_microbatches
            print(f"Rank {rank} Iter {iteration} Loss: {loss.item()}")
            loss.backward()
        # Send input gradient to stage 1
        for micro in microbatches_in:
            dist.send(micro.grad.cpu(), dst=rank - 1)

    # -----------------------
    # BACKWARD PIPELINE
    # -----------------------
    if stage_id == 1:
        microbatches_grads = []
        for _ in range(num_microbatches):
            grad_in = torch.empty((batch_size // num_microbatches, seq_l, dmodel))
            dist.recv(grad_in, src=rank + 1)  # from stage 2
            microbatches_grads.append(grad_in)
        # For each microbatch, do backward and propagate gradient
        for (inp, out), grad_in in zip(stored_activations, microbatches_grads):
            out.backward(grad_in.to(device))
            dist.send(inp.grad.cpu(), dst=rank - 1)

    elif stage_id == 0:
        microbatches_grads = []
        for _ in range(num_microbatches):
            grad_in = torch.empty((batch_size // num_microbatches, seq_l, dmodel))
            dist.recv(grad_in, src=rank + 1)  # from stage 1
            microbatches_grads.append(grad_in)
        # If you saved embedded outputs, chunk them and backward
        # Possibly you stored 'embedded' from above:
        microbatches_emb = torch.chunk(embedded, num_microbatches, dim=0)
        for emb, grad_in in zip(microbatches_emb, microbatches_grads):
            emb.backward(grad_in.to(device))

    # -----------------------
    # DATA-PARALLEL ALL-REDUCE (across the same stage in the 2 pipelines)
    # -----------------------
    for param in net.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=dp_group)
            param.grad /= 2.0

    optimizer.step()

dist.destroy_process_group()
print(f"Rank {rank} done.")