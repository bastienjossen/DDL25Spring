from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from simplellm.losses import causalLLMLoss # our loss
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torch
import torch.distributed as dist
import os
from sys import argv
rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 3
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // world_size
seq_l = 256
batch_size = 3
num_microbatches = 3
device = "mps"

# make the tokenizer

# make the model
if rank == 0:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l) # no skip
    iter_ds = iter(ds)
elif rank == 1:
    net = LLamaStage(dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
elif rank == 2:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l) # no skip
    iter_ds = iter(ds)



optim = Adam(net.parameters(),lr=8e-4)

for itr in range(5_000):
    optim.zero_grad()
    # FORWARD PASS:
    if rank == 0:
        full_batch = next(iter_ds).to(device)
        embedded = net.embed(full_batch)
        microbatches = torch.chunk(embedded, num_microbatches, dim=0)
        for micro in microbatches:
            req = dist.isend(tensor=micro.to("cpu"), dst=1)
            req.wait()

    elif rank == 1:

        micro_batch_size = batch_size // num_microbatches
        microbatches_in = []
        for _ in range(num_microbatches):
            recv_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(tensor=recv_tensor, src=0)
            req.wait()
            microbatches_in.append(recv_tensor)
        processed_out = []
        for micro in microbatches_in:
            micro = micro.to(device)
            micro.requires_grad_()
            micro.retain_grad()
            out_micro = net(micro)
            processed_out.append(out_micro)
        for out_micro in processed_out:
            req = dist.isend(tensor=out_micro.to("cpu"), dst=2)
            req.wait()

    elif rank == 2:
        micro_batch_size = batch_size // num_microbatches
        microbatches_in = []
        processed_out_2 = []  # store outputs for backward pass
        for _ in range(num_microbatches):
            recv_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(tensor=recv_tensor, src=1)
            req.wait()
            microbatches_in.append(recv_tensor)
        full_target = next(iter_ds).to(device)
        target_microbatches = torch.chunk(full_target, num_microbatches, dim=0)
        for micro, target_micro in zip(microbatches_in, target_microbatches):
            micro = micro.to(device)
            micro.requires_grad_()
            micro.retain_grad()
            logits = net(micro)
            loss = causalLLMLoss(logits, target_micro, tokenizer.vocab_size) / num_microbatches
            print(loss.item())
            loss.backward()
            processed_out_2.append(micro)  # or store the net output if needed
        # send gradients back
        for out_micro in processed_out_2:
            req = dist.isend(tensor=out_micro.grad.to("cpu"), dst=1)
            req.wait()


    # BACKWARD PASS:
    if rank == 1:
        micro_batch_size = batch_size // num_microbatches
        microbatches_grads = []
        for _ in range(num_microbatches):
            grad_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(tensor=grad_tensor, src=2)
            req.wait()
            microbatches_grads.append(grad_tensor)
        for out_micro, grad_tensor in zip(processed_out, microbatches_grads):
            out_micro.backward(grad_tensor.to(device))
        for micro in microbatches_in:
            # make sure you're sending the gradient from 'micro', not 'recv_tensor'
            req = dist.isend(tensor=micro.grad.to("cpu"), dst=0)
            req.wait()
    elif rank == 0:
        micro_batch_size = batch_size // num_microbatches
        microbatches_grads = []
        for _ in range(num_microbatches):
            grad_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(tensor=grad_tensor, src=1)
            req.wait()
            microbatches_grads.append(grad_tensor)
        for micro, grad_tensor in zip(torch.chunk(net.embed(full_batch), num_microbatches, dim=0), microbatches_grads):
            micro.backward(grad_tensor.to(device))


    optim.step()
    torch.cuda.empty_cache()
