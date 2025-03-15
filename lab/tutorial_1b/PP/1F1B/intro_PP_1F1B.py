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
        out = next(iter_ds)
        out = out.to(device)
        out = net.embed(out)

        dist.send(out.to("cpu"),1)

    elif rank == 1:

        inp_batch = torch.empty((batch_size,seq_l,dmodel))
        dist.recv(inp_batch,0)
        with torch.no_grad():
            inp_batch = inp_batch.to(device)
            inp_batch.requires_grad_()
            inp_batch.retain_grad()

        out = net(inp_batch)
        dist.send(out.to("cpu"),2)

    elif rank == 2:
        target = next(iter_ds)
        inp_batch = torch.empty((batch_size,seq_l,dmodel))
        dist.recv(inp_batch,1)
        with torch.no_grad():
            inp_batch = inp_batch.to(device)
            inp_batch.requires_grad_()
            inp_batch.retain_grad()

        logits = net(inp_batch)
        loss = causalLLMLoss(logits,target,tokenizer.vocab_size)
        print(loss.item())
        loss.backward()


    # BACKWARD PASS:
    if rank == 2:
        dist.send(inp_batch.grad.to("cpu"),1)
    elif rank == 1:
        inp_grad = torch.empty((batch_size,seq_l,dmodel))
        dist.recv(inp_grad,2)
        out.backward(inp_grad.to(device))
        dist.send(inp_batch.grad.to("cpu"),0)
    elif rank == 0:
        inp_grad = torch.empty((batch_size,seq_l,dmodel))
        dist.recv(inp_grad,1)
        out.backward(inp_grad.to(device))


    optim.step()
    torch.cuda.empty_cache()
