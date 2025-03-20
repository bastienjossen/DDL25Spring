from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.losses import causalLLMLoss
from torch.optim import Adam
import torch
import torch.distributed as dist
import os
from sys import argv

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

rank = int(argv[1])
world_size = 6
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
torch.manual_seed(0)

dmodel = 288
num_heads = 6
n_layers = 6 // 3  
seq_l = 256
batch_size = 3
num_microbatches = 3
device = "cpu"

pipeline_id = rank // 3   # 0 or 1
stage_id = rank % 3       # 0, 1, or 2

if stage_id == 0:
    stage_group = dist.new_group(ranks=[0, 3])
elif stage_id == 1:
    stage_group = dist.new_group(ranks=[1, 4])
elif stage_id == 2:
    stage_group = dist.new_group(ranks=[2, 5])

if stage_id == 0:
    print(f"Rank {rank} (Stage 0): Initializing tokenizer and dataset", flush=True)
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l, skip=pipeline_id*3000)
    iter_ds = iter(ds)
elif stage_id == 1:
    print(f"Rank {rank} (Stage 1): Initializing model", flush=True)
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                     device=device, n_layers=n_layers, ctx_size=seq_l)
elif stage_id == 2:
    print(f"Rank {rank} (Stage 2): Initializing tokenizer and dataset", flush=True)
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l, skip=pipeline_id*3000)
    iter_ds = iter(ds)

optim = Adam(net.parameters(), lr=8e-4)

for itr in range(50):
    print(f"Rank {rank} Iter {itr}: Starting iteration", flush=True)
    optim.zero_grad()

    
    # FORWARD PASS (Forward Pipeline)
    if stage_id == 0:
        print(f"Rank {rank} (Stage 0): Starting forward pass", flush=True)
        full_batch = next(iter_ds).to(device)
        embedded = net.embed(full_batch)
        saved_embedded = embedded  
        microbatches = torch.chunk(embedded, num_microbatches, dim=0)
        # Batch asynchronous sends to stage 1.
        send_reqs = []
        for idx, micro in enumerate(microbatches):
            print(f"Rank {rank} (Stage 0): Sending microbatch {idx}", flush=True)
            send_reqs.append(dist.isend(tensor=micro.to("cpu"), dst=rank+1))
        for req in send_reqs:
            req.wait()
        print(f"Rank {rank} (Stage 0): Completed forward send", flush=True)

    elif stage_id == 1:
        print(f"Rank {rank} (Stage 1): Starting forward pass", flush=True)
        micro_batch_size = batch_size // num_microbatches
        stored_activations = []
        stored_gradients = []

        # First process initial microbatches (F phase only)
        for i in range(min(num_microbatches, 1)):  # Only 1 initial forward for 1F1B
            # Forward pass for microbatch i
            recv_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(tensor=recv_tensor, src=rank-1)
            req.wait()
            
            recv_tensor = recv_tensor.to(device)
            recv_tensor.requires_grad_()
            out = net(recv_tensor)
            stored_activations.append((recv_tensor, out))
            
            # Send output forward
            req = dist.isend(tensor=out.to("cpu"), dst=rank+1)
            req.wait()
            print(f"Rank {rank}: Forward pass completed for microbatch {i}")

        # Then alternate between F and B for remaining microbatches
        for i in range(1, num_microbatches):
            # Forward pass for microbatch i
            recv_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            req = dist.irecv(tensor=recv_tensor, src=rank-1)
            req.wait()
            
            recv_tensor = recv_tensor.to(device)
            recv_tensor.requires_grad_()
            out = net(recv_tensor)
            stored_activations.append((recv_tensor, out))
            
            # Send output forward
            req = dist.isend(tensor=out.to("cpu"), dst=rank+1)
            req.wait()
            print(f"Rank {rank}: Forward pass completed for microbatch {i}")
            
            # Backward pass for microbatch i-1
            grad_tensor = torch.empty_like(stored_activations[i-1][1])
            req = dist.irecv(tensor=grad_tensor, src=rank+1)
            req.wait()
            
            # Backward through the network
            stored_activations[i-1][1].backward(grad_tensor.to(device))
            
            # Send gradients backward
            req = dist.isend(tensor=stored_activations[i-1][0].grad.to("cpu"), dst=rank-1)
            req.wait()
            print(f"Rank {rank}: Backward pass completed for microbatch {i-1}")

        # Process remaining backward passes
        for i in range(num_microbatches-1, num_microbatches):
            # Backward pass for final microbatches
            grad_tensor = torch.empty_like(stored_activations[i][1])
            req = dist.irecv(tensor=grad_tensor, src=rank+1)
            req.wait()
            
            stored_activations[i][1].backward(grad_tensor.to(device))
            
            req = dist.isend(tensor=stored_activations[i][0].grad.to("cpu"), dst=rank-1)
            req.wait()
            print(f"Rank {rank}: Backward pass completed for microbatch {i}")

    elif stage_id == 2:
        print(f"Rank {rank} (Stage 2): Starting forward pass", flush=True)
        micro_batch_size = batch_size // num_microbatches
        recv_reqs = []
        microbatches_in = []
        for i in range(num_microbatches):
            recv_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            print(f"Rank {rank} (Stage 2): Receiving microbatch {i} from rank {rank-1}", flush=True)
            recv_reqs.append((dist.irecv(tensor=recv_tensor, src=rank-1), recv_tensor))
        for req, tensor in recv_reqs:
            req.wait()
            microbatches_in.append(tensor)
        full_target = next(iter_ds).to(device)
        target_microbatches = torch.chunk(full_target, num_microbatches, dim=0)
        processed_out_2 = [] 
        for i, (micro, target_micro) in enumerate(zip(microbatches_in, target_microbatches)):
            dmicro = micro.to(device)
            dmicro.requires_grad_()
            dmicro.retain_grad()
            print(f"Rank {rank} (Stage 2): Processing microbatch {i}", flush=True)
            logits = net(dmicro)
            loss = causalLLMLoss(logits, target_micro, tokenizer.vocab_size) / num_microbatches
            print(f"Rank {rank} Iter {itr} (Stage 2): Loss for microbatch {i} = {loss.item()}", flush=True)
            loss.backward()
            processed_out_2.append(dmicro)
        send_reqs = []
        for i, dmicro in enumerate(processed_out_2):
            print(f"Rank {rank} (Stage 2): Sending gradient for microbatch {i} to rank {rank-1}", flush=True)
            send_reqs.append(dist.isend(tensor=dmicro.grad.to("cpu"), dst=rank-1))
        for req in send_reqs:
            req.wait()
        print(f"Rank {rank} (Stage 2): Completed forward pass", flush=True)

    # -----------------------
    # BACKWARD PASS (Reverse Pipeline)
    # -----------------------
    if stage_id == 1:
        print(f"Rank {rank} (Stage 1): Starting backward pass", flush=True)
        micro_batch_size = batch_size // num_microbatches
        recv_reqs = []
        microbatches_grads = []

        for i in range(num_microbatches):
            grad_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            print(f"Rank {rank} (Stage 1): Receiving gradient for microbatch {i} from rank {rank+1}", flush=True)
            recv_reqs.append((dist.irecv(tensor=grad_tensor, src=rank+1), grad_tensor))

        for req, tensor in recv_reqs:
            req.wait()
            microbatches_grads.append(tensor)

        # Now backward through stored_activations:
        for i, (inp, out) in enumerate(stored_activations):
            print(f"Rank {rank} (Stage 1): Backward pass on microbatch {i}", flush=True)
            out.backward(microbatches_grads[i].to(device))

            # Then send gradients backward
            req = dist.isend(tensor=inp.grad.to("cpu"), dst=rank-1)
            req.wait()

        print(f"Rank {rank} (Stage 1): Completed backward pass", flush=True)

    elif stage_id == 0:
        print(f"Rank {rank} (Stage 0): Starting backward pass", flush=True)
        micro_batch_size = batch_size // num_microbatches
        recv_reqs = []
        microbatches_grads = []
        for i in range(num_microbatches):
            grad_tensor = torch.empty((micro_batch_size, seq_l, dmodel))
            print(f"Rank {rank} (Stage 0): Receiving gradient for microbatch {i} from rank {rank+1}", flush=True)
            recv_reqs.append((dist.irecv(tensor=grad_tensor, src=rank+1), grad_tensor))
        for req, tensor in recv_reqs:
            req.wait()
            microbatches_grads.append(tensor)
        microbatches_embedded = torch.chunk(saved_embedded, num_microbatches, dim=0)
        for i, (micro, grad_tensor) in enumerate(zip(microbatches_embedded, microbatches_grads)):
            print(f"Rank {rank} (Stage 0): Backward pass on microbatch {i}", flush=True)
            if i < len(microbatches_embedded) - 1:
                micro.backward(grad_tensor.to(device), retain_graph=True)
            else:
                micro.backward(grad_tensor.to(device))
        print(f"Rank {rank} (Stage 0): Completed backward pass", flush=True)

    # -----------------------
    # Data Parallel Gradient Aggregation across Pipelines (for same stage)
    # -----------------------
    for param in net.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=stage_group)
            param.grad /= 2

    optim.step()
    torch.cuda.empty_cache()
    print(f"Rank {rank} Iter {itr}: Completed iteration", flush=True)