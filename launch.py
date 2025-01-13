import os
import sys
import torch
import deepspeed

def main():
    # Get world size from GPU count
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Not enough GPUs available! Need at least 2 GPUs.")
        sys.exit(1)
    
    # Launch distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    
    for local_rank in range(world_size):
        env = os.environ.copy()
        env["LOCAL_RANK"] = str(local_rank)
        env["RANK"] = str(local_rank)
        
        # Launch training script with deepspeed
        cmd = [
            sys.executable, "-m", "deepspeed",
            "--num_gpus", str(world_size),
            "train.py",
            "--deepspeed",
            "--deepspeed_config", "ds_config.json"
        ]
        
        if local_rank == 0:
            print(f"Launching training on {world_size} GPUs...")
        
        os.system(" ".join(cmd))

if __name__ == "__main__":
    main()
