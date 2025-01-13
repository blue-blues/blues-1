import os
import sys
import subprocess
import torch

def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Not enough GPUs available! Need at least 2 GPUs.")
        sys.exit(1)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)

    processes = []
    for local_rank in range(world_size):
        env = os.environ.copy()
        env["LOCAL_RANK"] = str(local_rank)
        env["RANK"] = str(local_rank)
        
        cmd = [
            sys.executable,
            "train.py",
            "--local_rank", str(local_rank),
            "--deepspeed",
            "--deepspeed_config", "ds_config.json"
        ]
        
        if local_rank == 0:
            print(f"Launching training on {world_size} GPUs...")
            print(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd, env=env)
        processes.append(process)
    
    for process in processes:
        process.wait()
        if process.returncode != 0:
            print(f"Training failed with return code {process.returncode}")
            sys.exit(1)

if __name__ == "__main__":
    main()
