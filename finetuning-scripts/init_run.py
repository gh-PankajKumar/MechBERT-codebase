import wandb
import os

run = wandb.init()
os.environ["RUN_NAME"] = run.name
os.environ["RUN_ID"] = run.id
print(f"export RUN_NAME={run.name}")
print(f"export RUN_ID={run.id}")
wandb.finish()
