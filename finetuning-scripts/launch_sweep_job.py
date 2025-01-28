import os
import subprocess
import wandb


def run_sweep_iter():
    """
    Run a single iteration of a hyperparameter sweep.

    This function initializes a Weights & Biases run, sets environment variables for the run name,
    retrieves configuration parameters from the run, constructs a command string for the sweep script,
    and executes the sweep script with the specified parameters using subprocess.
    """
    run = wandb.init()
    os.environ["RUN_NAME"] = run.name
    config = wandb.config
    cmd_str = f"wandb_sweep_single_job.sh --learning_rate={str(config.learning_rate)} --num_train_epochs={str(config.num_train_epochs)} --per_device_train_batch_size={str(config.per_device_train_batch_size)} --run_name={run.name} --run_id={run.run_id}"
    subprocess.run(cmd_str, shell=True)


wandb.agent(sweep_id=os.getenv("SWEEP_NAME"), function=run_sweep_iter)
