# Train Tokenizer
module load conda; conda activate $CONDA_ENV

export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"


wandb login $WANDB_API

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NPROCESSES=$(( NNODES * NRANKS_PER_NODE ))
export RUN_NAME=$RUN_NAME
export PYTHONPATH=$PBS_O_WORKDIR
export OUT_NAME=$OUT_NAME
export WANDB_PROJECT=tokenizer
export HF_HOME=$HF_HOME

export MPICH_GPU_SUPPORT_ENABLED=1


## FOR DEBUGGING:
# echo Using Python `which python`
# echo $RUN_NAME
# echo $OUT_NAME
# echo Working directory is $PBS_O_WORKDIR


python train_tokenizer.py
