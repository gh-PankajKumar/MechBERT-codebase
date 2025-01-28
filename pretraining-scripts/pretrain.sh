# Init Pre-training
module load conda; conda activate $CONDA_ENV

export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"


wandb login $WANDB_API

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NPROCESSES=$(( NNODES * NRANKS_PER_NODE ))
RUN_NAME=$RUN_NAME
PYTHONPATH=$PBS_O_WORKDIR
OUTNAME=$OUTNAME
MAIN_IP_ADDR=$(hostname -i)
WANDB_PROJECT=$RUN_NAME

## If storage quota is maxed change to different directory:
# HF_DATASETS_DOWNLOADED_DATASETS_PATH=$HF_DATASETS
# TRANSFORMERS_CACHE=$TF_CACHE


HF_HOME=$HF_HOME
NCCL_COLLNET_ENABLE=1
NCCL_NET_GDR_LEVEL=PHB
WANDB__SERVICE_WAIT=300

export MPICH_GPU_SUPPORT_ENABLED=1


## FOR DEBUGGING:
# echo Using Python `which python`
# echo $RUN_NAME
# echo $OUTNAME
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR

# echo Jobid: $PBS_JOBID
# echo Running on host `hostname`
# echo Running on nodes `cat $PBS_NODEFILE`

ds_report

HOSTFILE=hostfile-$RUN_NAME

rm $HOSTFILE
touch $HOSTFILE
cat $PBS_NODEFILE > $HOSTFILE
sed -e 's/$/ slots=4/' -i $HOSTFILE

## FOR DEBUGGING:
# echo Hostfile path $HOSTFILE
# cat $HOSTFILE

rm .deepspeed_env
touch .deepspeed_env

echo "PATH=${PATH}" >> .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "no_proxy=${no_proxy}'" >> .deepspeed_env
echo "ftp_proxy=${ftp_proxy}'" >> .deepspeed_env
echo "HF_HOME=${HF_HOME}" >> .deepspeed_env
echo "WANDB_PROJECT=${WANDB_PROJECT}" >> .deepspeed_env
echo "WANDB__SERVICE_WAIT=300" >> .deepspeed_env

deepspeed \
    --hostfile=$HOSTFILE \
    run_mlm.py \
    --model_name_or_path $MODEL_NAME \
    --train_file $TRAIN_FILE \
    --report_to wandb \
    --do_train \
    --do_eval \
    --learning_rate $LEARN_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --max_steps $MAX_STEPS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --seed 0 \
    --output_dir "$OUTNAME" \
    --run_name $RUN_NAME \
    --save_total_limit 50 \
    --save_steps 5000 \
    --logging_steps 1000 \
    --warmup_steps $WARMUP_STEPS \
    --is_preprocessing_done \
    --dataloader_num_workers 16 \
    --fp16 \
    --deepspeed ds_config.json
    # --eval_steps 15000 \
    # --evaluation_strategy steps \
    # --metric_for_best_model eval_loss \
    # --load_best_model_at_end \
    # --overwrite-output-dir \
    # --fp16

rm $HOSTFILE
