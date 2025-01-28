#!/bin/bash
# Tokenize the corpus before pre-training and store in cache to be reused

unset LD_PRELOAD
module load conda; conda activate $CONDA_ENV
export RUN_NAME=$RUN_NAME
export OUTNAME=$OUT_NAME
export WANDB_PROJECT=$WAND_PROJECT_NAME
export HF_HOME=$HF_HOME
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONPATH=$PBS_O_WORKDIR

## FOR DEBUGGING:
# echo $RUN_NAME
# echo $OUTNAME
# echo Using Python `which python`

python run_mlm.py \
    --model_name_or_path $MODEL \
    --tokenizer_name $TOKENIZER_PATH \
    --train_file $TRAIN_FILE \
    --report_to wandb \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --seed 0 \
    --output_dir "$OUTNAME" \
    --run_name $RUN_NAME \
    --overwrite_cache True \
    # --overwrite-output-dir \
