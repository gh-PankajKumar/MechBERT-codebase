
for i in "$@"; do
  case $i in
    --learning_rate=*)
      LR="${i#*=}"
      shift # past argument=value
      ;;
    --num_train_epochs=*)
      NUM_EPOCHS="${i#*=}"
      shift # past argument=value
      ;;
    --per_device_train_batch_size=*)
      PER_DEVICE_BATCH_SIZE="${i#*=}"
      shift # past argument with no value
      ;;
    --run_name)
      RUN_NAME="${i#*=}"
      shift # past argument with no value
      ;;
    -*|--*)
      echo "Unknown option $i"
      shift
      ;;
    *)
      ;;
  esac
done

`python polaris_scripts/Sweep/init_run.py`

MODEL_NAME=$MODEL_NAME
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NPROCESSES=$(( NNODES * NRANKS_PER_NODE ))
BASE_MODEL_NAME=`echo "${MODEL_NAME##*/}"`
PYTHONPATH=$PBS_O_WORKDIR

OUTNAME=$OUT_PATH
#LD_LIBRARY_PATH=/soft/compilers/cudatoolkit/cuda-11.6.2/extras/CUPTI/lib64:/soft/compilers/cudatoolkit/cuda-11.6.2/lib64:/soft/libraries/trt/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/lib:/soft/libraries/nccl/nccl_2.14.3-1+cuda11.6_x86_64/lib:/soft/libraries/cudnn/cudnn-11.6-linux-x64-v8.4.1.50/lib:/opt/cray/pe/gcc/11.2.0/snos/lib64:/opt/cray/pe/papi/6.0.0.14/lib64:/opt/cray/libfabric/1.11.0.4.125/lib64:/dbhome/db2cat/sqllib/lib64:/dbhome/db2cat/sqllib/lib64/gskit:/dbhome/db2cat/sqllib/lib32

MAIN_IP_ADDR=$(hostname -i)
HF_HOME=$HF_HOME
TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
NCCL_COLLNET_ENABLE=1
NCCL_NET_GDR_LEVEL=PHB

export HF_HOME=$HF_HOME
export TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
export MPICH_GPU_SUPPORT_ENABLED=1


## FOR DEBUGGING:
# echo $RUN_NAME
# echo $OUTNAME
# echo Working directory is $PBS_O_WORKDIR
# cd $PBS_O_WORKDIR
#echo PATH is $PATH
#echo LD_LIB is $LD_LIBRARY_PATH
#echo Jobid: $PBS_JOBID
#echo Running on nodes `cat $PBS_NODEFILE`

echo Learning rate $LR
echo num epochs $NUM_EPOCHS
echo run name $RUN_NAME
echo batch sizeper device $PER_DEVICE_BATCH_SIZE

python -m torch.distributed.launch \
	--nproc_per_node 4 polaris_scripts/Sweep/run_qa_wandb_sweep.py \
    --fp16 \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name squad \
    --report_to wandb \
    --do_train \
    --do_eval \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --max_seq_length 512 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --doc_stride 128 \
    --seed 0 \
    --output_dir "$OUTNAME" \
    --change_cls_token 0 \
    --no_pad_to_max_length \
    --version_2_with_negative \
    --save_total_limit 5 \
    --save_steps 10000 \
    --run_id $RUN_ID \
    --deepspeed polaris_scripts/ds_config.json
    # --lr_scheduler_type $LR_SCHEDULER_TYPE \
    # --warmup_ratio $WARMUP_RATIO \
    # --overwrite-output-dir \
