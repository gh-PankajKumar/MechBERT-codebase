#!/bin/bash
#BASH_ENV=/usr/share/lmod/lmod/init/bash
cd $PBS_O_WORKDIR
export PATH=$PATH:$PBS_O_PATH
source .bashrc

export LD_LIBRARY_PATH=/soft/compilers/cudatoolkit/cuda-11.6.2/extras/CUPTI/lib64:/soft/compilers/cudatoolkit/cuda-11.6.2/lib64:/soft/libraries/trt/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/lib:/soft/libraries/nccl/nccl_2.14.3-1+cuda11.6_x86_64/lib:/soft/libraries/cudnn/cudnn-11.6-linux-x64-v8.4.1.50/lib:/opt/cray/pe/gcc/11.2.0/snos/lib64:/opt/cray/pe/papi/6.0.0.14/lib64:/opt/cray/libfabric/1.11.0.4.125/lib64:/dbhome/db2cat/sqllib/lib64:/dbhome/db2cat/sqllib/lib64/gskit:/dbhome/db2cat/sqllib/lib32

conda deactivate
conda activate $CONDA_ENV


export HTTP_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"


echo Using Python `which python`
echo Conda prefix in launch `echo $CONDA_PREFIX`

wandb agent $WANDB_AGENT
