#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate baseline

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sujin/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

python /home/sujin/project/Augmentation-For-OfflineRL/src/test.py \
--save_dir ./samples/ \
--config_path ../configs/ \
--ckpt_path ./checkpoints/sample_ckpts/ \