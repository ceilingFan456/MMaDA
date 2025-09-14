#!/usr/bin/env bash

source /home/azureuser/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/azureuser_envs/mmada

cd /home/azureuser/MMaDA
pwd

# create writable caches on your big disk
sudo mkdir -p /mnt/hf/{hub,transformers,datasets}
sudo chmod -R 775 /mnt/hf
sudo chown -R "$USER":"$USER" /mnt/hf

# use HF_HOME (and hub/datasets under it). unset old TRANSFORMERS_CACHE var.
export HF_HOME=/mnt/hf
export HUGGINGFACE_HUB_CACHE=/mnt/hf/hub
export HF_DATASETS_CACHE=/mnt/hf/datasets
unset TRANSFORMERS_CACHE

# (optional) NCCL nic tweaks if you see hangs
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0


export WANDB_API_KEY=e1b1fccedc5a6ad97a728268dc6a13c6bdc9f1ca
export WANDB_ENTITY=ceilingfan-national-university-of-singapore-students-union          # if needed
export WANDB_PROJECT=mmada-training-stage2
# wandb login --relogin   # ensure the token matches 'ceilingfan'
wandb login e1b1fccedc5a6ad97a728268dc6a13c6bdc9f1ca

accelerate launch \
    --main_process_port=29500 \
    training/train_mmada_stage2_mmu_alone.py \
    config=configs/mmada_pretraining_stage2_llada_instruct_mmu_alone.yaml

# accelerate launch \
#     --config_file=accelerate_configs/1_node_4_gpus_deepspeed_zero2.yaml \
#     --main_process_port=29500 \
#     training/train_mmada_stage2_mmu_alone.py \
#     config=configs/mmada_pretraining_stage2_llada_instruct_mmu_alone.yaml

# python - <<'PY'
# import wandb
# run = wandb.init(entity="ceilingfan", project="mmada-training-stage2")
# print("OK:", run.url); run.finish()
# PY
