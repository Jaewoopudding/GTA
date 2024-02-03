
config_name="smallmixer_denoiser_v4.yaml"
group_nm="smallmixer_denoiser_v4"

env_name="hopper"
dataset_type="medium-replay"
CUDA_VISIBLE_DEVICES="0,1,2,3" \
accelerate launch \
--multi_gpu \
--main_process_port=37117 ./src/diffusion/train_diffuser.py \
--dataset="${env_name}-${dataset_type}-v2" \
--config_name="${config_name}" \
--wandb_project="synther_${env_name}" \
--wandb_group="${group_nm}-${dataset_type}"&
env_name="walker2d"
CUDA_VISIBLE_DEVICES="4,5,6,7" \
accelerate launch \
--multi_gpu \
--main_process_port=45907 ./src/diffusion/train_diffuser.py \
--dataset="${env_name}-${dataset_type}-v2" \
--config_name="${config_name}" \
--wandb_project="synther_${env_name}" \
--wandb_group="${group_nm}-${dataset_type}"&


wait