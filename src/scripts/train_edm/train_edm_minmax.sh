
# for env_name in 'walker2d' 'hopper' #'halfcheetah'
# do
#     for dataset_type in 'medium' #'medium-expert' 'medium-replay'
#     do
#         echo "Training edm for ${env_name}-${dataset_type}-v2"

        # config_name="mixer_denoiser.yaml"
        # CUDA_VISIBLE_DEVICES="6,7" \
        # accelerate launch \
        # --multi_gpu \
        # --main_process_port=47967 ./src/diffusion/train_diffuser.py \
        # --dataset="${env_name}-${dataset_type}-v2" \
        # --config_name="${config_name}" \
        # --wandb-project="synther:${env_name}" \
        # --wandb-group="${config_name}-${dataset_type}" &\

config_name="smallmixer_denoiser.yaml"
group_nm="smallmixer_denoiser"

env_name1="hopper"
env_name2="walker2d"
dataset_type="medium"
CUDA_VISIBLE_DEVICES="0,1,2,3" \
accelerate launch \
--multi_gpu \
--main_process_port=37117 ./src/diffusion/train_diffuser.py \
--dataset="${env_name1}-${dataset_type}-v2" \
--config_name="${config_name}" \
--wandb_project="synther_${env_name1}" \
--wandb_group="${group_nm}-${dataset_type}-minmax-woreweighted" &\

wait
CUDA_VISIBLE_DEVICES="0,1,2,3" \
accelerate launch \
--multi_gpu \
--main_process_port=47967 ./src/diffusion/train_diffuser.py \
--dataset="${env_name2}-${dataset_type}-v2" \
--config_name="${config_name}" \
--wandb_project="synther_${env_name2}" \
--wandb_group="${group_nm}-${dataset_type}-minmax-woreweighted"&\
# env_name="halfcheetah"
# config_name="smallmixer_denoiser_16.yaml"

# CUDA_VISIBLE_DEVICES="4,5" \
# accelerate launch \
# --multi_gpu \
# --main_process_port=45895 ./src/diffusion/train_diffuser.py \
# --dataset="${env_name}-${dataset_type}-v2" \
# --config_name="${config_name}" \
# --wandb-project="synther_${env_name}" \
# --wandb-group="${config_name}-${dataset_type}"&\
# config_name="smallmixer_denoiser_8.yaml"
# CUDA_VISIBLE_DEVICES="6,7" \
# accelerate launch \
# --multi_gpu \
# --main_process_port=57439 ./src/diffusion/train_diffuser.py \
# --dataset="${env_name}-${dataset_type}-v2" \
# --config_name="${config_name}" \
# --wandb-project="synther_${env_name}" \
# --wandb-group="${config_name}-${dataset_type}"&\


wait