

config_name="smallmixer_denoiser_v4.yaml"
group_nm="smallmixer_denoiser_v4"

env_name="antmaze"
dataset_type="umaze-diverse"
CUDA_VISIBLE_DEVICES="0,1,2,3" \
accelerate launch \
--multi_gpu \
--main_process_port=37117 ./src/diffusion/train_diffuser.py \
--dataset="${env_name}-${dataset_type}-v2" \
--config_name="${config_name}" \
--wandb_project="synther_${env_name}" \
--wandb_group="${group_nm}-${dataset_type}" 


wait
# CUDA_VISIBLE_DEVICES="6,7" \
# accelerate launch \
# --multi_gpu \
# --main_process_port=53201 ./src/diffusion/train_diffuser.py \
# --dataset="${env_name}-large-diverse-v2" \
# --config_name="${config_name}" \
# --wandb-project="synther_${env_name}" \
# --wandb-group="${env_name}-large-diverse" &\





# env_name="antmaze"
# for maze_type in 'umaze' 'medium' 'large'
# do
#         for dataset_type in "diverse"
#         do
#                 echo "Training edm for ${env_name}-${maze_type}-${dataset_type}-v2"

#                 config_name="temporalattention_denoiser.yaml"
#                 CUDA_VISIBLE_DEVICES="3,4" \
#                 accelerate launch \
#                 --multi_gpu \
#                 --main_process_port=47967 ./src/diffusion/train_diffuser.py \
#                 --dataset="${env_name}-${maze_type}-${dataset_type}-v2" \
#                 --config_name="${config_name}" \
#                 --wandb-project="synther_${env_name}" \
#                 --wandb-group="$${env_name}-${maze_type}-${dataset_type}" 
#         done
# done

