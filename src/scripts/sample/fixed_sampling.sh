


start_id=3

gpu_id=$start_id
for fixed_rewardscale in 0.95 0.9 0.8
do
    for noise_level in 0.75 1.0
    do
        CUDA_VISIBLE_DEVICES=$gpu_id \
        accelerate launch src/diffusion/train_diffuser.py \
        --load_checkpoint \
        --dataset=hopper-medium-v2 \
        --ckpt_path=results/hopper-medium-v2/smallmixer_denoiser_penalty/2023-12-22/16:02/model-1000000-v2.pt \
        --back_and_forth \
        --fixed_rewardscale=$fixed_rewardscale \
        --config_name=smallmixer_denoiser_penalty.yaml \
        --noise_level=$noise_level
        wait
        
    done
done
# CUDA_VISIBLE_DEVICES="0" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
# --back_and_forth \
# --guidance_rewardscale=1.15 \
# --noise_level=0.75 \
# --config_name=smallmixer_denoiser.yaml &\
# CUDA_VISIBLE_DEVICES="0" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
# --back_and_forth \
# --guidance_rewardscale=1.2 \
# --noise_level=0.75 \
# --config_name=smallmixer_denoiser.yaml &\
# CUDA_VISIBLE_DEVICES="1" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
# --back_and_forth \
# --guidance_rewardscale=1.25 \
# --noise_level=0.75 \
# --config_name=smallmixer_denoiser.yaml &\
# CUDA_VISIBLE_DEVICES="1" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
# --back_and_forth \
# --guidance_rewardscale=1.3 \
# --noise_level=0.75 \
# --config_name=smallmixer_denoiser.yaml &\
# CUDA_VISIBLE_DEVICES="2" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
# --back_and_forth \
# --guidance_rewardscale=1.15 \
# --noise_level=0.5 \
# --config_name=smallmixer_denoiser.yaml &\
# CUDA_VISIBLE_DEVICES="2" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
# --back_and_forth \
# --guidance_rewardscale=1.2 \
# --noise_level=0.5 \
# --config_name=smallmixer_denoiser.yaml &\
# CUDA_VISIBLE_DEVICES="3" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
# --back_and_forth \
# --guidance_rewardscale=1.25 \
# --noise_level=0.5 \
# --config_name=smallmixer_denoiser.yaml &\
# CUDA_VISIBLE_DEVICES="3" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
# --back_and_forth \
# --guidance_rewardscale=1.3 \
# --noise_level=0.5 \
# --config_name=smallmixer_denoiser.yaml &

