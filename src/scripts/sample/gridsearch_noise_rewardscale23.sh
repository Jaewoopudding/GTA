
start_id=0

gpu_id=$start_id
for noise_level in 0.5
do
    for guidance_rewardscale in 1.1 1.05 1.15
    do  
        for temperature in 1.0
        do
            CUDA_VISIBLE_DEVICES=$gpu_id \
            accelerate launch src/diffusion/train_diffuser.py \
            --load_checkpoint \
            --dataset=hopper-medium-v2 \
            --ckpt_path=results/hopper-medium-v2/smallmixer_denoiser_penalty/2023-12-22/04:23/model-1000000-v2.pt \
            --back_and_forth \
            --guidance_rewardscale=$guidance_rewardscale \
            --config_name=smallmixer_denoiser_penalty23.yaml \
            --temperature=$temperature \
            --noise_level=$noise_level
            wait
        done
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

