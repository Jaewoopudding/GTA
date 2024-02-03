
start_id=6

gpu_id=$start_id
for guidance_rewardscale in 1.05 1.1 1.15 1.2 
do
    for noise_level in 0.5 0.75 1.0
    do  
        for temperature in 1.0 0.5 0.1 2.0
        do
            if [ $gpu_id -eq 7 ]; then
                CUDA_VISIBLE_DEVICES=$gpu_id \
                accelerate launch src/diffusion/train_diffuser.py \
                --load_checkpoint \
                --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
                --back_and_forth \
                --guidance_rewardscale=$guidance_rewardscale \
                --config_name=smallmixer_denoiser.yaml \
                --temperature=$temperature \
                --noise_level=$noise_level &\
                gpu_id=$start_id
                wait
            else
                CUDA_VISIBLE_DEVICES=$gpu_id \
                accelerate launch src/diffusion/train_diffuser.py \
                --load_checkpoint \
                --ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
                --back_and_forth \
                --guidance_rewardscale=$guidance_rewardscale \
                --config_name=smallmixer_denoiser.yaml \
                --temperature=$temperature \
                --noise_level=$noise_level &\
                gpu_id=$((gpu_id+1))
            fi
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

