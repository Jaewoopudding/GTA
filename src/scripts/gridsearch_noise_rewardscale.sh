
start_id=0

gpu_id=$start_id
env_nm="halfcheetah"

for guidance_rewardscale in 1.25 1.3
do
    for noise_level in 0.5 0.75 1.0
    do  
        for temperature in 0.1 0.5 1.0 2.0 
        do
            if [ $gpu_id -eq 7 ]; then
                CUDA_VISIBLE_DEVICES=$gpu_id \
                accelerate launch src/diffusion/train_diffuser.py \
                --load_checkpoint \
                --ckpt_path=/input/smixer.pt \
                --back_and_forth \
                --dataset="${env_nm}-medium-v2" \
                --guidance_rewardscale=$guidance_rewardscale \
                --config_name=smallmixer_denoiser.yaml \
                --temperature=$temperature \
                --noise_level=$noise_level &&\
                gpu_id=$start_id
            else
                CUDA_VISIBLE_DEVICES=$gpu_id \
                accelerate launch src/diffusion/train_diffuser.py \
                --load_checkpoint \
                --ckpt_path=/input/smixer.pt \
                --back_and_forth \
                --dataset="${env_nm}-medium-v2" \
                --guidance_rewardscale=$guidance_rewardscale \
                --config_name=smallmixer_denoiser.yaml \
                --temperature=$temperature \
                --noise_level=$noise_level &\
                gpu_id=$((gpu_id+1))
            fi
        done
    done
done


wait
