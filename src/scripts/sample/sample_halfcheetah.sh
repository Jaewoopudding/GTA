

gpu_id=1
dataset="halfcheetah-medium-expert-v2"
ckpt="results/halfcheetah-medium-expert-v2/smallmixer_denoiser_v4/2024-01-18/17:11/model-1000000.pt"

# dataset="halfcheetah-medium-v2"
# ckpt="results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"


ext="npz"
config_name="smallmixer_denoiser_v4"

for noise_level in 0.5
do
    for guidance_rewardscale in 1.1 1.2
    do
        for temperature in 2.0
        do
            CUDA_VISIBLE_DEVICES=$gpu_id \
            accelerate launch src/diffusion/train_diffuser.py \
            --load_checkpoint \
            --dataset=$dataset\
            --config_name=$config_name \
            --ckpt_path=$ckpt \
            --back_and_forth \
            --noise_level=$noise_level \
            --temperature=$temperature \
            --guidance_rewardscale=$guidance_rewardscale


            wait
        done
    done
done

wait

