

gpu_id=4
dataset="hopper-medium-replay-v2"
ckpt="results/hopper-medium-replay-v2/smallmixer_denoiser_v5/14:59-200/model-1000000.pt"
ext="npz"
config_name="smallmixer_denoiser_v5"

for noise_level in 0.5
do
    for guidance_rewardscale in 1.25 1.3 1.05
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

