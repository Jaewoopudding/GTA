

dataset=$1
gpu_id=$2
ckpt=$3
config_name=$4




ext="npz"


for noise_level in 0.5 
do
    for guidance_rewardscale in 1.2 1.1 
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

