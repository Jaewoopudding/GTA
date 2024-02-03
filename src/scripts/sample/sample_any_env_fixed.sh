

dataset=$1
gpu_id=$2
ckpt=$3



config_name="smallmixer_denoiser_v4.yaml"
ext="npz"


for noise_level in 0.5 0.25 0.75
do
    for fixed_rewardscale in 0.95 
    do
        for temperature in 1.0 1.5 2.0
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
            --fixed_rewardscale=$fixed_rewardscale 

            wait
        done
    done
done

wait


folder_nm="results/${dataset}/${config_name}/2023-12-25"
for folder in $folder_nm/*
do
    for file in $folder/*
    do
        if [[ "$file" == *"$ext"* ]]; then
            cp $file "/home/orl/jaewoo/v4/${dataset}/${file##*/}"
        fi
    done
done