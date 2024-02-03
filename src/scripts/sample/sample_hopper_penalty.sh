ckpt=results/hopper-medium-v2/smallmixer_denoiser_penalty/2023-12-22/16:02/model-1000000-v2.pt

start_id=4
gpu_id=$start_id
noise_level=0.5
guidance_rewardscale=1.1
temperature=2.0

for guidance_rewardscale in 1.1
do
    for temperature in 1.0 0.5 2.0 10.0
    do
        CUDA_VISIBLE_DEVICES=$gpu_id \
        accelerate launch src/diffusion/train_diffuser.py \
        --load_checkpoint \
        --dataset=hopper-medium-v2 \
        --config_name=smallmixer_denoiser_penalty.yaml \
        --ckpt_path=$ckpt \
        --back_and_forth \
        --noise_level=$noise_level \
        --temperature=$temperature \
        --guidance_rewardscale=$guidance_rewardscale

        wait
    done
done

wait

ext="npz"
for folder_nm in results/hopper-medium-v2/smallmixer_denoiser_penalty/2023-12-22/*
do
    for file_nm in $folder_nm/*
    do 
        if [[ "$file_nm" == *"$ext"* ]]; then
            cp $file_nm "/home/orl/jaewoo/v2/hopper-medium-v2/penalty100/${file_nm##*/}"
        fi
    done
done


# for guidance_rewardscale in 1.05
# do
#     for temperature in 1.0 0.5 2.0 10.0
#     do
#         CUDA_VISIBLE_DEVICES=$gpu_id \
#         accelerate launch src/diffusion/train_diffuser.py \
#         --load_checkpoint \
#         --dataset=hopper-medium-v2 \
#         --config_name=smallmixer_denoiser_penalty.yaml \
#         --ckpt_path=results/hopper-medium-v2/smallmixer_denoiser_penalty/2023-12-18/16:02/model-1000000.pt \
#         --back_and_forth \
#         --noise_level=$noise_level \
#         --temperature=$temperature \
#         --guidance_rewardscale=$guidance_rewardscale

#         wait
#     done
# done

# wait

# ext="npz"
# for folder_nm in results/hopper-medium-v2/smallmixer_denoiser_penalty/2023-12-23/*
# do
#     for file_nm in $folder_nm/*
#     do 
#         if [[ "$file_nm" == *"$ext"* ]]; then
#             cp $file_nm "/home/orl/jaewoo/v2/hopper-medium-v2/penalty100/${file_nm##*/}"
#         fi
#     done
# done

