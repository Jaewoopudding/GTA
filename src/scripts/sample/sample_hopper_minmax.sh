
start_id=0
gpu_id=$start_id
for noise_level in 0.5 0.75 1.0
do
    for guidance_rewardscale in 1.1 1.15 1.2 1.05 
    do  
        for temperature in 1.0 0.5 0.1 2.0
        do
            CUDA_VISIBLE_DEVICES=$gpu_id \
            accelerate launch src/diffusion/train_diffuser.py \
            --load_checkpoint \
            --dataset=hopper-medium-v2 \
            --config_name=smallmixer_denoiser_minmax.yaml \
            --ckpt_path=results/hopper-medium-v2/smallmixer_denoiser/2023-12-18/21:17/model-1000000.pt \
            --back_and_forth \
            --noise_level=$noise_level \
            --temperature=$temperature \
            --guidance_rewardscale=$guidance_rewardscale \

            wait
        
        done
    done
done
