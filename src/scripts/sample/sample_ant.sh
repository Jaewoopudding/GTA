
start_id=2
temperature=1.0


gpu_id=$start_id
for noise_level in 0.5 0.25 
do
    for guidance_rewardscale in 1.1 1.2
    do  
        CUDA_VISIBLE_DEVICES=$gpu_id \
        accelerate launch src/diffusion/train_diffuser.py \
        --load_checkpoint \
        --dataset=antmaze-umaze-diverse-v2 \
        --ckpt_path=results/antmaze-umaze-diverse-v2/smallmixer_denoiser_ant/2023-12-21/19:43/model-1000000.pt \
        --back_and_forth \
        --guidance_rewardscale=$guidance_rewardscale \
        --config_name=smallmixer_denoiser_ant.yaml \
        --temperature=$temperature \
        --noise_level=$noise_level
        wait
    done
done