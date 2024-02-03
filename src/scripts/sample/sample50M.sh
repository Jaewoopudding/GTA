

CUDA_VISIBLE_DEVICES="0" \
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--ckpt_path=/home/sujin/project/Augmentation-For-OfflineRL/results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
--back_and_forth \
--guidance_rewardscale=1.3 \
--noise_level=0.5 \
--config_name=smallmixer_denoiser.yaml