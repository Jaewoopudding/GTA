<<<<<<< HEAD:src/scripts/sample_smixer.sh
# CUDA_VISIBLE_DEVICES="4" \
# accelerate launch src/diffusion/train_diffuser.py \
# --load_checkpoint \
# --ckpt_path=/home/orl/sujin/edm/output/halfcheetah-medium-v2/smallmixer_denoiser_16/2023-12-11/00:51/model-1000000-v2.pt \
# --back_and_forth \
# --config_name=smallmixer_denoiser_16.yaml \
# --guidance_rewardscale=1.25 \
# --noise_level=0.5 \

CUDA_VISIBLE_DEVICES="7" \
=======



CUDA_VISIBLE_DEVICES="2" \
>>>>>>> dev_sj:src/scripts/sample/sample_smixer.sh
accelerate launch src/diffusion/train_diffuser.py \
--load_checkpoint \
--ckpt_path=results/halfcheetah-medium-v2/mixer_denoiser/2023-11-27/17:39/model-1000000.pt \
--back_and_forth \
--config_name=smallmixer_denoiser.yaml \
--guidance_rewardscale=1.2 \
--noise_level=0.5 \



# note : reward scale need to be 0.001
# it scales to 0.001 after update on 1127
# construct_diffusion_model.denoising_network.dim = 16