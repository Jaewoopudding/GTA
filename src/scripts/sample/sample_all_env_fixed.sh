

dataset1="hopper-medium-v2"
ckpt1="results/hopper-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset2="halfcheetah-medium-v2"
# ckpt2="results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset3="walker2d-medium-v2"
# ckpt3="results/walker2d-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

bash src/scripts/sample/sample_any_env_fixed.sh $dataset1 4 $ckpt1 &
# bash src/scripts/sample/sample_any_env_low.sh $dataset2 6 $ckpt2 &
# bash src/scripts/sample/sample_any_env_low.sh $dataset3 4 $ckpt3 &



# dataset4="antmaze-umaze-diverse-v2"
# ckpt4="results/antmaze-umaze-diverse-v2/smallmixer_denoiser_v4/2023-12-25/model-1000000.pt"

# bash src/scripts/sample/sample_any_env_low.sh $dataset4 2 $ckpt4 &
# #bash src/scripts/sample/sample_any_env_high.sh $dataset4 3 $ckpt4 &
