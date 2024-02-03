


tar -xvf /model/medium-replay-v2.tar.gz  -C /model

# dataset1="hopper-medium-v2"
# ckpt1="results/hopper-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset2="halfcheetah-medium-v2"
# ckpt2="results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset3="walker2d-medium-v2"
# ckpt3="results/walker2d-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# bash src/scripts/sample/sample_any_env_low.sh $dataset1 6 $ckpt1 &
# bash src/scripts/sample/sample_any_env_high.sh $dataset1 7 $ckpt1 &
# bash src/scripts/sample/sample_any_env_low.sh $dataset2 6 $ckpt2 &
# bash src/scripts/sample/sample_any_env_high.sh $dataset2 7 $ckpt2 &
# # bash src/scripts/sample/sample_any_env_low.sh $dataset3 4 $ckpt3 &
# # bash src/scripts/sample/sample_any_env_high.sh $dataset3 5 $ckpt3 &



# dataset5="maze2d-umaze-v1"
# ckpt5="results/maze2d-umaze-v1/smallmixer_denoiser_v4/2023-12-29/13:04/model-1000000.pt"
# bash src/scripts/sample/sample_any_env_low.sh $dataset5 2 $ckpt5 &
# bash src/scripts/sample/sample_any_env_high.sh $dataset5 3 $ckpt5 &

# dataset6="maze2d-large-v1"
# ckpt6="results/maze2d-large-v1/smallmixer_denoiser_v4/2023-12-29/13:05/model-1000000.pt"
# bash src/scripts/sample/sample_any_env_low.sh $dataset6 4 $ckpt6 &
# bash src/scripts/sample/sample_any_env_high.sh $dataset6 5 $ckpt6 &

# dataset7="maze2d-medium-v1"
# ckpt7="results/maze2d-medium-v1/smallmixer_denoiser_v4/2023-12-29/15:03/model-1000000.pt"
# bash src/scripts/sample/sample_any_env_low.sh $dataset7 6 $ckpt7 &
# bash src/scripts/sample/sample_any_env_high.sh $dataset7 7 $ckpt7 &



dataset1="hopper-medium-replay-v2"
ckpt1="/model/replays/hopper-model-1000000.pt"

dataset2="halfcheetah-medium-replay-v2"
ckpt2="/model/replays/halfcheetah-model-1000000.pt"

dataset3="walker2d-medium-replay-v2"
ckpt3="/model/replays/walker2d-model-1000000.pt"

config_name="smallmixer_denoiser_v4"

bash src/scripts/sample/sample_any_env_final.sh $dataset1 0 $ckpt1 $config_name&
bash src/scripts/sample/sample_any_env_final.sh $dataset2 1 $ckpt2 $config_name&
bash src/scripts/sample/sample_any_env_final.sh $dataset3 2 $ckpt3 $config_name&



wait
