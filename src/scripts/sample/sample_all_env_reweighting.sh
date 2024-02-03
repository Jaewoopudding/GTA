


# tar -xvf /model/medium-expert-v2.tar.gz  -C /model

dataset1="hopper-medium-v2"
ckpt1="results/hopper-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-08/03:02/model-1000000.pt"

# dataset2="halfcheetah-medium-v2"
# ckpt2="results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

dataset2="walker2d-medium-v2"
ckpt2="results/walker2d-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-08/03:01/model-1000000.pt"

dataset3="hopper-medium-replay-v2"
ckpt3="results/hopper-medium-replay-v2/smallmixer_denoiser_v4_reweighting/2024-01-14/06:01/model-1000000.pt"

config_name="smallmixer_denoiser_v4_reweighting"

bash src/scripts/sample/sample_any_env_final.sh $dataset1 5 $ckpt1 $config_name&
bash src/scripts/sample/sample_any_env_final.sh $dataset2 6 $ckpt2 $config_name&
bash src/scripts/sample/sample_any_env_final.sh $dataset3 7 $ckpt3 $config_name&



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



# dataset1="hopper-medium-expert-v2"
# ckpt1="/model/experts/hopper-model-1000000.pt"

# dataset2="halfcheetah-medium-expert-v2"
# ckpt2="results/halfcheetah-medium-expert-v2/smallmixer_denoiser_v4/2024-01-11/12:31/model-1000000.pt"

# dataset3="walker2d-medium-expert-v2"
# ckpt3="results/walker2d-medium-expert-v2/smallmixer_denoiser_v4/2024-01-11/12:49/model-1000000.pt"

# config_name="smallmixer_denoiser_v4"

# # bash src/scripts/sample/sample_any_env_final.sh $dataset1 0 $ckpt1 $config_name&
# bash src/scripts/sample/sample_any_env_final.sh $dataset2 0 $ckpt2 $config_name&
# bash src/scripts/sample/sample_any_env_final.sh $dataset3 1 $ckpt3 $config_name&


# dataset5="maze2d-umaze-v1"
# config_name="smallmixer_denoiser_v4_maze"
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


# wait



# dataset5="hopper-medium-expert-v2"
# ckpt5="results/hopper-medium-expert-v2/smallmixer_denoiser_v4/2024-01-07/18:59/model-1000000.pt"

# config_name="smallmixer_denoiser_v4"

# bash src/scripts/sample/sample_any_env_final.sh $dataset5 5 $ckpt5 $config_name&


# dataset1="hopper-medium-v2"
# ckpt1="results/hopper-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# dataset3="walker2d-medium-v2"
# ckpt3="results/walker2d-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"

# config_name="smallmixer_denoiser_v4"

# bash src/scripts/sample/sample_any_env_final.sh $dataset1 6 $ckpt1 $config_name&
# bash src/scripts/sample/sample_any_env_final.sh $dataset3 7 $ckpt3 $config_name&