
dataset1="hopper-medium-replay-v2"
ckpt1="results/hopper-medium-replay-v2/smallmixer_denoiser_v4/2023-12-30/18:45/model-1000000.pt"
test_dataset1="/home/orl/0117/${dataset1}.npz"

dataset2="hopper-medium-v2"
ckpt2="results/hopper-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"
test_dataset2="/home/orl/0117/${dataset2}.npz"

dataset3="hopper-medium-expert-v2"
ckpt3="results/hopper-medium-expert-v2/smallmixer_denoiser_v4/2024-01-07/18:59/model-1000000.pt"
test_dataset3="/home/orl/0117/${dataset3}.npz"

dataset4="walker2d-medium-replay-v2"
ckpt4="results/walker2d-medium-replay-v2/smallmixer_denoiser_v4/2023-12-30/19:04/model-1000000.pt"
test_dataset4="/home/orl/0117/${dataset4}.npz"

dataset5="walker2d-medium-v2"
ckpt5="results/walker2d-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"
test_dataset5="/home/orl/0117/${dataset5}.npz"

dataset6="walker2d-medium-expert-v2"
ckpt6="results/walker2d-medium-expert-v2/smallmixer_denoiser_v4/2024-01-11/12:49/model-1000000.pt"
test_dataset6="/home/orl/0117/${dataset6}.npz"

dataset7="halfcheetah-medium-replay-v2"
ckpt7="results/halfcheetah-medium-replay-v2/smallmixer_denoiser_v4/2024-01-07/06:34/model-1000000.pt"
test_dataset7="/home/orl/0117/${dataset7}.npz"

dataset8="halfcheetah-medium-v2"
ckpt8="results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2023-12-23/model-1000000.pt"
test_dataset8="/home/orl/0117/${dataset8}.npz"

dataset9="halfcheetah-medium-expert-v2"
ckpt9="results/halfcheetah-medium-expert-v2/smallmixer_denoiser_v4/2024-01-11/12:31/model-1000000.pt"
test_dataset9="/home/orl/0117/${dataset9}.npz"

config_name="smallmixer_denoiser_v4.yaml"

dataset10="maze2d-umaze-v1"
ckpt10="results/maze2d-umaze-v1/smallmixer_denoiser_v4_maze/2024-01-04/01:14/model-1000000.pt"
test_dataset10="/home/orl/0117/${dataset10}.npz"

dataset11="maze2d-large-v1"
ckpt11="results/maze2d-large-v1/smallmixer_denoiser_v4_maze/2024-01-04/06:45/model-1000000.pt"
test_dataset11="/home/orl/0117/${dataset11}.npz"

dataset12="maze2d-medium-v1"
ckpt12="results/maze2d-medium-v1/smallmixer_denoiser_v4_maze/2024-01-04/01:14/model-1000000.pt"
test_dataset12="/home/orl/0117/${dataset12}.npz"

config_name_maze="smallmixer_denoiser_v4_maze.yaml"

CUDA_VISIBLE_DEVICES="4" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt1 --dataset $dataset1 --config_name $config_name --test_dataset $test_dataset1 &\
CUDA_VISIBLE_DEVICES="5" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt2 --dataset $dataset2 --config_name $config_name --test_dataset $test_dataset2 &\
CUDA_VISIBLE_DEVICES="6" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt3 --dataset $dataset3 --config_name $config_name --test_dataset $test_dataset3 &\

wait

CUDA_VISIBLE_DEVICES="4" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt5 --dataset $dataset5 --config_name $config_name --test_dataset $test_dataset5 &\
CUDA_VISIBLE_DEVICES="5" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt6 --dataset $dataset6 --config_name $config_name --test_dataset $test_dataset6 &\
CUDA_VISIBLE_DEVICES="6" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt7 --dataset $dataset7 --config_name $config_name --test_dataset $test_dataset7 &\

wait

CUDA_VISIBLE_DEVICES="4" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt9 --dataset $dataset9 --config_name $config_name --test_dataset $test_dataset9 &\
CUDA_VISIBLE_DEVICES="5" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt10 --dataset $dataset10 --config_name $config_name_maze --test_dataset $test_dataset10 &\
CUDA_VISIBLE_DEVICES="6" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt11 --dataset $dataset11 --config_name $config_name_maze --test_dataset $test_dataset11 &\
CUDA_VISIBLE_DEVICES="7" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt12 --dataset $dataset12 --config_name $config_name_maze --test_dataset $test_dataset12 &\
wait

CUDA_VISIBLE_DEVICES="4" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt4 --dataset $dataset4 --config_name $config_name --test_dataset $test_dataset4 &\
CUDA_VISIBLE_DEVICES="5" python src/scripts/envtest/get_loglikelihood.py --ckpt_path $ckpt8 --dataset $dataset8 --config_name $config_name --test_dataset $test_dataset8 &\
