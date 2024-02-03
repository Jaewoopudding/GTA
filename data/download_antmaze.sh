envs='antmaze-umaze-v2 antmaze-umaze-diverse-v2 antmaze-medium-diverse-v2 antmaze-medium-play-v2 antmaze-large-diverse-v2 antmaze-large-play-v2'

for env in $envs
do
    echo $env
    python data/download_d4rl_datasets.py --env_name=$env&
    wait
done
