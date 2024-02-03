
envs='halfcheetah-medium-v2 halfcheetah-medium-expert-v2 halfcheetah-medium-replay-v2 halfcheetah-random-v2 walker2d-medium-v2 walker2d-medium-expert-v2'

for env in $envs
do
    echo $env
    python /home/sujin/project/Augmentation-For-OfflineRL/src/scripts/envtest/put_RTG.py --env $env
done