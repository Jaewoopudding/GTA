
envs='walker2d-medium-replay-v2 walker2d-random-v2 hopper-medium-v2 hopper-medium-expert-v2 hopper-medium-replay-v2 hopper-random-v2'

for env in $envs
do
    echo $env
    python /home/sujin/project/Augmentation-For-OfflineRL/src/scripts/envtest/put_RTG.py --env $env
done

# walker random