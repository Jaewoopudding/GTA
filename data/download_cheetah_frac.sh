envs='halfcheetah_medium'
fractions='0.05 0.1 0.15 0.2'
for env in $envs
do
    for fraction in $fractions
    do
        python data/download_d4rl_datasets.py --env_name=$env --fraction=$fraction&
        wait
    done
done

