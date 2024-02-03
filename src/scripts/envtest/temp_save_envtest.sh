
dataset='halfcheetah-medium-v2'
ext="np"


for folder_nm in results/halfcheetah-medium-v2/smallmixer_denoiser_v4/2024-01-23/12:05/*
do
    for file_nm in $folder_nm/*
    do
        echo $file_nm
        if [[ "$file_nm" == *"$ext"* ]]; then
            data_path=$file_nm
            echo $file_nm
            python src/scripts/envtest/save_env_stats.py \
            --dataset $dataset \
            --data_path $data_path \
            --percentage=0.1 &
        fi
    done
done