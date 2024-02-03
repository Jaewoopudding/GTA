
# config_name="smallmixer_denoiser_v4"
# ext="npz"

# for dataset in maze2d-umaze-v1 maze2d-medium-v1 maze2d-large-v1
# do
#     for date in 2023-12-31 2024-01-01
#     do
#         folder_nm="results/${dataset}/${config_name}/${date}"
#         for folder in $folder_nm/*
#         do
#             for file in $folder/*
#             do
#                 if [[ "$file" == *"$ext"* ]]; then
#                     echo $file
#                     cp $file "/home/orl/jaewoo/v4/${dataset}/${file##*/}"
#                 fi
#             done
#         done
#     done
# done




config_name="smallmixer_denoiser_v4"
ext="npz"

for dataset in hopper-medium-v2 walker2d-medium-v2 halfcheetah-medium-v2
do
    for date in 2024-01-02 2024-01-03 2024-01-04
    do
        folder_nm="results/${dataset}/${config_name}/${date}"
        for folder in $folder_nm/*
        do
            for file in $folder/*
            do
                if [[ "$file" == *"$ext"* ]]; then
                    echo $file
                    cp $file "/home/orl/jaewoo/v4/higher_guidance/${dataset}/${file##*/}"
                fi
            done
        done
    done
done