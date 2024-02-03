#!/bin/bash

# 압축할 폴더 설정
FOLDER_TO_COMPRESS="/home/jaewoo/practices/diffusers/Augmentation-For-OfflineRL/data/generated_data/synther"

# 압축된 파일의 이름 설정
OUTPUT_FILE="compressed_folder.tar.gz"

# 폴더 압축
tar -czvf $OUTPUT_FILE -C $FOLDER_TO_COMPRESS .

echo finished