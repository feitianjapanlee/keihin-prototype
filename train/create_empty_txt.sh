#!/bin/bash

# imagesフォルダとlabelsフォルダのパスを指定
IMAGE_FOLDER="./workspace/keihin-test-4-1000-w640h480/images/Train"
LABEL_FOLDER="./workspace/keihin-test-4-1000-w640h480/labels/Train"

# imagesフォルダ内のすべての.jpgファイルを処理
for jpg_file in "$IMAGE_FOLDER"/*.jpg; do
    # ファイル名から拡張子を除去（XXX.jpg -> XXX）
    filename=$(basename -- "$jpg_file")
    filename_noext="${filename%.*}"

    # 対応する.txtファイルのパス
    txt_file="$LABEL_FOLDER/$filename_noext.txt"

    # .txtファイルが存在しない場合に作成
    if [ ! -f "$txt_file" ]; then
        touch "$txt_file"
        echo "Created: $txt_file"
    else
        echo "Exists: $txt_file"
    fi
done
