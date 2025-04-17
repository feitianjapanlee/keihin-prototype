#!/bin/bash

# このスクリプトは、指定されたimagesフォルダ内のすべての.jpgファイルに対して、
# 対応する.txtファイルが存在しない場合に空の.txtファイルを作成します。
# 画像とラベルのフォルダパスを指定
IMAGE_FOLDER="../datasets/keihin-test-4-1000-w640h480-fine/images/train"
LABEL_FOLDER="../datasets/keihin-test-4-1000-w640h480-fine/labels/train"

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
