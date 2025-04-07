import cv2
import numpy as np
import os

count = 0
cpf = 10                        #何フレーム毎に切り出すか

# 画像のサイズ
image_width = 640
image_heigh = 480
# ボケ画像判定のためのlaplacian.var
laplacian_thr = 50             #ボケ画像判定をするときのスレッショルド

# 動画の読み込み
cap = cv2.VideoCapture('./mp4/4.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()                   #動画を読み込む

    if ret == False:
        print('Finished')                    #動画の切り出しが終了した時
        break

    if count%cpf == 0:                      #何フレームに１回切り出すか

        #サイズを小さくする
        resize_frame = cv2.resize(frame,(image_width,image_heigh))

         #画像がぶれていないか確認する
        laplacian = cv2.Laplacian(resize_frame, cv2.CV_64F)

        if ret and laplacian.var() >= laplacian_thr: # ピンぼけ判定がしきい値以上のもののみ出力
            
            #第１引数画像のファイル名、第２引数保存したい画像
            jpg_name = f'./images/MP4_4_{count:0=5}.jpg'
            write = cv2.imwrite(jpg_name, resize_frame)  # 切り出した画像を表示する
            assert write, "保存に失敗"
            print(f'Save {jpg_name}')          #確認用表示
    
    count = count + 1

cap.release()
