# 動画をフレーム毎に切り出して保存するプログラム
# 動画のパスをコマンドライン引数から取得する
import cv2
import os
import sys

cut_frame_per = 10                        #何フレーム毎に切り出すか
# 画像のサイズ
image_width = 640
image_heigh = 480
# ボケ画像判定のためのlaplacian.var
laplacian_thr = 50             #ボケ画像判定をするときのスレッショルド

total_cut = 0

# コマンドライン引数から動画のdir,切り出した画像のdir,cut_frame_perを取得
if len(sys.argv) < 2:
    print("Usage: python mp42jpg.py <video_dir> <jpg_dir> <cut_frame_per>")
    exit()

mp4_dir = sys.argv[1]
if not os.path.exists(mp4_dir):
    print(f"Directory {mp4_dir} does not exist")
    exit()

if len(sys.argv) > 2:
    if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])
    print(f"Directory {sys.argv[2]} does not exist, creating it")
    jpg_dir = sys.argv[2]

if len(sys.argv) > 3:
    cut_frame_per = int(sys.argv[3])
    if cut_frame_per <= 0:
        cut_frame_per = 10
    print(f"Cut frame per: {cut_frame_per}")
else:
    print("Using default cut frame per: 10")

# 動画dirの中にあるすべて動画ファイルを読み込み
mp4_list = os.listdir(mp4_dir)
mp4_list = [f for f in mp4_list if f.endswith('.mp4')]
if len(mp4_list) == 0:
    print("No mp4 files found in the directory")
    exit()

for mp4 in mp4_list:
    frame_count = 0
    mp4_path = os.path.join(mp4_dir, mp4)
    print(f"Processing {mp4_path}")
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"Could not open video file {mp4_path}")
        exit()

    while(cap.isOpened()):
        ret, frame = cap.read()                   #動画を読み込む

        if ret == False:
            print(f'Finished {mp4_path}')                    #動画の切り出しが終了した時
            break

        if frame_count%cut_frame_per == 0:                      #何フレームに１回切り出すか

            #サイズを小さくする
            resize_frame = cv2.resize(frame,(image_width,image_heigh))

            #画像がぶれていないか確認する
            laplacian = cv2.Laplacian(resize_frame, cv2.CV_64F)

            if ret and laplacian.var() >= laplacian_thr: # ピンぼけ判定がしきい値以上のもののみ出力
                
                jpg_name = f'{os.path.splitext(mp4)[0]}_{frame_count//cut_frame_per:0=5}.jpg' # 切り出した画像のファイル名
                jpg_path = os.path.join(jpg_dir, jpg_name)
                write = cv2.imwrite(jpg_path, resize_frame)  # 切り出した画像を表示する
                assert write, "Error: Failed to save image"
                print(f'Save {jpg_path}')          #確認用表示
                total_cut = total_cut + 1
        frame_count = frame_count + 1       
    cap.release()
print(f"Finished all {len(mp4_list)} videos. Total cuts: {total_cut}")


