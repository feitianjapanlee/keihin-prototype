# AI画像識別で景品の識別と計数　プロトタイプ
- プライズゲーム機のブース出口に落ちた景品を識別し、種類毎にカウント、結果を外部に送信
- ターゲットデバイスはMaixCam（Sipeed社のシングルボードコンピューター）
- YOLOv5/v8で試作
- カスタムデータセットはカーゾック吉祥寺店にてプレーして獲得した景品４種各１個
## 使うツール
- **MaixVision**：MaixCamへのDeployやファイルコピーなど
- **MaixHub**：初期のデータセットの作成とアノテーション、その後はオフラインでtrain
- **MaixCam**：ターゲットデバイスですが、カスタムデータセットの画像を撮影するためにも使う

## 参考資料
- wiki.sipeed.com
- https://github.com/sipeed/MaixPy
- https://farml1.com/yolov8/
- https://blog.csdn.net/m0_75041317/category_12808588.html

## メモ
オフラインtrainのDocker環境構築
```bash
nvidia-smi
cd \workspace\keihin-prototype
docker run -it --shm-size=2g --gpus all -p 8888:8888 -v ${PWD}:/workspace nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 bash
cd workspace
apt update -y && apt upgrade -y && apt install python3 python3.10-venv libopencv-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook --allow-root --ip="0.0.0.0"
```
