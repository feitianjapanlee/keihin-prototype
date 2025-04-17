import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_main_contour(image_path, threshold=128, visualize=True):
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("画像を読み込めませんでした。パスを確認してください。")
    
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # しきい値処理で二値化
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("輪郭が見つかりませんでした。")
    
    # 面積が最大の輪郭を取得（主要な物体と仮定）
    main_contour = max(contours, key=cv2.contourArea)
    
    # 輪郭をPolygon形式に変換 (x1,y1,x2,y2,...)
    polygon = main_contour.flatten().tolist()
    
    if visualize:
        # 可視化用の画像を作成
        vis_image = image.copy()
        cv2.drawContours(vis_image, [main_contour], -1, (0, 255, 0), 3)
        
        # 元画像と処理結果を並べて表示
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title("Detected Contour")
        plt.axis('off')
        
        plt.show()
    
    return polygon

# 使用例
if __name__ == "__main__":
    try:
        # 画像パスを指定
        image_path = "dogclip1.jpg"  # ここを実際の画像パスに変更
        
        # 主要な輪郭を取得
        polygon = get_main_contour(image_path)
        
        # 結果を表示
        print("輪郭のPolygon形式:")
        print(polygon)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
