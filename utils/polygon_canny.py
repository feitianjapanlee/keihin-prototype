import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_main_contour_improved(image_path, visualize=True):
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("画像を読み込めませんでした。パスを確認してください。")
    
    # ガウシアンブラーでノイズ低減
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # グレースケール変換
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    
    # 適応的しきい値処理
    binary = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # エッジ検出 (Canny)
    edges = cv2.Canny(gray, 30, 100)
    
    # 二値画像とエッジ画像を結合
    combined = cv2.bitwise_or(binary, edges)
    
    # モルフォロジー操作でノイズ除去
    kernel = np.ones((3,3), np.uint8)
    refined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 輪郭検出
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("輪郭が見つかりませんでした。")
    
    # 面積が最大の輪郭を取得
    main_contour = max(contours, key=cv2.contourArea)
    
    # 輪郭を近似（不要な点を削減）
    epsilon = 0.005 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Polygon形式に変換
    polygon = approx.flatten().tolist()
    
    if visualize:
        # 可視化用の画像を作成
        vis_image = image.copy()
        cv2.drawContours(vis_image, [approx], -1, (0, 255, 0), 3)
        
        # 処理過程を可視化
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(binary, cmap='gray')
        plt.title("Adaptive Threshold")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(edges, cmap='gray')
        plt.title("Edge Detection")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(combined, cmap='gray')
        plt.title("Combined Mask")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(refined, cmap='gray')
        plt.title("Refined Mask")
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title("Final Result")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return polygon

# 使用例
if __name__ == "__main__":
    try:
        image_path = "dogclip1.jpg"  # 画像パスを指定
        polygon = get_main_contour_improved(image_path)
        
        print("輪郭のPolygon形式:")
        print(polygon)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
