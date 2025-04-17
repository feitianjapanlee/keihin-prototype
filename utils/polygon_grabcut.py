import cv2
import numpy as np
import matplotlib.pyplot as plt

def grabcut_contour(image_path, rect=None, visualize=True):
    # 画像読み込み
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("画像を読み込めませんでした。パスを確認してください。")
    
    # 矩形領域が指定されていない場合、画像全体を使用
    if rect is None:
        rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
    
    # GrabCut用のマスク初期化
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # GrabCutで使用する一時配列
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # GrabCut実行
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # マスクの修正 (背景:0, 前景:1)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # マスクを画像に適用
    segmented = image * mask[:, :, np.newaxis]
    
    # 輪郭検出用にグレースケール変換
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    
    # 輪郭検出
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("輪郭が見つかりませんでした。")
    
    # 面積が最大の輪郭を取得
    main_contour = max(contours, key=cv2.contourArea)
    
    # 輪郭を近似
    epsilon = 0.005 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    # Polygon形式に変換
    polygon = approx.flatten().tolist()
    
    if visualize:
        # 可視化用の画像を作成
        vis_image = image.copy()
        cv2.drawContours(vis_image, [approx], -1, (0, 255, 0), 3)
        
        # マスク可視化
        mask_vis = mask * 255
        
        # 処理過程を可視化
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(mask_vis, cmap='gray')
        plt.title("GrabCut Mask")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
        plt.title("Segmented Image")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale for Contour")
        plt.axis('off')
        
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, [approx], -1, (0, 255, 0), 2)
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Contour")
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
        
        # 必要に応じて矩形領域を手動指定 (x,y,w,h)
        # rect = (50, 50, 300, 300)
        
        polygon = grabcut_contour(image_path, rect=None)
        
        print("輪郭のPolygon形式:")
        print(polygon)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
