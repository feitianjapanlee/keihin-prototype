import os
import cv2
import numpy as np

import albumentations as A

def create_augmentations():
    """Create a list of augmentation pipelines"""
    augmentations = [
        # A.MedianBlur(p=1),
        # A.GaussianBlur(p=1),
        # A.MotionBlur(p=1),
        # A.CLAHE(p=1),
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.GaussNoise(p=1, std_range=(0.05, 0.1)),
        A.RandomRotate90(p=1),
        A.RandomGamma(p=1),
        A.RandomBrightnessContrast(p=1),
        # A.ImageCompression(p=1),
        # A.ISONoise(p=1),
        # A.HueSaturationValue(p=1),
        # A.RandomSunFlare(p=1),
        # A.RandomShadow(p=1),
        # A.CoarseDropout(p=1),
    ]
    
    return [A.Compose(
        [aug], 
        bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0.1)
    ) for aug in augmentations]

def process_image(image_path, label_path, aug_transform, output_dir, subset_dir, idx):
    """Process single image with augmentation"""
    print(f"Processing {image_path} aug idx {idx}")
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read YOLO format labels
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                bboxes.append([x, y, w, h, class_id])
    
    # Apply augmentation
    transformed = aug_transform(image=image, bboxes=bboxes)
    
    # Save augmented image
    aug_image = cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR)
    aug_image_path = os.path.join(output_dir, f'images/{subset_dir}', 
                                 f"{os.path.splitext(os.path.basename(image_path))[0]}_{idx}.jpg")
    cv2.imwrite(aug_image_path, aug_image)
    
    # Save augmented labels
    aug_label_path = os.path.join(output_dir, f'labels/{subset_dir}',
                                 f"{os.path.splitext(os.path.basename(image_path))[0]}_{idx}.txt")
    with open(aug_label_path, 'w') as f:
        for bbox in transformed['bboxes']:
            x, y, w, h, class_id = bbox
            f.write(f"{int(class_id)} {x} {y} {w} {h}\n")

def main(org_dataset_dir, subset_dir):
    # Get augmentation transforms
    augmentations = create_augmentations()
    
    # Set paths
    dataset_dir = org_dataset_dir
    output_dir = org_dataset_dir[:-1] if org_dataset_dir.endswith('/') else org_dataset_dir
    output_dir = f'{output_dir}-aug{len(augmentations)}x/'
   
    # Create output directories
    os.makedirs(os.path.join(output_dir, f'images/{subset_dir}'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'labels/{subset_dir}'), exist_ok=True)
    
    # Process all images
    images_dir = os.path.join(dataset_dir, f'images/{subset_dir}')
    labels_dir = os.path.join(dataset_dir, f'labels/{subset_dir}')
    
    for image_file in os.listdir(images_dir):
        if not image_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, 
                                os.path.splitext(image_file)[0] + '.txt')
        
        # Apply each augmentation
        for idx, aug in enumerate(augmentations):
            process_image(image_path, label_path, aug, output_dir, subset_dir, idx)

if __name__ == "__main__":
    main('../datasets/keihin-test-4-1000-w640h480-fine/', 'train')
    # main('../datasets/keihin-test-4-1000-w640h480-fine/', 'val') # Usually do not augment validation set
