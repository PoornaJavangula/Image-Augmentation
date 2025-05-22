import cv2
import numpy as np
import os

# Configuration
FOLDER_NAME = "augmented_image_part4"
EXTENSION = ".jpg"
os.makedirs(FOLDER_NAME, exist_ok=True)

def save_image(name, image):
    path = os.path.join(FOLDER_NAME, f"{name}{EXTENSION}")
    cv2.imwrite(path, image)

# Augmentation functions
def scale_image(image, fx, fy):
    scaled = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    save_image(f"Scale-{fx}_{fy}", scaled)

def translate_image(image, x, y):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(image, M, (cols, rows))
    save_image(f"Translate-{x}_{y}", translated)

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    save_image(f"Rotate-{angle}", rotated)

def apply_affine_transforms(image):
    rows, cols = image.shape[:2]
    base_pts = np.float32([[50, 50], [200, 50], [50, 200]])
    transform_pts_list = [
        [[10, 100], [200, 50], [100, 250]],
        [[100, 10], [200, 50], [0, 150]],
        [[100, 10], [200, 50], [30, 175]],
        [[100, 10], [200, 50], [70, 150]]
    ]
    
    for i, target_pts in enumerate(transform_pts_list, 1):
        target_pts = np.float32(target_pts)
        M = cv2.getAffineTransform(base_pts, target_pts)
        transformed = cv2.warpAffine(image, M, (cols, rows))
        save_image(f"Transform-{i}", transformed)

# Main execution
if __name__ == "__main__":
    image_file = "flower.jpg"
    image = cv2.imread(image_file)

    # Scaling
    for fx, fy in [(0.3, 0.3), (0.7, 0.7), (2, 2), (3, 3)]:
        scale_image(image, fx, fy)

    # Translation
    for x, y in [(150, 150), (-150, 150), (150, -150), (-150, -150)]:
        translate_image(image, x, y)

    # Rotation
    for angle in [90, 180, 270]:
        rotate_image(image, angle)

    # Affine Transformations
    apply_affine_transforms(image)
