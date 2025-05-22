import cv2
import numpy as np
import sys
import os

# Output folder and extension
FOLDER_NAME = "augmented_image_part3"
EXTENSION = ".jpg"
os.makedirs(FOLDER_NAME, exist_ok=True)

# Load image
image_file = "flower.jpg"
image = cv2.imread(image_file)

if image is None:
    print(f"Error: Cannot read image '{image_file}'. Please check the file path.")
    sys.exit(1)

# Augmentation functions
def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def emboss_image(img):
    kernel_emboss = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
    return cv2.filter2D(img, -1, kernel_emboss) + 128

def edge_image(img, ksize):
    return cv2.Sobel(img, cv2.CV_16U, 1, 0, ksize=ksize)

def addeptive_gaussian_noise(img):
    h, s, v = cv2.split(img)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return cv2.merge([h, s, v])

def salt_image(img, p, a):
    noisy = img.copy()
    num_salt = np.ceil(a * img.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy[tuple(coords)] = 255
    return noisy

def paper_image(img, p, a):
    noisy = img.copy()
    num_pepper = np.ceil(a * img.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
    noisy[tuple(coords)] = 0
    return noisy

def salt_and_paper_image(img, p, a):
    noisy = img.copy()
    noisy = salt_image(noisy, p, a)
    noisy = paper_image(noisy, p, a)
    return noisy

def contrast_image(img, contrast):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = np.clip(np.where(img[:, :, 2] < 190, img[:, :, 2] - contrast, img[:, :, 2] + contrast), 0, 255)
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def edge_detect_canny_image(img, th1, th2):
    return cv2.Canny(img, th1, th2)

def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Augmentation mapping
augmentations = {
    0: ("Sharpen", lambda img: sharpen_image(img)),
    1: ("Emboss", lambda img: emboss_image(img)),
    2: ("Edge Sobel 1", lambda img: edge_image(img, 1)),
    3: ("Edge Sobel 3", lambda img: edge_image(img, 3)),
    4: ("Edge Sobel 5", lambda img: edge_image(img, 5)),
    5: ("Edge Sobel 9", lambda img: edge_image(img, 9)),
    6: ("Adaptive Gaussian Noise", lambda img: addeptive_gaussian_noise(img)),
    7: ("Salt 0.5,0.009", lambda img: salt_image(img, 0.5, 0.009)),
    8: ("Salt 0.5,0.09", lambda img: salt_image(img, 0.5, 0.09)),
    9: ("Salt 0.5,0.9", lambda img: salt_image(img, 0.5, 0.9)),
    10: ("Pepper 0.5,0.009", lambda img: paper_image(img, 0.5, 0.009)),
    11: ("Pepper 0.5,0.09", lambda img: paper_image(img, 0.5, 0.09)),
    12: ("Pepper 0.5,0.9", lambda img: paper_image(img, 0.5, 0.9)),
    13: ("Salt & Pepper 0.5,0.009", lambda img: salt_and_paper_image(img, 0.5, 0.009)),
    14: ("Salt & Pepper 0.5,0.09", lambda img: salt_and_paper_image(img, 0.5, 0.09)),
    15: ("Salt & Pepper 0.5,0.9", lambda img: salt_and_paper_image(img, 0.5, 0.9)),
    16: ("Contrast 25", lambda img: contrast_image(img, 25)),
    17: ("Contrast 50", lambda img: contrast_image(img, 50)),
    18: ("Contrast 100", lambda img: contrast_image(img, 100)),
    19: ("Edge Canny 100-200", lambda img: edge_detect_canny_image(img, 100, 200)),
    20: ("Edge Canny 200-400", lambda img: edge_detect_canny_image(img, 200, 400)),
    21: ("Grayscale", lambda img: grayscale_image(img))
}

# Apply augmentation
def apply_augmentation(index):
    try:
        name, func = augmentations[index]
        result = func(image)
        filename = f"{FOLDER_NAME}/{name.replace(' ', '_')}{EXTENSION}"
        cv2.imwrite(filename, result)
        cv2.imshow(f"Augmentation: {name}", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except KeyError:
        print("Invalid index. Please choose a valid augmentation index.")

# Main logic
if len(sys.argv) > 1:
    try:
        idx = int(sys.argv[1])
        apply_augmentation(idx)
    except ValueError:
        print("Invalid argument. Please provide a numeric index.")
else:
    print("Available augmentation techniques:")
    for idx, (name, _) in augmentations.items():
        print(f"{idx}: {name}")
    try:
        selected = int(input("Enter the index of the augmentation to apply: "))
        apply_augmentation(selected)
    except ValueError:
        print("Invalid input. Please enter a number.")
