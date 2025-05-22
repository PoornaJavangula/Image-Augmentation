import cv2
import numpy as np
import sys
import os

# Output folder
FOLDER_NAME = "augmented_image_part2"
EXTENSION = ".jpg"
os.makedirs(FOLDER_NAME, exist_ok=True)

# Load image
image_file = "flower.jpg"
image = cv2.imread(image_file)

if image is None:
    print(f"Error: Cannot read image '{image_file}'.")
    sys.exit(1)

# Augmentation functions
def multiply_image(image, R, G, B):
    result = np.clip(image * [R, G, B], 0, 255).astype(np.uint8)
    return result

def gaussian_blur(image, blur):
    return cv2.GaussianBlur(image, (5, 5), blur)

def averaging_blur(image, shift):
    return cv2.blur(image, (shift, shift))

def median_blur(image, shift):
    return cv2.medianBlur(image, shift)

def bilateral_blur(image, d, color, space):
    return cv2.bilateralFilter(image, d, color, space)

def erosion_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def dilation_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def opening_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def morphological_gradient_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

def top_hat_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

def black_hat_image(image, shift):
    kernel = np.ones((shift, shift), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

# Augmentation techniques dictionary
augmentations = {
    0: ("Multiply RGB (0.5,1,1)", lambda img: multiply_image(img, 0.5, 1, 1)),
    1: ("Multiply RGB (1,0.5,1)", lambda img: multiply_image(img, 1, 0.5, 1)),
    2: ("Multiply RGB (1,1,0.5)", lambda img: multiply_image(img, 1, 1, 0.5)),
    3: ("Gaussian Blur σ=1", lambda img: gaussian_blur(img, 1)),
    4: ("Averaging Blur k=5", lambda img: averaging_blur(img, 5)),
    5: ("Median Blur k=3", lambda img: median_blur(img, 3)),
    6: ("Bilateral Blur d=9", lambda img: bilateral_blur(img, 9, 75, 75)),
    7: ("Erosion k=3", lambda img: erosion_image(img, 3)),
    8: ("Dilation k=3", lambda img: dilation_image(img, 3)),
    9: ("Opening k=3", lambda img: opening_image(img, 3)),
    10: ("Closing k=3", lambda img: closing_image(img, 3)),
    11: ("Morphological Gradient k=5", lambda img: morphological_gradient_image(img, 5)),
    12: ("Top Hat k=200", lambda img: top_hat_image(img, 200)),
    13: ("Black Hat k=200", lambda img: black_hat_image(img, 200)),
}

# Function to apply augmentation and show result
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
