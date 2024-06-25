import cv2
import numpy as np


def otsu_binarization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive_binarization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary


def apply_morphology(binary_image, operation, iterations):
    kernel = np.ones((3, 3), np.uint8)
    if operation == 'erosion':
        result = cv2.erode(binary_image, kernel, iterations=iterations)
    elif operation == 'dilation':
        result = cv2.dilate(binary_image, kernel, iterations=iterations)
    elif operation == 'opening':
        result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'closing':
        result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("Invalid morphology operation. Choose from 'erosion', 'dilation', 'opening', 'closing'.")
    return result


def main():
    # Load the image
    image = cv2.imread('Lena.png')

    if image is None:
        print("Error loading image.")
        return

    # Show the original image
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # Get binarization method from user
    binarization_method = input("Choose binarization method (otsu/adaptive): ").strip().lower()

    if binarization_method == 'otsu':
        binary_image = otsu_binarization(image)
    elif binarization_method == 'adaptive':
        binary_image = adaptive_binarization(image)
    else:
        print("Invalid binarization method.")
        return

    # Show the binarized image
    cv2.imshow('Binarized Image', binary_image)
    cv2.waitKey(0)

    # Get morphology operation and iterations from user
    operation = input("Choose morphology operation (erosion/dilation/opening/closing): ").strip().lower()
    iterations = int(input("Enter the number of iterations: "))

    # Apply morphology operation
    morphed_image = apply_morphology(binary_image, operation, iterations)

    # Show the morphology result
    cv2.imshow(f'{operation.capitalize()} Result', morphed_image)
    cv2.waitKey(0)

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
