import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def apply_median_filter(image):
    return cv2.medianBlur(image, 5)


def apply_bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


def compute_absolute_difference(image1, image2):
    return cv2.absdiff(image1, image2)


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Load the image
    image = cv2.imread('Lena.png')

    if image is None:
        print("Error loading image.")
        return

    # Show the original image
    show_image('Original Image', image)

    # Add noise to the image
    noisy_image = add_noise(image)
    show_image('Noisy Image', noisy_image)

    # Apply Gaussian Filter
    gaussian_filtered = apply_gaussian_filter(noisy_image)
    show_image('Gaussian Filtered Image', gaussian_filtered)

    # Apply Median Filter
    median_filtered = apply_median_filter(noisy_image)
    show_image('Median Filtered Image', median_filtered)

    # Apply Bilateral Filter
    bilateral_filtered = apply_bilateral_filter(noisy_image)
    show_image('Bilateral Filtered Image', bilateral_filtered)

    # Compute absolute difference
    diff_gaussian = compute_absolute_difference(image, gaussian_filtered)
    diff_median = compute_absolute_difference(image, median_filtered)
    diff_bilateral = compute_absolute_difference(image, bilateral_filtered)

    # Show absolute differences
    show_image('Difference after Gaussian Filtering', diff_gaussian)
    show_image('Difference after Median Filtering', diff_median)
    show_image('Difference after Bilateral Filtering', diff_bilateral)


if __name__ == "__main__":
    main()
