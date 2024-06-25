import cv2
import numpy as np
import matplotlib.pyplot as plt


def dft(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform DFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Calculate magnitude spectrum
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return dft_shift, magnitude_spectrum


def create_bandpass_filter(shape, radius1, radius2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols, 2), np.uint8)

    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if radius1 < distance < radius2:
                mask[i, j] = 1

    return mask


def apply_bandpass_filter(dft_shift, mask):
    # Apply mask
    fshift = dft_shift * mask

    # Inverse DFT to get the image back
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back


def main():
    # Load the image
    image = cv2.imread('Lena.png')

    if image is None:
        print("Error loading image.")
        return

    # Perform DFT
    dft_shift, magnitude_spectrum = dft(image)

    # Show magnitude spectrum
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

    # Get radii from user
    radius1 = int(input("Enter the inner radius: "))
    radius2 = int(input("Enter the outer radius: "))

    # Create bandpass filter
    mask = create_bandpass_filter(dft_shift.shape[:2], radius1, radius2)

    # Apply bandpass filter
    filtered_image = apply_bandpass_filter(dft_shift, mask)

    # Show the filtered image
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
