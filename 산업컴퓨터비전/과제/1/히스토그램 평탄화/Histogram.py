import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(image, channel_name):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Plot histogram
    plt.figure()
    plt.title(f'Histogram for {channel_name} channel')
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


def equalize_histogram(image):
    # Equalize the histogram
    equalized_image = cv2.equalizeHist(image)
    return equalized_image


def process_rgb_channel(image, channel):
    # Split the image into R, G, B channels
    channels = list(cv2.split(image))

    if channel == 'R':
        index = 2
    elif channel == 'G':
        index = 1
    elif channel == 'B':
        index = 0
    else:
        raise ValueError("Invalid channel. Choose from 'R', 'G', or 'B'.")

    # Get the specified channel
    channel_image = channels[index]

    # Plot the histogram before equalization
    plot_histogram(channel_image, channel)

    # Equalize the histogram
    equalized_channel = equalize_histogram(channel_image)

    # Replace the original channel with the equalized one
    channels[index] = equalized_channel
    equalized_image = cv2.merge(channels)

    return equalized_image


def process_v_channel(image):
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the HSV channels
    h, s, v = cv2.split(hsv_image)

    # Plot the histogram for the V channel before equalization
    plot_histogram(v, 'V')

    # Equalize the histogram of the V channel
    equalized_v = equalize_histogram(v)

    # Merge the channels back
    equalized_hsv = cv2.merge([h, s, equalized_v])

    # Convert the HSV image back to BGR
    equalized_image = cv2.cvtColor(equalized_hsv, cv2.COLOR_HSV2BGR)

    return equalized_image


def main():
    # Load the image
    image = cv2.imread('Lena.png')

    if image is None:
        print("Error loading image.")
        return

    # Show the original image
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # Get the channel from the user
    channel = input("Enter the channel to process (R, G, B): ").strip().upper()

    # Process the specified RGB channel
    equalized_image_rgb = process_rgb_channel(image, channel)

    # Show the image after RGB channel equalization
    cv2.imshow(f'Image after {channel} channel equalization', equalized_image_rgb)
    cv2.waitKey(0)

    # Process the V channel
    equalized_image_v = process_v_channel(image)

    # Show the image after V channel equalization
    cv2.imshow('Image after V channel equalization', equalized_image_v)
    cv2.waitKey(0)

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
