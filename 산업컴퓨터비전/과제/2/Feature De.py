import cv2
import numpy as np
import glob
import os


def load_images_from_folder(folder):
    images = {}
    for filename in glob.glob(os.path.join(folder, '*.jpg')):
        img = cv2.imread(filename)
        if img is not None:
            images[os.path.basename(filename)] = img
    return images


def detect_canny_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges


def detect_harris_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    return image


def main():
    folder = 'Stitching'  # 폴더 경로를 설정하세요
    images = load_images_from_folder(folder)

    selected_images = ['stitching/boat1.jpg', 'stitching/budapest1.jpg', 'stitching/newspaper1.jpg', 'stitching/s1.jpg']

    for img_name in selected_images:
        if img_name in images:
            image = images[img_name]

            # Detect Canny edges
            edges = detect_canny_edges(image)
            cv2.imshow(f'Canny Edges - {img_name}', edges)

            # Detect Harris corners
            corners = detect_harris_corners(image.copy())
            cv2.imshow(f'Harris Corners - {img_name}', corners)

            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
