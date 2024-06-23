import cv2
import numpy as np
#from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import cosine_similarity


# 1. K-means 클러스터링을 이용한 이미지 분할
def kmeans_segmentation(image, k=2):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    return segmented_image


# 2. SIFT를 이용한 특징 검출
def detect_and_compute_sift(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


# 3. ORB를 이용한 특징 검출
def detect_and_compute_orb(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


# 4. FLANN을 이용한 특징 매칭
def match_features_flann(des1, des2):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


# 5. RANSAC을 이용한 외곽점 제거 및 호모그래피 추정
def find_homography_ransac(kp1, kp2, matches):
    if len(matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M, mask
    else:
        return None, None


# 6. Cosine Similarity를 이용한 디스크립터 비교
def cosine_similarity_match(des1, des2):
    sim_matrix = cosine_similarity(des1, des2)
    good_matches = []
    for i in range(sim_matrix.shape[0]):
        best_match = np.argmax(sim_matrix[i])
        good_matches.append((i, best_match))
    return good_matches


# 부분 이미지와 여러 장의 완성 이미지 목록
part_image = cv2.imread('part_image.jpg')
whole_images = ['whole_image1.jpg', 'whole_image2.jpg', 'whole_image3.jpg']

# K-means 클러스터링으로 이미지 분할
segmented_part = kmeans_segmentation(part_image)

# SIFT와 ORB로 부분 이미지에서 특징 검출
kp1_sift, des1_sift = detect_and_compute_sift(segmented_part)
kp1_orb, des1_orb = detect_and_compute_orb(segmented_part)

# 일치하는 이미지 찾기
matching_image = None
for image_path in whole_images:
    whole_image = cv2.imread(image_path)
    segmented_whole = kmeans_segmentation(whole_image)

    # SIFT와 ORB로 전체 이미지에서 특징 검출
    kp2_sift, des2_sift = detect_and_compute_sift(segmented_whole)
    kp2_orb, des2_orb = detect_and_compute_orb(segmented_whole)

    # FLANN을 이용한 SIFT 매칭
    good_matches_sift = match_features_flann(des1_sift, des2_sift)

    # RANSAC을 이용한 호모그래피 추정 및 검증 (SIFT)
    M_sift, mask_sift = find_homography_ransac(kp1_sift, kp2_sift, good_matches_sift)

    if M_sift is not None:
        matching_image = image_path
        break

if matching_image:
    print(f"Matching image found: {matching_image}")
else:
    print("No matching image found.")