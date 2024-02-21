import cv2
import numpy as np

# Function to load images
def load_images(image_path_1, image_path_2):
    image1 = cv2.imread(image_path_1)
    image2 = cv2.imread(image_path_2)
    return image1, image2

# Function to find and match ORB features
def find_and_match_features(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

# Function to estimate essential matrix and recover pose
def estimate_pose(kp1, kp2, matches, K):
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    return R, t

# Main workflow
if __name__ == "__main__":
    # Update these paths with the actual paths to your images
    image_path_1 = '/home/runyi/Project/CP6D/chess/seq-01/frame-000000.color.png'
    image_path_2 = '/home/runyi/Project/CP6D/chess/seq-01/frame-000001.color.png'

    # Load images
    image1, image2 = load_images(image_path_1, image_path_2)

    # Find and match features
    kp1, kp2, matches = find_and_match_features(image1, image2)

    # Default intrinsic parameters (given for the depth camera, used here for demonstration)
    K = np.array([[585, 0, 320],
                  [0, 585, 240],
                  [0, 0, 1]], dtype=np.float32)

    # Estimate pose
    R, t = estimate_pose(kp1, kp2, matches, K)

    print("Estimated Rotation:\n", R)
    print("Estimated Translation:\n", t)
