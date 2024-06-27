import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np


def align_images(source, target):
    # images already converted to grayscale
    src = source
    tgt = target

    # find size of image
    size = src.shape

    # define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # define 2x3 or 3x3 matrices and initalize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # specify the number of iterations
    number_of_iterations = 5000

    # specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # define termination criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps,
    )

    # run the ECC algorithm. the results are stored in warp_matrix
    (cc, warp_matrix) = cv2.findTransformECC(src, tgt, warp_matrix, warp_mode, criteria)

    # use warpAffine for Translation, Euclidean and Affine
    aligned_image = cv2.warpAffine(
        tgt,
        warp_matrix,
        (size[1], size[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )

    return aligned_image


def get_image_size(source):
    from PIL import Image

    img = Image.open(source)
    width, height = img.size
    print(f"Image size ({width} x {height})")
    return (width, height)


def compare_ssim_with_align(source, target):
    src = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2GRAY)
    tgt = cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2GRAY)
    aligned_tgt = align_images(src, tgt)
    gray1 = src
    gray2 = aligned_tgt

    score, _ = ssim(gray1, gray2, full=True)
    return score


def compare_ssim(source, target):
    src = cv2.cvtColor(cv2.imread(source), cv2.COLOR_BGR2GRAY)
    tgt = cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2GRAY)
    gray1 = src
    gray2 = tgt

    score, _ = ssim(gray1, gray2, full=True)
    return score


if __name__ == "__main__":
    source_name = "sample-1.jpg"
    source_path = os.path.join("./ImageCompare/images/", source_name)

    for target_name in [f"sample-{i}.jpg" for i in range(2, 11)]:
        print(f"[Compare `{source_name}` vs. `{target_name}`]")
        target_path = os.path.join("./ImageCompare/images/", target_name)
        score = compare_ssim(source_path, target_path)
        print(f"SSIM: {score}\n")

    for target_name in [f"move-{i}.jpg" for i in range(1, 6)]:
        print(f"[Compare `{source_name}` vs. `{target_name}`]")
        target_path = os.path.join("./ImageCompare/images/", target_name)
        score = compare_ssim(source_path, target_path)
        print(f"SSIM: {score}")
        score = compare_ssim_with_align(source_path, target_path)
        print(f"SSIM(align): {score}\n")
