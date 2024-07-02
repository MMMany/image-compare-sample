import random
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

image = cv2.imread(cv2.samples.findFile("sample-1.jpg"))


def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image, alpha):
    return cv2.addWeighted(
        image, alpha, np.zeros(image.shape, image.dtype), 0, 128 * (1 - alpha)
    )


def adjust_saturation(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, value)
    s[s > 255] = 255
    s[s < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def adjust_hue(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = cv2.add(h, value)
    h[h > 180] = 180
    h[h < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


# Define range of image values
brightness_range = np.arange(-20, 21)
contrast_range = np.arange(0.5, 1.51, 0.05)
saturation_range = np.arange(-50, 51)
hue_range = np.arange(-10, 11)

# Define list for saving
pos_images_98 = []
pos_images_95 = []
neg_images = []

# Define constants
GEN_LIMITS_98 = 50
GEN_LIMITS_95 = 50
GEN_LIMITS = GEN_LIMITS_98 + GEN_LIMITS_95
SSIM_THRESHOLD_95 = 0.95
SSIM_THRESHOLD_98 = 0.98
SSIM_THRESHOLD_MARGIN = 0.05

# Convert original image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Generate adjusted images
print("Start image generation")
time_start = perf_counter()
time_monitor = perf_counter()
value_adjusted_1 = False
value_adjusted_2 = False
while len(pos_images_98) + len(pos_images_95) + len(neg_images) < GEN_LIMITS * 2:
    # Monitor
    if perf_counter() - time_monitor > 1:
        time_monitor = perf_counter()
        print(
            " - "
            + ", ".join(
                [
                    f"P98({len(pos_images_98)})",
                    f"P95({len(pos_images_95)})",
                    f"NEG({len(neg_images)})",
                ]
            )
        )

    # Get adjusting values
    b = int(random.choice(brightness_range))
    c = float(random.choice(contrast_range))
    s = int(random.choice(saturation_range))
    h = int(random.choice(hue_range))

    # Adjust original image
    adjusted_image = adjust_brightness(image, b)
    adjusted_image = adjust_contrast(adjusted_image, c)
    adjusted_image = adjust_saturation(adjusted_image, s)
    adjusted_image = adjust_hue(adjusted_image, h)

    # Convert adjusted image to grayscale
    gray_adjusted = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM score
    score = ssim(gray_image, gray_adjusted)

    # Save
    if score > SSIM_THRESHOLD_98:
        if len(pos_images_98) >= GEN_LIMITS_98:
            continue
        pos_images_98.append((adjusted_image, score, (b, c, s, h)))
    elif score > SSIM_THRESHOLD_95:
        if len(pos_images_95) >= GEN_LIMITS_95:
            # Adjust image values range
            if not value_adjusted_2:
                value_adjusted_2 = True
                print(" - Image values adjusted 2")
                brightness_range = np.arange(-5, 6)
                contrast_range = np.arange(0.85, 1.16, 0.02)
                saturation_range = np.arange(-12, 13)
                hue_range = np.arange(-3, 4)
            continue
        pos_images_95.append((adjusted_image, score, (b, c, s, h)))
    elif score > SSIM_THRESHOLD_95 - SSIM_THRESHOLD_MARGIN:
        # Skip 'threshold - margin' score image
        continue
    else:
        if len(neg_images) >= GEN_LIMITS:
            # Adjust image values range
            if not value_adjusted_1:
                value_adjusted_1 = True
                print(" - Image values adjusted 1")
                brightness_range = np.arange(-10, 11)
                contrast_range = np.arange(0.75, 1.26, 0.05)
                saturation_range = np.arange(-25, 26)
                hue_range = np.arange(-5, 6)
            continue
        neg_images.append((adjusted_image, score, (b, c, s, h)))

print(f"Done image generation (Elapsed: {perf_counter() - time_start:.2f} sec)\n")

# Check output directories
output_dir = Path("./output")
if not output_dir.exists():
    print(f"Directory not found, create new - {output_dir.as_posix()}")
    output_dir.mkdir()

output_pos = output_dir.joinpath("pos_images")
if not output_pos.exists():
    print(f"Directory not found, create new - {output_pos.as_posix()}")
    output_pos.mkdir()

output_neg = output_dir.joinpath("neg_images")
if not output_neg.exists():
    print(f"Directory not found, create new - {output_neg.as_posix()}")
    output_neg.mkdir()

# Save image and information to file
with open(output_dir.joinpath("info.csv"), "w", encoding="utf8") as info:
    for idx, (img, score, (b, c, s, h)) in enumerate(pos_images_98):
        img_name = output_pos.joinpath(f"positive-{idx + 1}.jpg")
        cv2.imwrite(img_name, img)
        info.write(f"{img_name.as_posix()},{score:.4f},{b},{c:.2f},{s},{h}\n")
    for idx, (img, score, (b, c, s, h)) in enumerate(neg_images):
        img_name = output_neg.joinpath(f"negative-{idx + 1}.jpg")
        cv2.imwrite(img_name, img)
        info.write(f"{img_name.as_posix()},{score:.4f},{b},{c:.2f},{s},{h}\n")

print(f"Positive 98 : {np.mean([score for _, score, _ in pos_images_98]):.4f}")
print(f"Positive 95 : {np.mean([score for _, score, _ in pos_images_95]):.4f}")
print(f"Negative : {np.mean([score for _, score, _ in neg_images]):.4f}")
