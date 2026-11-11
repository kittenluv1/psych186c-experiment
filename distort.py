import io
import os

import numpy as np
from PIL import Image, ImageFilter

SRC = "originals"  # folder of extracted originals
OUT = "distorted"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Each entry is a severity level that applies the distortions in sequence.
# The final output for each original image is a single image with all distortions applied.
DISTORTION_LEVELS = [
    {"awgn_sigma": 5, "blur_sigma": 0.5, "jpeg_quality": 50},
    {"awgn_sigma": 10, "blur_sigma": 1.0, "jpeg_quality": 30},
    {"awgn_sigma": 20, "blur_sigma": 2.0, "jpeg_quality": 15},
    {"awgn_sigma": 40, "blur_sigma": 4.0, "jpeg_quality": 5},
]


def awgn(img, sigma, rng):
    arr = np.asarray(img, dtype=np.float32)
    noisy = arr + rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def apply_distortions(img, awgn_sigma, blur_sigma, jpeg_quality, rng):
    distorted = awgn(img, awgn_sigma, rng)
    if blur_sigma > 0:
        distorted = distorted.filter(ImageFilter.GaussianBlur(blur_sigma))

    # Apply JPEG compression as the final lossy step, then reopen to keep a single output image.
    buffer = io.BytesIO()
    distorted.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main():
    os.makedirs(OUT, exist_ok=True)

    for item in os.listdir(OUT):
        path = os.path.join(OUT, item)
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(path)
        else:
            os.remove(path)

    for root, _, files in os.walk(SRC):
        for filename in sorted(files):
            if not filename.lower().endswith(EXTS):
                continue

            src_path = os.path.join(root, filename)
            image_name = os.path.splitext(filename)[0]
            img = Image.open(src_path).convert("RGB")

            for idx, level in enumerate(DISTORTION_LEVELS, start=1):
                rng = np.random.default_rng((idx + 1) * 9973 + sum(ord(ch) for ch in image_name))
                distorted = apply_distortions(
                    img,
                    level["awgn_sigma"],
                    level["blur_sigma"],
                    level["jpeg_quality"],
                    rng,
                )
                output_path = os.path.join(OUT, f"{image_name}_distorted_{idx}.png")
                distorted.save(output_path)


if __name__ == "__main__":
    main()