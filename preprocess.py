import os
import cv2
from pathlib import Path

def rotate_image(image, angle):
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Only 0, 90, 180, and 270 degrees are supported.")

def mirror_image(image, mode="horizontal"):
    if mode == "horizontal":
        return cv2.flip(image, 1)
    elif mode == "vertical":
        return cv2.flip(image, 0)
    else:
        raise ValueError("Mode should be 'horizontal' or 'vertical'")

def augment_images_with_rotations_and_mirrors(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = list(input_folder.glob("*.*"))
    if len(image_files) != 6353:
        print(f"Warning: Expected 6353 images, found {len(image_files)}.")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        base_name = img_path.stem
        ext = img_path.suffix

        for angle in [0, 90, 180, 270]:
            rotated = rotate_image(img, angle)
            # Save rotated
            filename = f"{base_name}_rotated_{angle}{ext}"
            cv2.imwrite(str(output_folder / filename), rotated)

            # Horizontal mirror
            horiz = mirror_image(rotated, "horizontal")
            filename_h = f"{base_name}_rotated_{angle}_horiz{ext}"
            cv2.imwrite(str(output_folder / filename_h), horiz)

            # Vertical mirror
            vert = mirror_image(rotated, "vertical")
            filename_v = f"{base_name}_rotated_{angle}_vert{ext}"
            cv2.imwrite(str(output_folder / filename_v), vert)

    print(f"Saved rotated + horizontal + vertical mirrored versions for each image in {output_folder}")

def rename_images(folder_path, prefix="image_", start=1):
    folder = Path(folder_path)
    image_files = sorted([f for f in folder.iterdir() if f.is_file()])

    for i, img_file in enumerate(image_files, start=start):
        ext = img_file.suffix.lower()
        new_name = f"{prefix}{i:05d}{ext}"
        img_file.rename(folder / new_name)

    print(f"Renamed {len(image_files)} images in '{folder_path}' with prefix '{prefix}'.")

# === Run the full pipeline ===
input_dir = "dataset/Aadhaar"
output_dir = "dataset/Aadhaar_rotated"

augment_images_with_rotations_and_mirrors(input_dir, output_dir)
rename_images(output_dir, prefix="Aadhaar_", start=1)