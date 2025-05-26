import os
import cv2
import re
import easyocr
from typing import List, Tuple

reader = easyocr.Reader(['en', 'hi', 'mr'], gpu=True)

# Regex patterns
AADHAAR_FULL_REGEX = re.compile(r'(?<!\d)(\d{4}[\s-]?\d{4}[\s-]?\d{4})(?!\d)')
DOB_REGEX = re.compile(r'\b(?:19|20)\d{2}\b')  # Year-only DOB

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def auto_orient_image(image_path: str):
    image = cv2.imread(image_path)

    # Step 1: Try original image with all 4 rotations
    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(image, angle)
        results = reader.readtext(rotated)
        for (bbox, text, prob) in results:
            if prob < 0.3:
                continue
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            if re.search(AADHAAR_FULL_REGEX, cleaned_text):
                print(f"✅ Found Aadhaar number at {angle}° rotation.")
                return rotated

    print("⚠️ No Aadhaar number detected confidently in normal orientation. Trying mirrored image...")

    # Step 2: Try mirrored image with all 4 rotations
    mirrored_image = cv2.flip(image, 1)
    for angle in [0, 90, 180, 270]:
        rotated = rotate_image(mirrored_image, angle)
        results = reader.readtext(rotated)
        for (bbox, text, prob) in results:
            if prob < 0.3:
                continue
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            if re.search(AADHAAR_FULL_REGEX, cleaned_text):
                print(f"🔁 Found Aadhaar number in MIRRORED image at {angle}° rotation. Flipping back.")
                return cv2.flip(rotated, 1)  # Flip back to correct orientation

    print("⚠️ Aadhaar number not confidently detected even after mirroring. Using original image.")
    return image

def extract_text_regions(image) -> List[Tuple[str, Tuple[int, int], Tuple[int, int], float]]:
    results = []
    ocr_results = reader.readtext(image)
    print("\n🔍 OCR Results:")
    for (bbox, text, prob) in ocr_results:
        print(f"Detected: '{text}' with confidence {prob:.2f}")
        if prob < 0.20:
            continue
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        results.append((text, top_left, bottom_right, prob))
    return results

def mask_text_on_image(image_path: str, output_path: str):
    image = auto_orient_image(image_path)
    text_regions = extract_text_regions(image)

    aadhaar_chunks = []
    block_groups = []

    # Grouping text boxes into blocks based on y-coordinate proximity (same line ~ same block)
    sorted_regions = sorted(text_regions, key=lambda x: x[1][1])  # sort by top-left y
    current_block = []

    for region in sorted_regions:
        _, top_left, _, _ = region
        if not current_block:
            current_block.append(region)
            continue

        prev_y = current_block[-1][1][1]
        curr_y = top_left[1]

        if abs(curr_y - prev_y) < 30:  # heuristic for same block
            current_block.append(region)
        else:
            block_groups.append(current_block)
            current_block = [region]

    if current_block:
        block_groups.append(current_block)

    address_keywords = [
        # English
        "address", "addr", "s/o", "d/o", "w/o", "c/o", "father", "mother", "guardian",
        "house", "building", "flat", "apartment", "floor", "block", "sector", "lane",
        "road", "street", "near", "opposite", "behind", "beside", "village", "town",
        "city", "district", "taluka", "tehsil", "state", "pin", "pincode", "post",
        "po", "p.o", "area", "locality", "colony", "ward", "location", "residence",

        # Hindi (Devanagari)
        "पता", "ग्राम", "पोस्ट", "पिन", "पिनकोड", "जिला", "राज्य", "तालुका", "तहसील",
        "सड़क", "गली", "मार्ग", "नगर", "शहर", "मकान", "फ्लैट", "मंजिल", "ब्लॉक",
        "सेक्टर", "निकट", "सामने", "पीछे", "बगल", "नजदीक", "पड़ोस", "स्थान", "इलाका",
        "कालोनी", "वार्ड", "आवास", "स्थाई", "अस्थाई", "रहائش", "पिता", "माता", "अभिभावक",
        "स/ओ", "डी/ओ", "डब्ल्यू/ओ", "सी/ओ",

        # Marathi (Devanagari)
        "पत्ता", "गाव", "पोस्ट", "पिन", "पिनकोड", "जिल्हा", "राज्य", "तालुका", "तहसील",
        "रस्ता", "गली", "मार्ग", "नगर", "शहर", "घर", "इमारत", "मजला", "ब्लॉक",
        "सेक्टर", "जवळ", "समोर", "मागे", "शेजारी", "पडोसी", "ठिकाण", "परिसर", "वसाहत",
        "वार्ड", "निवास", "स्थायी", "अस्थायी", "राहण्याचे", "वडील", "आई", "पालक",
        "स/ओ", "ड/ओ", "डब्ल्यू/ओ", "सी/ओ"
        ]



    # Pass 1: Aadhaar detection
    for text, top_left, bottom_right, _ in text_regions:
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        match = re.search(AADHAAR_FULL_REGEX, cleaned_text)
        if match:
            print(f"🔒 Found full Aadhaar Number: {match.group(1)} (raw: '{text}')")
            x1, y1 = top_left
            x2, y2 = bottom_right
            total_width = x2 - x1
            masked_width = int((8 / 12) * total_width)
            mask_end_x = x1 + masked_width
            cv2.rectangle(image, (x1, y1), (mask_end_x, y2), (0, 0, 0), -1)
            print(f"🔒 Masked first 8 digits: from x={x1} to x={mask_end_x}")
        elif re.fullmatch(r'\d{4}', cleaned_text):
            aadhaar_chunks.append((cleaned_text, top_left, bottom_right))

    # Pass 2: Aadhaar chunk masking (rolling 3 window)
    masked_indices = set()
    for i in range(len(aadhaar_chunks) - 2):
        chunk1, chunk2, chunk3 = aadhaar_chunks[i:i+3]
        y1 = chunk1[1][1]
        y2 = chunk2[1][1]
        y3 = chunk3[1][1]
        same_line = abs(y1 - y2) < 20 and abs(y2 - y3) < 20
        x_gap = chunk3[2][0] - chunk1[1][0] < 300

        if same_line and x_gap and i not in masked_indices:
            x1, y1 = chunk1[1]
            x2, y2 = chunk2[2]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
            print(f"🔒 Masked Aadhaar chunks: {chunk1[0]} {chunk2[0]} (first 8 digits)")
            masked_indices.update([i, i+1, i+2])

    # Pass 3: Mask full blocks if any text has address-like words
    for block in block_groups:
        block_has_address = False
        for text, _, _, _ in block:
            cleaned_lower = text.strip().lower()
            if any(keyword in cleaned_lower for keyword in address_keywords):
                block_has_address = True
                break

        if block_has_address:
            for _, top_left, bottom_right, _ in block:
                cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
            print(f"🔒 Masked address block with keywords: {[text for text, *_ in block]}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    print(f"\n✅ Masked image saved at: {output_path}")

# Test block
if __name__ == "__main__":
    input_path = "dataset/test_aadhaar/21.jpg"
    output_path = f"aadhaar_masked/masked_sample.jpg" 
    mask_text_on_image(input_path, output_path)
