# Digital Forensics Apprentice (Rule-Based Image Matcher)

This repository contains my “Digital Forensics Detective” prototype. It is a rule-based expert system that attempts to match modified images back to their original source image using simple handcrafted rules. The system does not use machine learning; instead, it combines multiple image comparison rules and scoring logic to make a decision.

---

## Project Structure

```
Digital-Forensic-Classifier/
├── forensics_detective.py   # main script (runs the detective)
├── rules.py                 # expert system rules + helper functions
├── README.md
├── originals/               # 10 original reference images
├── modified_images/         # easy test set (single transformations)
├── hard/                    # hard test set (combined transformations)
└── random/                  # unrelated images + noise (should be rejected)
```

---

## What This System Does

When the script runs:

1. It loads all images in `originals/` and registers them as targets.
2. It computes simple image signatures for each original:
   - file size
   - whole-image perceptual hash (dHash)
   - center-crop hashes (75%, 50%, 25%)
   - tiny 32×32 grayscale thumbnails for crop comparison
3. The user chooses which test folder to run:
   - `modified_images` (easy)
   - `hard`
   - `random`
4. Each test image is compared against every original.
5. Scores from multiple rules are combined to determine the best match or rejection.

The output prints transparent reasoning so you can see how each rule contributed to the final decision.

---

## Expert System Rules

### Rule 1 — Metadata Comparison
Compares file size between the input image and each target image.

### Rule 2 — Whole Image dHash
Uses perceptual hashing to compare overall image structure, even if compression or brightness changes occur.

### Rule 3 — Center Crop Matching
Compares hashes from cropped regions to detect strong center crops (25%, 50%, 75%).

### Rule 4 — Tiny Thumbnail Comparison
Creates small grayscale thumbnails and compares pixel differences to estimate similarity.

---

## Dataset Description

### `originals/`
10 original JPEG images used as the reference database.

### `modified_images/` (Easy Cases)
Each original has simple single transformations such as:
- Brightness enhancement
- JPEG compression
- 25% center crop
- 50% center crop
- 75% center crop
- PNG conversion

### `hard/` (Hard Cases)
Each original has combined transformations, for example:
- Off-center crop + compression
- Crop + brightness + compression
- Resize + compression
- Rotation + compression
- Contrast + compression
- Crop + resize + compression

### `random/`
Images that are not related to any original.  
The system should normally reject these.

---

## Setup

Install dependencies:

```bash
pip install pillow opencv-python numpy
```

---

## Running the Project

From the project folder:

```bash
python forensics_detective.py
```

The script will ask:

```
Choose test folder:
1 - modified_images (easy)
2 - hard
3 - random
Enter choice (1/2/3):
```

After selection, the program processes all images in that folder.

---

## Example Output

The system prints reasoning like:

```
Processing: modified_00_crop_50pct.jpg
  Rule 1: Size ratio ...
  Rule 2: dHash similarity ...
  Rule 3: Crop similarity ...
  Rule 4: Tiny thumbnail difference ...
  Total score: XX/30
Final: MATCH / REJECTED
```

---

## Ground Truth

File names indicate the original source image:

- `modified_03_*` → `original_03.jpg`
- `original_03__rotate...__v4.jpg` → `original_03.jpg`

Images in `random/` are unrelated and should be rejected.

---


