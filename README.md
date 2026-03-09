# Digital Forensic Classifier

This repository contains a rule-based image comparison system developed for **EAS 510: Basics of AI – Assignment 1: Digital Forensics Apprentice**. The goal of the project is to identify which original image a modified image was derived from using a transparent expert-system approach rather than machine learning.

The system evaluates evidence from multiple rules and assigns a confidence score indicating how strongly an input image matches one of the known originals. Each rule contributes evidence that is printed in the output so the reasoning behind every decision is visible.

---

## Repository Structure

```
Digital-Forensic-Classifier/
├── originals/              # 10 original reference images
├── modified_images/        # easier transformations
├── hard/                   # harder transformations
├── random/                 # unrelated images that should be rejected
├── forensics_detective.py  # main expert system class
├── rules.py                # rule implementations for Version 1
├── rules_v2.py             # extended rule set for Version 2
├── test_system.py          # script used to run the system
├── results_v1.txt          # output from Version 1 (easy + random)
├── results_v1_hard.txt     # output from Version 1 on hard cases
├── results_v2.txt          # output from Version 2 evaluation
├── requirements.txt
└── README.md
```

---

## Project Overview

The classifier works by registering a set of **original images** as targets. Each image is analyzed to extract simple visual signatures such as metadata information, color distributions, and grayscale structures.

When a modified image is tested, the system compares it against every original using the defined rules. Each rule produces a score and explanation. The scores are combined into a final confidence value that determines whether the system considers the image a match or rejects it.

This approach follows the style of **early AI expert systems**, where reasoning is explicit and interpretable rather than learned through statistical models.

---

## Version 1 Rules

Version 1 implements the three required expert system rules from the assignment.

### Rule 1 — Metadata Analysis

This rule compares properties such as:

* file size
* image width and height
* aspect ratio

These properties often remain partially related even after transformations such as cropping, compression, or format conversion. The rule calculates ratios between the input image and each original and converts them into a confidence score.

---

### Rule 2 — Color Histogram Comparison

This rule compares the color distribution of images.

A histogram is computed for each color channel using OpenCV and normalized. The histograms of the input image and target image are compared using correlation. Similar color distributions produce higher scores.

Histogram comparison is useful because it remains informative even when images are spatially modified.

---

### Rule 3 — Template Matching

This rule estimates whether one image may visually appear inside another.

The system uses OpenCV's `matchTemplate` method to compare grayscale images and compute a similarity score. To improve speed, images are resized before matching.

This rule is especially useful for identifying cropped images derived from the originals.

---

## Version 2 Improvement

After evaluating Version 1 on the **hard dataset**, an additional rule is introduced.

### Rule 4 — ORB Keypoint Matching

This rule uses ORB feature detection to compare distinctive local features between images.

Keypoints are extracted from both images and matched using a Hamming distance matcher. The number of good matches relative to detected features becomes the similarity score.

This rule improves robustness against transformations such as:

* resizing
* compression
* rotation
* combined transformations

---

## Installation

Install dependencies using pip:

```
pip install -r requirements.txt
```

The project primarily relies on:

* OpenCV
* NumPy
* Pillow

---

## Running the System

From the project directory run:

```
python test_system.py
```

The script will:

1. load the original images
2. run Version 1 on easy cases and random images
3. run Version 1 on hard cases
4. run Version 2 on all datasets
5. save the results into text files

---

## Output Files

Three output files are generated:

```
results_v1.txt
results_v1_hard.txt
results_v2.txt
```

Each file contains the full reasoning output for every image, including which rules fired and how many points they contributed.

Example output format:

```
Processing: modified_image_01.jpg
Rule 1 (Metadata): FIRED - Size ratio 0.85 -> 20/30 points
Rule 2 (Histogram): FIRED - Correlation 0.92 -> 25/30 points
Rule 3 (Template): FIRED - Match score 0.76 -> 30/40 points
Final Score: 75/100 -> MATCH to original_03.jpg
```

---

## Evaluation

The system is evaluated in two ways:

**Accuracy on modified images**

Measures how often the system correctly identifies the original source.

**False positive rate on random images**

Measures how often unrelated images are incorrectly classified as matches.

The random dataset should ideally produce a **0% false positive rate**.

---

## Notes

The purpose of the project is not perfect classification accuracy. The objective is to demonstrate how rule-based reasoning systems operate and how they can be iteratively improved by analyzing failures and refining rules.

---

## License

Apache-2.0 License
