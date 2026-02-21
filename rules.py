import os
import cv2
import numpy as np

print("loading rules")


_last_path = None
_last_gray = None

def _load_gray(path):
    """
    Load image once and reuse it if the same path
    is requested repeatedly.
    """
    global _last_path, _last_gray

    if path == _last_path and _last_gray is not None:
        return _last_gray

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    _last_path = path
    _last_gray = img

    return img


def _dhash(gray, size=8):
    if gray is None:
        return 0

    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, :-1] > small[:, 1:]

    h = 0
    bit = 0
    for r in range(size):
        for c in range(size):
            if diff[r, c]:
                h |= (1 << bit)
            bit += 1
    return h


def _hamming(a, b):
    return (a ^ b).bit_count()


def _center_crop(gray, keep=0.75):
    if gray is None:
        return None
    if keep >= 0.999:
        return gray

    h, w = gray.shape[:2]
    new_w = max(1, int(w * keep))
    new_h = max(1, int(h * keep))

    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    return gray[y0:y0+new_h, x0:x0+new_w]


def _tiny(gray, w=32, h=32):
    if gray is None:
        return None
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)


def _mean_abs_diff(a, b):
    if a is None or b is None:
        return 255.0
    diff = cv2.absdiff(a, b)
    return float(np.mean(diff))


# Functions of rules

def rule1_metadata(target_info, input_path):
    input_size = os.path.getsize(input_path)
    target_size = target_info['size']

    if target_size > 0 and input_size > 0:
        ratio = min(input_size, target_size) / max(input_size, target_size)
    else:
        ratio = 0

    score = int(ratio * 10)
    fired = ratio > 0.5
    evidence = f"Size ratio {ratio:.2f}"
    return score, fired, evidence


def rule2_dhash_whole(target_info, input_path):
    gray = _load_gray(input_path)
    h_in = _dhash(gray)

    h_t = target_info['dhash_whole']
    dist = _hamming(h_in, h_t)

    sim = 1.0 - (dist / 64.0)
    if sim < 0:
        sim = 0

    score = int(sim * 15)
    fired = sim > 0.70
    evidence = f"dHash sim {sim:.2f} (dist {dist})"
    return score, fired, evidence


def rule3_dhash_center_crop(target_info, input_path):
    gray = _load_gray(input_path)
    h_in = _dhash(gray)

    best_sim = 0.0
    best_keep = None

    for k in [0.75, 0.50, 0.25]:
        if k == 0.75:
            h_t = target_info['dhash_crop75']
        elif k == 0.50:
            h_t = target_info['dhash_crop50']
        else:
            h_t = target_info['dhash_crop25']

        dist = _hamming(h_in, h_t)
        sim = 1.0 - (dist / 64.0)
        if sim < 0:
            sim = 0

        if sim > best_sim:
            best_sim = sim
            best_keep = k

    score = int(best_sim * 5)
    fired = best_sim > 0.60
    evidence = f"input vs target keep {best_keep} sim {best_sim:.2f}"
    return score, fired, evidence


def rule4_tiny_compare(target_info, input_path):
    gray_in = _load_gray(input_path)
    inp_t = _tiny(gray_in)

    best_score = 0
    best_keep = None
    best_mad = None

    for k in [0.75, 0.5, 0.25]:
        tgt_t = target_info["tiny_keep"][k]

        mad = _mean_abs_diff(inp_t, tgt_t)
        sim = 1.0 - (mad / 255.0)
        if sim < 0:
            sim = 0

        score = int(sim * 5)

        if score > best_score:
            best_score = score
            best_keep = k
            best_mad = mad

    fired = best_score >= 3
    evidence = f"input vs target keep {best_keep} mad {best_mad:.1f}"
    return best_score, fired, evidence