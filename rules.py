import os
from typing import Dict, Any

import cv2
import numpy as np
from PIL import Image


MATCH_THRESHOLD = 50


def _safe_read_color(path: str):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def _safe_read_gray(path: str):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def _pil_size(path: str):
    with Image.open(path) as img:
        return img.size  # (width, height)


def _aspect_ratio(width: int, height: int) -> float:
    if height == 0:
        return 0.0
    return width / float(height)


def _compute_histogram_bgr(image_bgr):
    hist_b = cv2.calcHist([image_bgr], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image_bgr], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([image_bgr], [2], None, [32], [0, 256])

    hist = np.vstack([hist_b, hist_g, hist_r]).astype("float32")
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def _template_match_score(img_a_gray, img_b_gray) -> float:
    if img_a_gray is None or img_b_gray is None:
        return 0.0

    h1, w1 = img_a_gray.shape[:2]
    h2, w2 = img_b_gray.shape[:2]

    if h1 == 0 or w1 == 0 or h2 == 0 or w2 == 0:
        return 0.0

    max_dim = 256

    def resize_if_needed(img):
        h, w = img.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    img_a_gray = resize_if_needed(img_a_gray)
    img_b_gray = resize_if_needed(img_b_gray)

    h1, w1 = img_a_gray.shape[:2]
    h2, w2 = img_b_gray.shape[:2]

    area1 = h1 * w1
    area2 = h2 * w2

    if area1 <= area2:
        template = img_a_gray
        search = img_b_gray
    else:
        template = img_b_gray
        search = img_a_gray

    th, tw = template.shape[:2]
    sh, sw = search.shape[:2]

    if th > sh or tw > sw:
        return 0.0

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def build_target_signature(path: str) -> Dict[str, Any]:
    image_bgr = _safe_read_color(path)
    image_gray = _safe_read_gray(path)
    width, height = _pil_size(path)
    file_size = os.path.getsize(path)

    return {
        "path": path,
        "file_size": file_size,
        "width": width,
        "height": height,
        "aspect_ratio": _aspect_ratio(width, height),
        "histogram": _compute_histogram_bgr(image_bgr) if image_bgr is not None else None,
        "gray": image_gray,
    }


def build_input_signature(path: str) -> Dict[str, Any]:
    image_bgr = _safe_read_color(path)
    image_gray = _safe_read_gray(path)
    width, height = _pil_size(path)
    file_size = os.path.getsize(path)

    return {
        "path": path,
        "file_size": file_size,
        "width": width,
        "height": height,
        "aspect_ratio": _aspect_ratio(width, height),
        "histogram": _compute_histogram_bgr(image_bgr) if image_bgr is not None else None,
        "gray": image_gray,
    }


def rule1_metadata(target_info: Dict[str, Any], input_info: Dict[str, Any]) -> Dict[str, Any]:
    t_size = target_info["file_size"]
    i_size = input_info["file_size"]

    size_ratio = min(t_size, i_size) / max(t_size, i_size) if max(t_size, i_size) > 0 else 0.0

    t_w, t_h = target_info["width"], target_info["height"]
    i_w, i_h = input_info["width"], input_info["height"]

    width_ratio = min(t_w, i_w) / max(t_w, i_w) if max(t_w, i_w) > 0 else 0.0
    height_ratio = min(t_h, i_h) / max(t_h, i_h) if max(t_h, i_h) > 0 else 0.0

    t_ar = target_info["aspect_ratio"]
    i_ar = input_info["aspect_ratio"]
    aspect_ratio_similarity = min(t_ar, i_ar) / max(t_ar, i_ar) if max(t_ar, i_ar) > 0 else 0.0

    combined = (0.4 * size_ratio) + (0.3 * width_ratio) + (0.2 * height_ratio) + (0.1 * aspect_ratio_similarity)
    score = int(round(combined * 30))
    score = max(0, min(30, score))

    fired = score >= 15
    status = "FIRED" if fired else "NO MATCH"

    line = (
        f"Rule 1 (Metadata): {status} - "
        f"Size ratio {size_ratio:.2f}, Dimension ratio {((width_ratio + height_ratio) / 2):.2f} "
        f"-> {score}/30 points"
    )

    return {
        "name": "Rule 1 (Metadata)",
        "score": score,
        "fired": fired,
        "line": line,
        "size_ratio": size_ratio,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "aspect_ratio_similarity": aspect_ratio_similarity,
    }


def rule2_histogram(target_info: Dict[str, Any], input_info: Dict[str, Any]) -> Dict[str, Any]:
    t_hist = target_info["histogram"]
    i_hist = input_info["histogram"]

    correlation = 0.0
    if t_hist is not None and i_hist is not None:
        correlation = float(
            cv2.compareHist(
                t_hist.astype("float32"),
                i_hist.astype("float32"),
                cv2.HISTCMP_CORREL
            )
        )

    normalized = (correlation + 1.0) / 2.0
    normalized = max(0.0, min(1.0, normalized))

    score = int(round(normalized * 30))
    score = max(0, min(30, score))

    fired = score >= 15
    status = "FIRED" if fired else "NO MATCH"

    line = f"Rule 2 (Histogram): {status} - Correlation {correlation:.2f} -> {score}/30 points"

    return {
        "name": "Rule 2 (Histogram)",
        "score": score,
        "fired": fired,
        "line": line,
        "correlation": correlation,
    }


def rule3_template(target_info: Dict[str, Any], input_info: Dict[str, Any]) -> Dict[str, Any]:
    match_score = _template_match_score(target_info["gray"], input_info["gray"])
    match_score = max(0.0, min(1.0, match_score))

    score = int(round(match_score * 40))
    score = max(0, min(40, score))

    fired = score >= 20
    status = "FIRED" if fired else "NO MATCH"

    line = f"Rule 3 (Template): {status} - Match score {match_score:.2f} -> {score}/40 points"

    return {
        "name": "Rule 3 (Template)",
        "score": score,
        "fired": fired,
        "line": line,
        "match_score": match_score,
    }