from typing import Dict, Any

import cv2

from rules import (
    build_target_signature as v1_build_target_signature,
    build_input_signature as v1_build_input_signature,
    rule1_metadata,
    rule2_histogram,
    rule3_template,
)


MATCH_THRESHOLD = 50


def _compute_orb_similarity(gray_a, gray_b) -> float:
    if gray_a is None or gray_b is None:
        return 0.0

    orb = cv2.ORB_create(nfeatures=500)

    kp1, des1 = orb.detectAndCompute(gray_a, None)
    kp2, des2 = orb.detectAndCompute(gray_b, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if not matches:
        return 0.0

    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = [m for m in matches if m.distance <= 50]

    denom = max(min(len(kp1), len(kp2)), 1)
    similarity = len(good_matches) / float(denom)

    return max(0.0, min(1.0, similarity))


def build_target_signature(path: str) -> Dict[str, Any]:
    return v1_build_target_signature(path)


def build_input_signature(path: str) -> Dict[str, Any]:
    return v1_build_input_signature(path)


def rule4_extra(target_info: Dict[str, Any], input_info: Dict[str, Any]) -> Dict[str, Any]:
    orb_similarity = _compute_orb_similarity(target_info["gray"], input_info["gray"])

    score = int(round(orb_similarity * 20))
    score = max(0, min(20, score))

    fired = score >= 8
    status = "FIRED" if fired else "NO MATCH"

    line = f"Rule 4 (ORB Keypoints): {status} - Match score {orb_similarity:.2f} -> {score}/20 points"

    return {
        "name": "Rule 4 (ORB Keypoints)",
        "score": score,
        "fired": fired,
        "line": line,
        "orb_similarity": orb_similarity,
    }