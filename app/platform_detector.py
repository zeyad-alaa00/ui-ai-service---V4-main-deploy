# platform_detector.py

import logging
import re
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


ASPECT_RATIO_MOBILE_THRESHOLD = 1.3
ASPECT_RATIO_WEB_THRESHOLD = 0.8

RESOLUTION_MOBILE_MAX_WIDTH = 900
MIN_VALID_DIMENSION = 50

FALLBACK_PLATFORM = "web"

# stronger keyword sets
MOBILE_KEYWORDS = [
    "login",
    "sign in",
    "home",
    "profile",
    "continue",
    "get started",
]

WEB_KEYWORDS = [
    "dashboard",
    "admin",
    "analytics",
    "sidebar",
    "table",
    "users",
]


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _keyword_score(text: str, keywords: list[str]) -> int:
    score = 0
    for kw in keywords:
        if kw in text:
            score += 1
    return score




def _aspect_ratio_score(image: Image.Image) -> Optional[str]:
    width, height = image.size

    if width < MIN_VALID_DIMENSION or height < MIN_VALID_DIMENSION:
        logger.warning("Image too small for reliable detection: %s", image.size)
        return None

    ratio = height / width

    if ratio > ASPECT_RATIO_MOBILE_THRESHOLD:
        return "mobile"
    if ratio < ASPECT_RATIO_WEB_THRESHOLD:
        return "web"

    return None


def _resolution_score(image: Image.Image) -> str:
    width, _ = image.size
    return "mobile" if width < RESOLUTION_MOBILE_MAX_WIDTH else "web"


def _ocr_score(ocr_text: str) -> Optional[str]:
    normalized = _normalize_text(ocr_text)

    mobile_hits = _keyword_score(normalized, MOBILE_KEYWORDS)
    web_hits = _keyword_score(normalized, WEB_KEYWORDS)

    logger.debug(
        "OCR scoring | mobile_hits=%d | web_hits=%d",
        mobile_hits,
        web_hits,
    )

    if mobile_hits > web_hits:
        return "mobile"
    if web_hits > mobile_hits:
        return "web"

    return None




def detect_platform(image: Image.Image, ocr_text: Optional[str] = None) -> str:
    votes = {"mobile": 0, "web": 0}


    ar = _aspect_ratio_score(image)
    if ar:
        votes[ar] += 3   # stronger weight


    if ocr_text and ocr_text.strip():
        try:
            ocr = _ocr_score(ocr_text)
            if ocr:
                votes[ocr] += 2
        except Exception as e:
            logger.warning("OCR scoring failed: %s", e)


    try:
        res = _resolution_score(image)
        votes[res] += 1
    except Exception as e:
        logger.warning("Resolution scoring failed: %s", e)

    mobile_score = votes["mobile"]
    web_score = votes["web"]

    logger.info(
        "Platform detection | mobile=%d | web=%d | size=%s",
        mobile_score,
        web_score,
        image.size,
    )


    if mobile_score > web_score:
        return "mobile"

    if web_score > mobile_score:
        return "web"

    logger.warning("Tie in platform detection → fallback=%s", FALLBACK_PLATFORM)
    return FALLBACK_PLATFORM