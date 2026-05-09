# app/router.py

import json
import logging
import re
from typing import Any, List

from fastapi import APIRouter, HTTPException

from app.model import generate_from_image
from app.schemas import (
    AnalyzeUIRequest,
    AnalyzeUIResponse,
    UIElement,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai",
    tags=["AI"],
)

VALID_TYPES = {
    "text": "label",
    "label": "label",
    "button": "button",
    "icon": "icon",
    "input": "input",
    "image": "image",
    "card": "card",
    "navigation": "navigation",
    "tab": "tab",
    "link": "button",
    "container": "card",
    "section": "card",
}


@router.get("/")
def test():

    return {
        "status": "ok",
    }


def _wrap_list_response(
    parsed: list,
) -> dict:

    return {
        "screen_id": None,
        "elements": parsed,
    }


def _safe_json_loads(
    raw: str,
):

    parsed = json.loads(raw)

    if isinstance(parsed, list):

        return _wrap_list_response(
            parsed
        )

    return parsed


def _extract_json(
    raw: str,
) -> dict:

    raw = raw.strip()

    if raw == "[]":

        return {
            "screen_id": None,
            "elements": [],
        }

    raw = raw.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()

    if raw.startswith("["):

        end = raw.rfind("]")

        if end == -1:

            return {
                "screen_id": None,
                "elements": [],
            }

        raw = raw[:end + 1]

    else:

        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1:

            return {
                "screen_id": None,
                "elements": [],
            }

        raw = raw[start:end + 1]

    raw = raw.replace(",}", "}")
    raw = raw.replace(",]", "]")

    try:

        return _safe_json_loads(raw)

    except Exception:
        pass

    raw = re.sub(
        r'("left"\s*:\s*[0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)',
        r'\1, "top": \2',
        raw,
    )

    raw = re.sub(
        r'("x_norm"\s*:\s*[0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)',
        r'\1, "y_norm": \2',
        raw,
    )

    raw = re.sub(
        r'"bbox_norm"\s*:\s*\[\s*([0-9eE\.\-]+)\s*,\s*([0-9eE\.\-]+)\s*\]',
        (
            r'"bbox_norm": {'
            r'"x_norm": \1, '
            r'"y_norm": \2, '
            r'"w_norm": 0.1, '
            r'"h_norm": 0.05}'
        ),
        raw,
    )

    raw = re.sub(
        (
            r'"bbox_norm"\s*:\s*\['
            r'\s*([0-9eE\.\-]+)\s*,'
            r'\s*([0-9eE\.\-]+)\s*,'
            r'\s*([0-9eE\.\-]+)\s*,'
            r'\s*([0-9eE\.\-]+)\s*'
            r'\]'
        ),
        (
            r'"bbox_norm": {'
            r'"x_norm": \1, '
            r'"y_norm": \2, '
            r'"w_norm": \3, '
            r'"h_norm": \4}'
        ),
        raw,
    )

    raw = re.sub(
        r'}\s*{',
        '}, {',
        raw,
    )

    raw = raw.replace(
        "\n",
        " "
    )

    raw = raw.replace(
        "\t",
        " "
    )

    raw = re.sub(
        r"\s+",
        " ",
        raw,
    )

    raw = raw.replace(",}", "}")
    raw = raw.replace(",]", "]")

    try:

        return _safe_json_loads(raw)

    except Exception as e:

        logger.error(
            "FINAL JSON RECOVERY FAILED | raw=%s",
            raw[:3000],
        )

        raise ValueError(
            f"JSON recovery failed: {e}"
        ) from e


def _normalize_bbox(
    bbox: Any,
) -> dict:

    if isinstance(bbox, dict):

        if "x" in bbox:

            return {
                "x_norm": float(
                    bbox.get("x", 0)
                ),
                "y_norm": float(
                    bbox.get("y", 0)
                ),
                "w_norm": float(
                    bbox.get("width", 0.1)
                ),
                "h_norm": float(
                    bbox.get("height", 0.05)
                ),
            }

        if (
            "left" in bbox
            and "top" in bbox
        ):

            return {
                "x_norm": float(
                    bbox.get("left", 0)
                ),
                "y_norm": float(
                    bbox.get("top", 0)
                ),
                "w_norm": float(
                    bbox.get("width", 0.1)
                ),
                "h_norm": float(
                    bbox.get("height", 0.05)
                ),
            }

        return {
            "x_norm": float(
                bbox.get("x_norm", 0)
            ),
            "y_norm": float(
                bbox.get("y_norm", 0)
            ),
            "w_norm": float(
                bbox.get("w_norm", 0.1)
            ),
            "h_norm": float(
                bbox.get("h_norm", 0.05)
            ),
        }

    if (
        isinstance(bbox, list)
        and len(bbox) >= 2
    ):

        return {
            "x_norm": float(bbox[0]),
            "y_norm": float(bbox[1]),
            "w_norm": float(
                bbox[2]
                if len(bbox) > 2
                else 0.1
            ),
            "h_norm": float(
                bbox[3]
                if len(bbox) > 3
                else 0.05
            ),
        }

    return {
        "x_norm": 0.0,
        "y_norm": 0.0,
        "w_norm": 0.1,
        "h_norm": 0.05,
    }


def _clean_text(
    value: Any,
):

    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if value == "":
        return None

    return value


def _normalize_type(
    raw_type: Any,
) -> str:

    if raw_type is None:
        return "label"

    raw_type = str(
        raw_type
    ).lower().strip()

    return VALID_TYPES.get(
        raw_type,
        "label",
    )


def _build_elements(
    data: List[dict[str, Any]],
) -> List[UIElement]:

    elements = []

    for i, item in enumerate(data):

        try:

            if not isinstance(item, dict):
                continue

            bbox = _normalize_bbox(
                item.get("bbox_norm")
            )

            normalized_type = (
                _normalize_type(
                    item.get("type")
                )
            )

            text_value = _clean_text(
                item.get("text")
            )

            element = UIElement(

                id=str(
                    item.get(
                        "id",
                        i + 1,
                    )
                ),

                type=normalized_type,

                bbox_norm=bbox,

                is_clickable=bool(
                    item.get(
                        "is_clickable",
                        normalized_type in [
                            "button",
                            "tab",
                            "navigation",
                            "input",
                        ],
                    )
                ),

                text=text_value,

                font_weight=(
                    item.get(
                        "font_weight"
                    )
                    or "regular"
                ),

                font_size_role=(
                    item.get(
                        "font_size_role"
                    )
                    or "body"
                ),
            )

            elements.append(element)

        except Exception as e:

            logger.warning(
                (
                    "Skipping invalid element "
                    "| index=%d "
                    "| error=%s "
                    "| data=%s"
                ),
                i,
                e,
                str(item)[:500],
            )

    return elements


@router.post(
    "/analyze-ui",
    response_model=AnalyzeUIResponse,
    summary=(
        "Analyze UI screenshot "
        "→ structured elements"
    ),
)
async def analyze_ui(
    request: AnalyzeUIRequest,
) -> AnalyzeUIResponse:

    if not request.imageUrl:

        raise HTTPException(
            status_code=400,
            detail="imageUrl required",
        )

    image_url = str(
        request.imageUrl
    )

    logger.info(
        "Analyze request | image=%s",
        image_url[:200],
    )

    try:

        raw_output = generate_from_image(
            image_url=image_url,
        )

    except RuntimeError as e:

        logger.exception(
            "Model runtime failure"
        )

        raise HTTPException(
            status_code=503,
            detail=str(e),
        ) from e

    except Exception as e:

        logger.exception(
            "Inference failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Inference failed",
        ) from e

    try:

        parsed = _extract_json(
            raw_output
        )

        elements_data = parsed.get(
            "elements",
            [],
        )

        screen_id = parsed.get(
            "screen_id"
        )

        if not isinstance(
            elements_data,
            list,
        ):

            elements_data = []

        elements = _build_elements(
            elements_data
        )

        logger.info(
            "Parsed %d elements",
            len(elements),
        )

    except Exception as e:

        logger.error(
            (
                "JSON parsing failed "
                "| error=%s "
                "| raw=%s"
            ),
            e,
            raw_output[:3000],
        )

        return AnalyzeUIResponse(
            screen_id=None,
            elements=[],
        )

    return AnalyzeUIResponse(
        screen_id=screen_id,
        elements=elements,
    )