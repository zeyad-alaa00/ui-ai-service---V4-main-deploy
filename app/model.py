# app/model.py

import logging
import os
import time
from io import BytesIO
from typing import Literal

import requests
import torch

from PIL import Image

from peft import PeftModel

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

from app.platform_detector import detect_platform



logger = logging.getLogger(__name__)



BASE_MODEL_ID = os.getenv(
    "BASE_MODEL_ID",
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

logger.info(
    "BASE_MODEL_ID: %s",
    BASE_MODEL_ID,
)
ADAPTER_PATHS = {
    "mobile": "./adapters/mobile",
    "web": "./adapters/web",
}



_processor = AutoProcessor.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True,
)

MAX_IMAGE_MB = 10

MAX_IMAGE_SIDE = 1280


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

logger.info(
    "DEVICE: %s",
    DEVICE,
)

if DEVICE == "cuda":

    logger.info(
        "GPU: %s",
        torch.cuda.get_device_name(0),
    )

    logger.info(
        "VRAM: %.2f GB",
        torch.cuda.get_device_properties(
            0
        ).total_memory / (1024 ** 3),
    )

logger.info(
    "DTYPE: %s",
    DTYPE,
)


_models = {}

_model_ready = False

_UI_PROMPT = """
Extract UI elements from this UI screenshot.

Return ONLY valid JSON.

Output format:

{
  "screen_id": "string",
  "elements": [
    {
      "id": "string",
      "type": "Label | Button | Icon | Input | Image | Card | Navigation | Tab",
      "text": "string",
      "bbox_norm": {
        "x_norm": 0.0,
        "y_norm": 0.0,
        "w_norm": 0.0,
        "h_norm": 0.0
      },
      "is_clickable": false,
      "font_weight": "regular | medium | semibold | bold",
      "font_size_role": "caption | body | subtitle | title | heading"
    }
  ]
}

Rules:
- Return ONLY JSON
- Detect all visible UI elements
- Preserve exact text
- bbox_norm must be object
- Never return explanations
- Never return markdown
"""


def _load_processor():

    global _processor

    if _processor is not None:
        return _processor

    logger.info(
        "Loading processor..."
    )

    _processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )

    logger.info(
        "Processor loaded ✅"
    )

    return _processor

def _load_base_model():

    logger.info(
        "Loading base model..."
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            device_map=(
                "auto"
                if DEVICE == "cuda"
                else None
            ),
        )
    )

    model.eval()

    model.config.use_cache = True

    return model


def load_models():

    global _models
    global _model_ready

    try:

        _load_processor()

        base_model = _load_base_model()

        for adapter_name, adapter_path in ADAPTER_PATHS.items():

            try:

                logger.info(
                    "Loading adapter: %s",
                    adapter_name,
                )

                model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    is_trainable=False,
                )

                model.eval()

                _models[adapter_name] = model

                logger.info(
                    "%s adapter loaded ✅",
                    adapter_name,
                )

            except Exception:

                logger.exception(
                    "%s adapter failed",
                    adapter_name,
                )

        if not _models:

            raise RuntimeError(
                "No adapters loaded"
            )

        _model_ready = True

        logger.info(
            "All models ready ✅"
        )

    except Exception:

        _model_ready = False

        logger.exception(
            "Model loading failed"
        )

        raise


def is_model_ready():

    return _model_ready


def _load_image(
    url: str,
) -> Image.Image:

    try:

        with requests.get(
            url,
            stream=True,
            timeout=20,
        ) as response:

            response.raise_for_status()

            content_type = response.headers.get(
                "content-type",
                "",
            )

            if "image" not in content_type:

                raise RuntimeError(
                    "URL is not an image"
                )

            size = 0

            max_bytes = (
                MAX_IMAGE_MB
                * 1024
                * 1024
            )

            data = BytesIO()

            for chunk in response.iter_content(8192):

                if not chunk:
                    continue

                size += len(chunk)

                if size > max_bytes:

                    raise RuntimeError(
                        "Image too large"
                    )

                data.write(chunk)

        image = Image.open(
            data
        ).convert("RGB")

        w, h = image.size

        if max(w, h) > MAX_IMAGE_SIDE:

            if w > h:

                new_w = MAX_IMAGE_SIDE

                new_h = int(
                    h * (MAX_IMAGE_SIDE / w)
                )

            else:

                new_h = MAX_IMAGE_SIDE

                new_w = int(
                    w * (MAX_IMAGE_SIDE / h)
                )

            image = image.resize(
                (new_w, new_h),
                Image.LANCZOS,
            )

        return image

    except Exception as e:

        raise RuntimeError(
            f"Image load failed: {e}"
        ) from e


def _resolve_platform(
    image: Image.Image,
) -> Literal["mobile", "web"]:

    try:

        platform = detect_platform(
            image=image,
            ocr_text=None,
        )

        if platform not in [
            "mobile",
            "web",
        ]:
            return "web"

        return platform

    except Exception:

        logger.exception(
            "Platform detection failed"
        )

        return "web"


def generate_from_image(
    image_url: str,
    max_new_tokens: int = 1024,
) -> str:

    if not _model_ready:

        raise RuntimeError(
            "Model not ready"
        )

    processor = _processor

    if processor is None:

        raise RuntimeError(
            "Processor missing"
        )

    start_time = time.time()

    image = _load_image(
        image_url
    )

    adapter = _resolve_platform(
        image
    )

    model = _models.get(adapter)

    if model is None:

        raise RuntimeError(
            f"Adapter not loaded: {adapter}"
        )

    logger.info(
        "Using adapter: %s",
        adapter,
    )

    try:

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": _UI_PROMPT,
                    },
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = (
            process_vision_info(messages)
        )

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

    except Exception as e:

        logger.exception(
            "Preprocess failed"
        )

        raise RuntimeError(
            "Preprocess failed"
        ) from e

    try:

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                repetition_penalty=1.02,
                do_sample=False,
            )

    except Exception as e:

        logger.exception(
            "Inference failed"
        )

        raise RuntimeError(
            "Inference failed"
        ) from e

    try:

        generated_ids = outputs[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        result = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        result = result.strip()

        logger.info(
            "RAW OUTPUT:\n%s",
            result,
        )

        logger.info(
            "Inference completed in %.2fs",
            time.time() - start_time,
        )

        return result

    except Exception as e:

        logger.exception(
            "Decode failed"
        )

        raise RuntimeError(
            "Decode failed"
        ) from e