import base64
import json
import os
import re
import io

from PIL import Image
from google import genai
from openai import OpenAI

GOOGLE_API_KEY = "AIzaSyAcQ7epIN7R0b8HSKKgYk9gw5J4Z18uUDk"
HF_MODEL = "Qwen/Qwen3.5-9B:together"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:latest")

client = genai.Client(api_key=GOOGLE_API_KEY)

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)

ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

_FEATURE_KEYS = ["beard", "eyebrows", "eyes", "hair", "nose", "pants", "shirt"]

_ANALYZE_FACE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "analyze_face_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "shapes": {
                    "type": "object",
                    "properties": {k: {"type": "string"} for k in _FEATURE_KEYS},
                    "required": _FEATURE_KEYS,
                    "additionalProperties": False,
                },
                "colors": {
                    "type": "object",
                    "properties": {**{k: {"type": "string"} for k in _FEATURE_KEYS}, "skin_tone": {"type": "string"}},
                    "required": _FEATURE_KEYS + ["skin_tone"],
                    "additionalProperties": False,
                },
                "color_descriptions": {
                    "type": "object",
                    "properties": {**{k: {"type": "string"} for k in _FEATURE_KEYS}, "skin_tone": {"type": "string"}},
                    "required": _FEATURE_KEYS + ["skin_tone"],
                    "additionalProperties": False,
                },
            },
            "required": ["shapes", "colors", "color_descriptions"],
            "additionalProperties": False,
        },
    },
}

_RERANK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "rerank_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                k: {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "best_match": {"type": "string"},
                    },
                    "required": ["index", "best_match"],
                    "additionalProperties": False,
                }
                for k in _FEATURE_KEYS
            },
            "required": _FEATURE_KEYS,
            "additionalProperties": False,
        },
    },
}


def generate_prompt():
    return """You are an AI component in an automated pipeline. Your output will be parsed directly by code — not read by a human. Any text outside the required JSON format will cause a parsing error and break the pipeline. Analyze the person in the image and describe their physical features to build a Lego character avatar.

Return ONLY a valid JSON object that matches this exact template. Replace each placeholder with the appropriate value, keeping in mind the following:
- Color hex values should represent the perceived color of the feature, similar to the color description, not the literal pixel colors in the image.
- Keep color hex values of facial hair, eyebrows, and hair THE SAME unless they are clearly, radically different in color. For example, if the person has brown hair and a lighter brown beard, use the same hex for both but describe the beard color as "light brown" in the color description.
- Pay close attention to eye color. DO NOT confuse eye color with hair/eyebrow color. If the person has blue eyes and black hair, the hair and eyebrow hex values should be the same (e.g. #1A1A1A) while the eye hex value should be different (e.g. #1E90FF).

{
  "shapes": {
    "hair": "<haircut name if identifiable (e.g. bob, mohawk, pompadour) + style (straight/wavy/curly/slicked) + length (short/medium/long). Example: short slicked pompadour>",
    "eyebrows": "<shape descriptor. Example: thick arched>",
    "eyes": "<eye shape. Example: almond>",
    "nose": "<nose shape. Example: button>",
    "beard": "<beard style, or 'none' if no facial hair. Example: full beard>",
    "shirt": "<basic pattern or style. Example: plain>",
    "pants": "<pants style. Example: jeans>"
  },
  "colors": {
    "hair": "<hair color as hex. Example: #1A1A1A for black, #F5DEB3 for blonde>",
    "eyebrows": "<eyebrow color as hex. Example: #4B3621 for brown>",
    "eyes": "<iris color as hex. Example: #1E90FF for blue, #4B3621 for brown>",
    "nose": "<nose color as hex (same as skin tone). Example: #FFDBB4 for light, #8D5524 for dark>",
    "beard": "<beard color as hex. Use #000000 if no facial hair>",
    "shirt": "<shirt color as hex. Example: #FF0000 for red>",
    "pants": "<pants color as hex. Example: #000000 for black>",
    "skin_tone": "<skin tone as hex. Example: #FFDBB4 for light, #8D5524 for dark>"
  },
  "color_descriptions": {
    "hair": "<hair color in plain text. Example: jet black, platinum blonde>",
    "eyebrows": "<eyebrow color in plain text. Example: dark brown>",
    "eyes": "<iris color in plain text. Example: blue, hazel brown>",
    "nose": "<skin tone in plain text. Example: fair, medium tan, dark brown>",
    "beard": "<beard color in plain text. Use 'none' if no facial hair. Example: black>",
    "shirt": "<shirt color in plain text. Example: red, navy blue>",
    "pants": "<pants color in plain text. Example: black, dark blue denim>",
    "skin_tone": "<skin tone in plain text. Example: fair, medium tan, dark brown>"
  }
}

Here is a filled example for reference:

{
  "shapes": {
    "hair": "short slicked pompadour",
    "eyebrows": "thick arched",
    "eyes": "almond",
    "nose": "button",
    "beard": "short stubble",
    "shirt": "plain",
    "pants": "jeans"
  },
  "colors": {
    "hair": "#1A1A1A",
    "eyebrows": "#1A1A1A",
    "eyes": "#4B3621",
    "nose": "#FFDBB4",
    "beard": "#1A1A1A",
    "shirt": "#1C3A6B",
    "pants": "#2B4B8C",
    "skin_tone": "#FFDBB4"
  },
  "color_descriptions": {
    "hair": "jet black",
    "eyebrows": "jet black",
    "eyes": "dark brown",
    "nose": "fair",
    "beard": "jet black",
    "shirt": "navy blue",
    "pants": "dark blue",
    "skin_tone": "fair"
  }
}

Your response MUST be a single valid JSON object exactly like the structure above with all three top-level keys (shapes, colors, color_descriptions) — no extra fields, no missing fields, no markdown, no explanations, no code blocks."""


def _image_to_data_url(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    fmt = image.format or "JPEG"
    mime = f"image/{fmt.lower()}"
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


def _parse_json(text: str) -> dict:
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except Exception as e:
            print(f"ERROR - JSON Parser: Failed to parse: {e}")
            raise

    try:
        return json.loads(text)
    except Exception as e:
        print(f"ERROR - JSON Parser: Failed to parse: {e}")
        raise


def _extract_genai_tokens(response) -> dict:
    usage = response.usage_metadata
    return {
        "input": getattr(usage, "prompt_token_count", 0) or 0,
        "output": getattr(usage, "candidates_token_count", 0) or 0,
        "total": getattr(usage, "total_token_count", 0) or 0,
    }


def _extract_openai_tokens(completion) -> dict:
    usage = completion.usage
    if not usage:
        return {"input": 0, "output": 0, "total": 0}
    return {
        "input": usage.prompt_tokens or 0,
        "output": usage.completion_tokens or 0,
        "total": usage.total_tokens or 0,
    }


def _analyze_face_genai(image_bytes: bytes) -> tuple[dict, dict]:
    image = Image.open(io.BytesIO(image_bytes))
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[generate_prompt(), image],
        )
    except Exception as e:
        print(f"ERROR - GenAI: API request failed: {e}")
        raise

    try:
        return _parse_json(response.text), _extract_genai_tokens(response)
    except Exception as e:
        print(f"ERROR - GenAI: Failed to parse response: {e}")
        raise


def _analyze_face_huggingface(image_bytes: bytes) -> tuple[dict, dict]:
    try:
        data_url = _image_to_data_url(image_bytes)
    except Exception as e:
        print(f"ERROR - HF: Failed to create data URL: {e}")
        raise

    try:
        completion = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": generate_prompt()},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
    except Exception as e:
        print(f"ERROR - HF: API request failed: {e}")
        raise

    if not completion.choices:
        raise ValueError("HF API returned no choices")

    text_response = completion.choices[0].message.content
    if not text_response:
        raise ValueError("HF API returned empty content")

    try:
        return _parse_json(text_response), _extract_openai_tokens(completion)
    except Exception as e:
        print(f"ERROR - HF: Failed to parse response: {e}")
        print(f"ERROR - HF: Raw text: {text_response}")
        raise


def _analyze_face_ollama(image_bytes: bytes) -> tuple[dict, dict]:
    try:
        data_url = _image_to_data_url(image_bytes)
    except Exception as e:
        print(f"ERROR - Ollama: Failed to create data URL: {e}")
        raise

    try:
        completion = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": generate_prompt()},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            response_format=_ANALYZE_FACE_SCHEMA,
            temperature=0,
            extra_body={"options": {"num_ctx": 8192}}
        )
    except Exception as e:
        print(f"ERROR - Ollama: API request failed: {e}")
        raise

    if not completion.choices:
        raise ValueError("Ollama returned no choices")

    text_response = completion.choices[0].message.content
    if not text_response:
        raise ValueError("Ollama returned empty content")

    try:
        return _parse_json(text_response), _extract_openai_tokens(completion)
    except Exception as e:
        print(f"ERROR - Ollama: Failed to parse response: {e}")
        print(f"ERROR - Ollama: Raw text: {text_response}")
        raise


def analyze_face(image_bytes: bytes) -> dict:
    provider = os.environ.get("LLM_PROVIDER", "genai")
    print(f"[extract-attributes] provider={provider}, image_size={len(image_bytes)}b")
    try:
        if provider == "huggingface":
            attributes, tokens = _analyze_face_huggingface(image_bytes)
        elif provider == "ollama":
            attributes, tokens = _analyze_face_ollama(image_bytes)
        else:
            attributes, tokens = _analyze_face_genai(image_bytes)
        print(f"[extract-attributes] result={attributes}, tokens={tokens}")
        return {"attributes": attributes, "tokens_used": tokens}
    except Exception as e:
        print(f"ERROR [extract-attributes]: {str(e)}")
        raise RuntimeError(f"Error during AI analysis: {str(e)}")


def _generate_selection_prompt(features: dict) -> str:
    input_obj = {
        feature: {
            "description": data["description"],
            "candidates": [{"index": i, "value": c} for i, c in enumerate(data["candidates"])],
        }
        for feature, data in features.items()
    }
    input_json = json.dumps(input_obj, indent=2)

    feature_keys = list(features.keys())
    template_obj = {k: {"index": "<integer>", "best_match": "<candidate value copied exactly>"} for k in feature_keys}
    template_json = json.dumps(template_obj, indent=2)

    return f"""/no_think
You are an AI component in an automated pipeline. Your output will be parsed directly by code — not read by a human. Any text outside the required JSON format will cause a parsing error and break the pipeline.

Your task: for each feature, select the candidate that is visually closest to the given description. COLOR match is the most important factor — prioritize candidates whose color best matches the description, then consider shape/style as a secondary factor.

INPUT:
{input_json}

Return ONLY a valid JSON object that matches this exact template. Replace each placeholder with the appropriate value:

{template_json}

The "index" must be the integer index of the chosen candidate. The "best_match" must be the candidate "value" string copied exactly.

Your response MUST be a single valid JSON object exactly like the template above — one entry per feature, no extra fields, no markdown, no explanations, no code blocks."""


def select_best_matches(features: dict) -> dict:
    provider = os.environ.get("LLM_PROVIDER", "genai")
    print(f"[rerank] provider={provider}, features={list(features.keys())}")

    prompt = _generate_selection_prompt(features)
    print(prompt)  # Debug: print the generated prompt

    try:
        if provider == "huggingface":
            completion = hf_client.chat.completions.create(
                model=HF_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text_response = completion.choices[0].message.content if completion.choices else ""
            if not text_response:
                raise ValueError("HF API returned empty content")
            raw = _parse_json(text_response)
            tokens = _extract_openai_tokens(completion)
        elif provider == "ollama":
            completion = ollama_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"},
                ],
                response_format=_RERANK_SCHEMA,
                temperature=0,
                extra_body={"options": {"num_ctx": 8192}}
            )
            text_response = completion.choices[0].message.content if completion.choices else ""
            if not text_response:
                raise ValueError("Ollama returned empty content")
            print(f"[rerank] Raw response: {text_response}")
            raw = _parse_json("{" + text_response)
            tokens = _extract_openai_tokens(completion)
        else:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt],
            )
            raw = _parse_json(response.text)
            tokens = _extract_genai_tokens(response)
    except Exception as e:
        print(f"ERROR [rerank]: LLM call failed: {e}")
        raise RuntimeError(f"Error during best match selection: {str(e)}")

    result = {}
    for feature, data in features.items():
        entry = raw.get(feature, {})
        index = int(entry["index"])
        result[feature] = {"index": index, "best_match": data["candidates"][index]}

    print(f"[rerank] result={result}, tokens={tokens}")
    return {"result": result, "tokens_used": tokens}
