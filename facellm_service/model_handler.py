import base64
import json
import os
import re
import io

import requests
from PIL import Image
from google import genai
from openai import OpenAI

# Gemini
GOOGLE_API_KEY = "AIzaSyAcQ7epIN7R0b8HSKKgYk9gw5J4Z18uUDk"

# HuggingFace
HF_MODEL = "Qwen/Qwen3.5-9B:together"

# Ollama (local and cloud)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:latest")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")

# Colman (Requires VPN)
COLMAN_BASE_URL = os.environ.get("COLMAN_BASE_URL", "http://10.10.248.41")
COLMAN_USERNAME = os.environ.get("COLMAN_USERNAME", "student1")
COLMAN_PASSWORD = os.environ.get("COLMAN_PASSWORD", "pass123")
COLMAN_VISION_MODEL = os.environ.get("COLMAN_VISION_MODEL", "gemma3:12b")
COLMAN_TEXT_MODEL = os.environ.get("COLMAN_TEXT_MODEL", "gpt-oss-120b")

client = genai.Client(api_key=GOOGLE_API_KEY)

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)

ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

_FEATURE_KEYS = ["beard", "eyebrows", "eyes", "glasses", "hair", "nose", "pants", "shirt"]

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
                    "properties": {**{k: {"type": "string"} for k in _FEATURE_KEYS}, "skin_tone": {"type": "string"}, "shirt_secondary": {"type": "string"}, "pants_secondary": {"type": "string"}},
                    "required": _FEATURE_KEYS + ["skin_tone", "shirt_secondary", "pants_secondary"],
                    "additionalProperties": False,
                },
                "color_descriptions": {
                    "type": "object",
                    "properties": {**{k: {"type": "string"} for k in _FEATURE_KEYS}, "skin_tone": {"type": "string"}, "shirt_secondary": {"type": "string"}, "pants_secondary": {"type": "string"}},
                    "required": _FEATURE_KEYS + ["skin_tone", "shirt_secondary", "pants_secondary"],
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
- For "shirt_secondary" (in both "colors" and "color_descriptions"), only fill it in if the shirt clearly has a second, distinct color (e.g. stripes, contrast trim, a tie in different color, a collar/cuffs in a different color, color-blocking). Otherwise use "" (empty string) for both. Do not guess a second color on a plain solid-color shirt.
- For "pants_secondary" (in both "colors" and "color_descriptions"), this is the color of the person's shoes. Fill it in whenever shoes are visible in the image, even if their color is close to the pants. If shoes are not visible, use "" (empty string) for both.
- If the person is wearing a one-piece dress (no separate visible shirt/pants), do not force it into a normal shirt+pants split. Set "shapes.shirt" to the dress's sleeve style instead (e.g. sleeveless/tank top, short sleeves, long sleeves), and set "shapes.pants" to "dress".

{
  "shapes": {
    "hair": "<haircut name if identifiable (e.g. bob, mohawk, pompadour) + style (straight/wavy/curly/slicked) + length (short/medium/long). Example: short slicked pompadour>",
    "eyebrows": "<shape descriptor. Example: thick arched, Downturned, Upturned>",
    "eyes": "<eye shape. Example: almond>",
    "nose": "<nose shape. Describe length, width, size... Example: small button, long rounded>",
    "beard": "<beard style, or 'none' if no facial hair. Example: full beard>",
    "glasses": "<glasses style, or 'none' if no glasses. Example: round glasses>",
    "shirt": "<pattern and style (e.g. plain, striped, has a tie, tank top, short sleeves, long sleeves, crop top). Example: long sleeves crop top. If a one-piece dress, describe its sleeve style instead (e.g. strapless, short sleeves, long sleeves, sleeveless/tank top)>",
    "pants": "<style and length (e.g. jeans, shorts, skirt, long pants, short pants). Example: long jeans. If a one-piece dress, use \"dress\">"
  },
  "colors": {
    "hair": "<hair color as hex. Example: #1A1A1A for black, #F5DEB3 for blonde>",
    "eyebrows": "<eyebrow color as hex. Example: #4B3621 for brown>",
    "eyes": "<iris color as hex. Example: #1E90FF for blue, #4B3621 for brown>",
    "nose": "<nose color as hex (same as skin tone). Example: #FFDBB4 for light, #8D5524 for dark>",
    "beard": "<beard color as hex. Use #000000 if no facial hair>",
    "glasses": "<glasses frame color as hex. Use #000000 if no glasses>",
    "shirt": "<shirt color as hex. Example: #FF0000 for red>",
    "shirt_secondary": "<hex of a second, clearly distinct shirt color if present (stripes/trim/color-blocking), otherwise \"\">",
    "pants": "<pants color as hex. Example: #000000 for black>",
    "pants_secondary": "<shoe color as hex if shoes are visible, otherwise \"\">",
    "skin_tone": "<skin tone as hex. Example: #FFDBB4 for light, #8D5524 for dark>"
  },
  "color_descriptions": {
    "hair": "<hair color in plain text. Example: jet black, light blonde>",
    "eyebrows": "<eyebrow color in plain text. Example: dark brown>",
    "eyes": "<iris color in plain text. Example: blue, hazel brown>",
    "nose": "<skin tone in plain text. Example: fair, medium tan, dark brown>",
    "beard": "<beard color in plain text. Use 'none' if no facial hair. Example: black>",
    "glasses": "<glasses frame color in plain text. Use 'none' if no glasses. Example: black>",
    "shirt": "<shirt color in plain text. Example: red, navy blue>",
    "shirt_secondary": "<second shirt color in plain text if present, otherwise \"\". Example: white stripes>",
    "pants": "<pants color in plain text. Example: black, dark blue denim>",
    "pants_secondary": "<shoe color in plain text if shoes are visible, otherwise \"\". Example: white sneakers>",
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
    "glasses": "square glasses",
    "shirt": "plain short sleeves",
    "pants": "long jeans"
  },
  "colors": {
    "hair": "#1A1A1A",
    "eyebrows": "#1A1A1A",
    "eyes": "#4B3621",
    "nose": "#FFDBB4",
    "beard": "#1A1A1A",
    "glasses": "#1A1A1A",
    "shirt": "#1C3A6B",
    "shirt_secondary": "#FF0000",
    "pants": "#2B4B8C",
    "pants_secondary": "#FFFFFF",
    "skin_tone": "#FFDBB4"
  },
  "color_descriptions": {
    "hair": "jet black",
    "eyebrows": "jet black",
    "eyes": "dark brown",
    "nose": "fair",
    "beard": "jet black",
    "glasses": "black",
    "shirt": "navy blue",
    "shirt_secondary": "red",
    "pants": "dark blue",
    "pants_secondary": "white sneakers",
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
                },
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


def _get_colman_headers() -> dict:
    credentials = base64.b64encode(f"{COLMAN_USERNAME}:{COLMAN_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {credentials}", "Content-Type": "application/json"}


def _analyze_face_colman(image_bytes: bytes) -> tuple[dict, dict]:
    b64_image = base64.b64encode(image_bytes).decode()

    url = f"{COLMAN_BASE_URL}/api/generate"
    payload = {
        "model": COLMAN_VISION_MODEL,
        "prompt": generate_prompt(),
        "images": [b64_image],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }

    try:
        response = requests.post(url, json=payload, headers=_get_colman_headers(), timeout=120)
        response.raise_for_status()
    except Exception as e:
        print(f"ERROR - Colman: API request failed: {e}")
        raise

    data = response.json()
    text_response = data.get("response", "")
    if not text_response:
        raise ValueError("Colman API returned empty response")

    tokens = {
        "input": data.get("prompt_eval_count", 0) or 0,
        "output": data.get("eval_count", 0) or 0,
        "total": (data.get("prompt_eval_count", 0) or 0) + (data.get("eval_count", 0) or 0),
    }

    try:
        return _parse_json(text_response), tokens
    except Exception as e:
        print(f"ERROR - Colman: Failed to parse response: {e}")
        print(f"ERROR - Colman: Raw text: {text_response}")
        raise


def analyze_face(image_bytes: bytes) -> dict:
    provider = os.environ.get("LLM_PROVIDER", "genai")
    print(f"[extract-attributes] provider={provider}, image_size={len(image_bytes)}b")
    try:
        if provider == "huggingface":
            attributes, tokens = _analyze_face_huggingface(image_bytes)
        elif provider == "ollama":
            attributes, tokens = _analyze_face_ollama(image_bytes)
        elif provider == "colman":
            attributes, tokens = _analyze_face_colman(image_bytes)
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
                ],
                response_format=_RERANK_SCHEMA,
                temperature=0,
                extra_body={"options": {"num_ctx": 8192}}
            )
            text_response = completion.choices[0].message.content if completion.choices else ""
            if not text_response:
                raise ValueError("Ollama returned empty content")
            print(f"[rerank] Raw response: {text_response}")
            raw = _parse_json(text_response)
            tokens = _extract_openai_tokens(completion)
        elif provider == "colman":
            url = f"{COLMAN_BASE_URL}/v1/chat/completions"
            payload = {
                "model": COLMAN_TEXT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000,
            }
            response = requests.post(url, json=payload, headers=_get_colman_headers(), timeout=120)
            response.raise_for_status()
            data = response.json()
            if not data.get("choices"):
                raise ValueError("Colman API returned no choices")
            text_response = data["choices"][0]["message"]["content"]
            if not text_response:
                raise ValueError("Colman API returned empty content")
            print(f"[rerank] Raw response: {text_response}")
            raw = _parse_json(text_response)
            usage = data.get("usage", {})
            tokens = {
                "input": usage.get("prompt_tokens", 0) or 0,
                "output": usage.get("completion_tokens", 0) or 0,
                "total": usage.get("total_tokens", 0) or 0,
            }
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
