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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nemotron3:33b")

client = genai.Client(api_key=GOOGLE_API_KEY)

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)

ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def generate_prompt():
    return """Analyze the person in the image and describe their physical features to build a Lego character avatar.

Return ONLY a valid JSON object with two top-level keys: "shapes" and "colors".

"shapes" contains a short shape/style description for each feature (no color words). These will be used for semantic vector search to find matching Lego pieces by shape.
"colors" contains only the color for each feature as a hexadecimal color code (e.g. "#000000"). These will be used separately to find matching Lego piece colors.

Both objects must have the same exact fields:

- hair: shapes = haircut name if identifiable (e.g. "bob", "mohawk", "pompadour"), combined with style (straight/wavy/curly/slicked) and length (short/medium/long). Example: "short slicked pompadour", colors = hair color as hex (e.g. "#000000" for black, "#F5DEB3" for blonde)
- eyebrows: shapes = shape descriptor (e.g. "thick arched"), colors = eyebrow color as hex (e.g. "#4B3621" for brown)
- eyes: shapes = eye shape (e.g. "almond", "round"), colors = iris color as hex (e.g. "#1E90FF" for blue, "#4B3621" for brown)
- nose: shapes = nose shape (e.g. "button", "wide", "pointed"), colors = skin tone as hex (e.g. "#FFDBB4" for light, "#8D5524" for dark)
- beard: shapes = beard style or "none" if no facial hair (e.g. "full beard", "stubble", "none"), colors = beard color as hex or "#000000" if no facial hair
- shirt: shapes = basic pattern or style (e.g. "plain", "striped", "graphic tee"), colors = shirt color as hex (e.g. "#FF0000" for red, "#000080" for navy blue)
- pants: shapes = pants style (e.g. "jeans", "chinos", "shorts"), colors = pants color as hex (e.g. "#000000" for black, "#C3B091" for khaki)

Output only the raw JSON. No markdown, no explanations."""


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
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": generate_prompt()},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
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
    feature_blocks = []
    for feature, data in features.items():
        candidates_text = "\n".join(
            f"  {i}: \"{c}\"" for i, c in enumerate(data["candidates"])
        )
        feature_blocks.append(
            f"--- {feature} ---\n"
            f"Description: \"{data['description']}\"\n"
            f"Candidates:\n{candidates_text}"
        )

    return f"""You are matching feature descriptions to the closest LEGO part descriptions.

For each feature below, select the candidate that is the closest semantic match to the given description.

{chr(10).join(feature_blocks)}

Return ONLY a valid JSON object with one entry per feature. Each entry must contain:
- "index": the 0-based position of the chosen candidate (integer)
- "best_match": the chosen candidate string copied exactly

Example output format:
{{
  "hair": {{"index": 0, "best_match": "black short pompadour"}},
  "beard": {{"index": 2, "best_match": "black full beard"}},
  "eyes": {{"index": 1, "best_match": "brown almond eyes"}}
}}

Include every feature from the input. Output only the raw JSON, no markdown, no explanations."""


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
                messages=[{"role": "user", "content": prompt}],
            )
            text_response = completion.choices[0].message.content if completion.choices else ""
            if not text_response:
                raise ValueError("Ollama returned empty content")
            raw = _parse_json(text_response)
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
