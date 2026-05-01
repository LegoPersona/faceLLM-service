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

client = genai.Client(api_key=GOOGLE_API_KEY)

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)


def generate_prompt():
    return """Analyze the person in the image and describe their physical features to build a Lego character avatar.

For each feature, write a short 2-6 word phrase capturing only the most visually distinctive attributes. These descriptions will be used for semantic vector search to find matching Lego pieces.

Return ONLY a valid JSON object with these exact fields:

- hair: color, haircut name if identifiable (e.g. "bob", "mohawk", "pompadour"), combined with style (straight/wavy/curly/slicked), length (short/medium/long). Example: "black short slicked pompadour" or "blonde long wavy"
- eyebrows: combined shape and color (e.g. "thick brown")
- eyes: iris color
- nose: nose shape
- beard: color and combined style, or "none" if no facial hair (e.g. "black full beard")
- shirt: color and basic pattern if any
- pants: color

Output only the raw JSON. No markdown, no explanations."""


def _image_to_data_url(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    fmt = image.format or "JPEG"
    mime = f"image/{fmt.lower()}"
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


def _parse_json(text: str) -> dict:
    print(f"DEBUG - JSON Parser: Attempting to parse: {text[:200]}..." if len(text) > 200 else f"DEBUG - JSON Parser: Attempting to parse: {text}")
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        print(f"DEBUG - JSON Parser: Found JSON object via regex")
        try:
            result = json.loads(json_match.group(0))
            print(f"DEBUG - JSON Parser: Successfully parsed via regex")
            return result
        except Exception as e:
            print(f"ERROR - JSON Parser: Failed to parse regex match: {e}")
            raise
    
    print(f"DEBUG - JSON Parser: No JSON object found via regex, trying direct parse")
    try:
        result = json.loads(text)
        print(f"DEBUG - JSON Parser: Successfully parsed directly")
        return result
    except Exception as e:
        print(f"ERROR - JSON Parser: Failed direct parse: {e}")
        raise


def _analyze_face_genai(image_bytes: bytes) -> dict:
    print(f"DEBUG - GenAI: Starting image analysis. Image size: {len(image_bytes)} bytes")
    image = Image.open(io.BytesIO(image_bytes))
    print(f"DEBUG - GenAI: Image loaded. Format: {image.format}, Size: {image.size}")
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[generate_prompt(), image],
        )
        print(f"DEBUG - GenAI: API request completed successfully")
    except Exception as e:
        print(f"ERROR - GenAI: API request failed: {e}")
        raise
    
    print(f"DEBUG - GenAI: Response object: {response}")
    print(f"DEBUG - GenAI: Response text length: {len(response.text)}")
    print(f"DEBUG - GenAI: Raw GenAI Response:\n{response.text}")
    
    try:
        result = _parse_json(response.text)
        print(f"DEBUG - GenAI: Successfully parsed response")
        return result
    except Exception as e:
        print(f"ERROR - GenAI: Failed to parse response: {e}")
        raise


def _analyze_face_huggingface(image_bytes: bytes) -> dict:
    print(f"DEBUG - HF: Starting image analysis. Image size: {len(image_bytes)} bytes")
    
    try:
        print(f"DEBUG - HF: Creating data URL from image...")
        data_url = _image_to_data_url(image_bytes)
        print(f"DEBUG - HF: Data URL created successfully. Length: {len(data_url)}")
    except Exception as e:
        print(f"ERROR - HF: Failed to create data URL: {e}")
        raise
    
    print(f"DEBUG - HF: Model being used: {HF_MODEL}")
    print(f"DEBUG - HF: Sending request to HF API...")
    
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
        print(f"DEBUG - HF: API request completed successfully")
    except Exception as e:
        print(f"ERROR - HF: API request failed: {e}")
        raise
    
    print(f"DEBUG - HF: Completion object: {completion}")
    print(f"DEBUG - HF: Choices length: {len(completion.choices)}")
    
    if not completion.choices:
        print(f"ERROR - HF: No choices in response")
        raise ValueError("HF API returned no choices")
    
    message = completion.choices[0].message
    print(f"DEBUG - HF: Message object: {message}")
    print(f"DEBUG - HF: Message content: {message.content}")
    print(f"DEBUG - HF: Message content type: {type(message.content)}")
    print(f"DEBUG - HF: Message content length: {len(message.content) if message.content else 0}")
    
    text_response = message.content
    
    if not text_response:
        print(f"ERROR - HF: Empty response from HF")
        raise ValueError("HF API returned empty content")
    
    print(f"DEBUG - HF: Raw HF Response:\n{text_response}")
    
    try:
        result = _parse_json(text_response)
        print(f"DEBUG - HF: Successfully parsed JSON response: {result}")
        return result
    except Exception as e:
        print(f"ERROR - HF: Failed to parse JSON response: {e}")
        print(f"ERROR - HF: Raw text that failed to parse: {text_response}")
        raise


def analyze_face(image_bytes: bytes) -> dict:
    provider = os.environ.get("LLM_PROVIDER", "genai")
    print(f"DEBUG - Main: Using LLM provider: {provider}")
    print(f"DEBUG - Main: Image size: {len(image_bytes)} bytes")
    
    try:
        if provider == "huggingface":
            print(f"DEBUG - Main: Delegating to HuggingFace provider")
            return _analyze_face_huggingface(image_bytes)
        print(f"DEBUG - Main: Delegating to GenAI provider")
        return _analyze_face_genai(image_bytes)
    except Exception as e:
        print(f"ERROR - Main: Error during AI analysis: {str(e)}")
        raise RuntimeError(f"Error during AI analysis: {str(e)}")
