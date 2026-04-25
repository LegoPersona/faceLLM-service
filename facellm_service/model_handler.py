import json
import re
import io
from PIL import Image
from google import genai

GOOGLE_API_KEY = "AIzaSyAcQ7epIN7R0b8HSKKgYk9gw5J4Z18uUDk"

client = genai.Client(api_key=GOOGLE_API_KEY)

def generate_prompt():
    return """Analyze the person in the image and describe their physical features to build a Lego character avatar.

For each feature, write a short 2-4 word phrase capturing only the most visually distinctive attributes. These descriptions will be used for semantic vector search to find matching Lego pieces.

Return ONLY a valid JSON object with these exact fields:

- hair: combined style, length, and color (e.g. "short black side part")
- eyebrows: combined shape and color (e.g. "thick dark brown")
- eyes: iris color
- nose: nose shape
- beard: combined style and color, or "none" if no facial hair (e.g. "full black beard")
- shirt: color and basic pattern if any
- pants: color

Output only the raw JSON. No markdown, no explanations."""

def analyze_face(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[generate_prompt(), image]
        )
        text_response = response.text
        
        print(f"DEBUG - Raw API Response:\n{text_response}")
        
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
            
        return json.loads(text_response)

    except Exception as e:
        raise RuntimeError(f"Error during AI analysis: {str(e)}")