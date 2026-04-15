import json
import re
import io
from PIL import Image
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyAcQ7epIN7R0b8HSKKgYk9gw5J4Z18uUDk"
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')

def generate_prompt():
    return """Analyze the person in the image. 
    IMPORTANT: Only provide details for items that are clearly visible in the image.
    If the image is a close-up of a face and clothes/pants are not visible, set their values to null.

    Return ONLY a valid JSON object with these fields:
    - hair: { "color": str, "length": "Short/Medium/Long/Bald", "style": "Straight/Wavy/Curly/None" }
    - eyebrows: { "color": str, "shape": "Thin/Thick/Arched/Straight" }
    - eyes: { "color": str }
    - nose: { "shape": "Straight/Hooked/Button/Wide/Small" }
    - beard: { "present": "Yes/No", "color": str or null, "style": str or null, "shape": str or null }
    - shirt: { "type": str or null, "color": str or null }
    - pants: { "type": str or null, "color": str or null }
    - glasses: "Yes/No"
    - skin_tone: "Light/Medium/Dark"

    Output only the JSON."""

def analyze_face(image_bytes: bytes) -> dict:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        response = model.generate_content([generate_prompt(), image])
        text_response = response.text
        
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
            
        return json.loads(text_response)

    except Exception as e:
        raise RuntimeError(f"Error during AI analysis: {str(e)}")