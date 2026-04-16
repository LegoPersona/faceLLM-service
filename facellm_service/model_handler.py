import json
import re
import io
from PIL import Image
from google import genai

GOOGLE_API_KEY = "AIzaSyAcQ7epIN7R0b8HSKKgYk9gw5J4Z18uUDk"

client = genai.Client(api_key=GOOGLE_API_KEY)

def generate_prompt():
    return """Analyze the person in the image. 
    You are building a Lego character avatar. You MUST map the person's features STRICTLY to the available options provided below. 
    Do not invent categories, colors, or styles. If a feature does not match exactly, choose the closest available option.

    Return ONLY a valid JSON object with these exact fields and STRICTLY allowed values:
    
    - hair: { "style": "Bald" or "Medium_Curly", "color": "Black" or "Brown" or "Yellow" or null (if Bald) }
    - eyebrows: { "shape": "Round" or "Straight", "color": "Black" or "Brown" or "Yellow" }
    - eyes: { "color": "Black" or "Brown" or "Green" }
    - nose: { "shape": "Long" or "Pointy" or "Round" }
    - beard: { "style": "French" or "Full" or "None", "color": "Black" or "Brown" or "Yellow" or null (if None) }
    - shirt: { "color": "Blue" or "Green" or "Red" }
    - pants: { "color": "Black" or "Blue" or "Red" }

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