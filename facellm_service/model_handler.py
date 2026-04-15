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


def map_json_to_lego_files(data: dict) -> dict:
    paths = {}

    hair = data.get("hair", {})
    if hair.get("style") == "Bald":
        paths["hair"] = "templates/hair/bald_hair.ldr"
    else:
        # דוגמה: black_medium_curly_hair.ldr
        style = hair.get("style", "").lower()
        color = hair.get("color", "").lower()
        paths["hair"] = f"templates/hair/{color}_{style}_hair.ldr"

    eb = data.get("eyebrows", {})
    # דוגמה: brown_round_eyebrows.ldr
    paths["eyebrows"] = f"templates/eyebrows/{eb.get('color').lower()}_{eb.get('shape').lower()}_eyebrows.ldr"

    eyes = data.get("eyes", {})
    # דוגמה: green_eyes.ldr
    paths["eyes"] = f"templates/eyes/{eyes.get('color').lower()}_eyes.ldr"

    nose = data.get("nose", {})
    # דוגמה: pointy_nose.ldr
    paths["nose"] = f"templates/nose/{nose.get('shape').lower()}_nose.ldr"

    beard = data.get("beard", {})
    if beard.get("style") == "None":
        paths["beard"] = "templates/beard/no_beard.ldr"
    else:
        # דוגמה: black_french_beard.ldr
        paths["beard"] = f"templates/beard/{beard.get('color').lower()}_{beard.get('style').lower()}_beard.ldr"

    shirt = data.get("shirt", {})
    # דוגמה: blue_shirt.ldr
    paths["shirt"] = f"templates/shirt/{shirt.get('color').lower()}_shirt.ldr"

    pants = data.get("pants", {})
    # דוגמה: red_pants.ldr
    paths["pants"] = f"templates/pants/{pants.get('color').lower()}_pants.ldr"

    return paths