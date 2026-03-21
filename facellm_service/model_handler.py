import json
from PIL import Image
import io

def generate_prompt():
    """
    Creates an enhanced, strict prompt to ensure the LLM returns ONLY a JSON object.
    Includes explicit instructions to prevent common visual hallucinations (like phantom beards).
    """
    return """You are an expert computer vision assistant. Carefully analyze the provided image of a person's face.

CRITICAL INSTRUCTIONS FOR ANALYSIS:
1. Pay extreme attention to the chin, jawline, and cheeks. 
2. If the skin is smooth, clear, or showing typical makeup without any visible facial hair, "beard" MUST be exactly "No".
3. Look closely at the eyes. If there are no frames resting on the nose/ears, "glasses" MUST be exactly "No".

Output the facial attributes strictly as a JSON object. Do NOT add markdown formatting (like ```json), explanations, or any extra text outside the braces.
The JSON must exactly match this format and use ONLY these allowed values:
{
    "hair_color": "Black" | "Brown" | "Blonde" | "Red" | "Gray" | "White" | "Bald",
    "skin_tone": "Light" | "Medium" | "Dark",
    "glasses": "Yes" | "No",
    "beard": "Yes" | "No"
}"""

def analyze_face(image_bytes: bytes) -> dict:
    """
    Processes the image bytes and queries the FaceLLM.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt = generate_prompt()
        
        # TODO: code that calls to Idiap/FaceLLM-8B
        # Example of response of the model
        mock_response_from_llm = '{"hair_color": "Brown", "skin_tone": "Medium", "glasses": "No", "beard": "Yes"}'

        attributes = json.loads(mock_response_from_llm)
        return attributes

    except json.JSONDecodeError:
        raise ValueError("The LLM did not return a valid JSON format.")
    except Exception as e:
        raise RuntimeError(f"Failed to process image: {str(e)}")