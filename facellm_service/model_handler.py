import json
from PIL import Image
import io

def generate_prompt():
    """
    Creates a strict prompt to ensure the LLM returns ONLY a JSON object
    with the exact attributes required by the Lego Service.
    """
    return """
    You are an expert computer vision assistant. Analyze the provided image of a person's face.
    You MUST extract the visual features and output them strictly as a JSON object.
    Do NOT output any conversational text, markdown formatting (like ```json), or explanations. 
    Return ONLY a raw JSON dictionary matching this exact schema:

    {
        "hair_color": "Black" | "Brown" | "Blonde" | "Red" | "Gray" | "White" | "Bald",
        "skin_tone": "Light" | "Medium" | "Dark",
        "glasses": "Yes" | "No",
        "beard": "Yes" | "No"
    }
    """

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