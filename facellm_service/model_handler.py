import json
import re
import torch
import io
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "Idiap/FaceLLM-8B"

print(f"Loading tokenizer and model {MODEL_ID}... This might take a minute.")

# FaceLLM uses InternVL architecture, which requires AutoTokenizer and AutoModel
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto"
).eval()
print("Model loaded successfully!")

def build_transform(input_size=448):
    """Image preprocessing required by the InternVL architecture."""
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])

def generate_prompt():
    return """You are an expert computer vision assistant. Analyze the face in the image carefully.

You MUST output ONLY a valid JSON object. Do not include markdown formatting.
Follow this exact schema. Pay special attention to the 'analysis' field: describe the skin smoothness and presence/absence of facial hair first.

{
    "analysis": "Briefly describe the face here, specifically noting if the skin is smooth or if there is any actual facial hair/beard...",
    "hair_color": "Black" | "Brown" | "Blonde" | "Red" | "Gray" | "White" | "Bald",
    "skin_tone": "Light" | "Medium" | "Dark",
    "glasses": "Yes" | "No",
    "beard": "Yes" | "No"
}"""

def extract_json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse valid JSON from model output. Raw output: {text}")

def analyze_face(image_bytes: bytes) -> dict:
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess the image to tensor for InternVL
        transform = build_transform()
        pixel_values = transform(image).unsqueeze(0).to(model.device, dtype=torch.float16)
        
        prompt_text = generate_prompt()
        
        # Generation config
        generation_config = dict(max_new_tokens=150, do_sample=False, temperature=0.1)
        
        # FaceLLM / InternVL inference uses the built-in chat method
        response, history = model.chat(tokenizer, pixel_values, prompt_text, generation_config)
        
        print(f"DEBUG - Raw output with analysis:\n{response}")
        
        result_dict = extract_json_from_text(response)
        result_dict.pop("analysis", None)
        return result_dict

    except Exception as e:
        raise RuntimeError(f"Failed to process image with LLM: {str(e)}")