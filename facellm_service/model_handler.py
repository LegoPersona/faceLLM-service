import json
from PIL import Image
import io

def generate_prompt():
    """
    Uses Chain of Thought (CoT) prompting within the JSON structure to force the model 
    to visually analyze the face BEFORE committing to the strict boolean/categorical values.
    """
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

def analyze_face(image_bytes: bytes) -> dict:
    """
    Processes the image bytes, queries the FaceLLM, and returns the attributes as a dictionary.
    """
    try:
        import io
        from PIL import Image
        import torch
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        prompt_text = generate_prompt()
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]}
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1, 
                do_sample=False
            )
            
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        
        print(f"DEBUG - Raw output with analysis:\n{generated_text}")
        
        result_dict = extract_json_from_text(generated_text)
        
        result_dict.pop("analysis", None)
        
        return result_dict

    except Exception as e:
        raise RuntimeError(f"Failed to process image with LLM: {str(e)}")