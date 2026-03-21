from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from model_handler import analyze_face

app = FastAPI(title="FaceLLM Service", description="Microservice for LegoPersona to extract facial attributes")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FaceLLM"}

@app.post("/api/v1/extract-attributes")
async def extract_facial_attributes(image_file: UploadFile = File(...)):
    """
    Receives an image, passes it to the FaceLLM, and returns a JSON of facial attributes.
    """

    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    try:
        image_bytes = await image_file.read()
        
        attributes = analyze_face(image_bytes)
        
        return attributes
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)