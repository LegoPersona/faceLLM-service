from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_handler import analyze_face
import uvicorn

app = FastAPI(title="LegoPersona Face Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/extract-attributes")
async def extract_attributes(image_file: UploadFile = File(...)):
    print(f"DEBUG - Endpoint: Received file: {image_file.filename}, ContentType: {image_file.content_type}")
    
    if not image_file.content_type.startswith("image/"):
        print(f"ERROR - Endpoint: Invalid content type: {image_file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await image_file.read()
        print(f"DEBUG - Endpoint: Read image bytes. Size: {len(image_bytes)} bytes")
        
        result = analyze_face(image_bytes)
        
        print(f"DEBUG - Endpoint: Analysis complete. Result: {result}")
        return result
    except Exception as e:
        print(f"ERROR - Endpoint: Exception occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
