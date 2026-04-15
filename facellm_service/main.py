from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_handler import analyze_face
from model_handler import map_json_to_lego_files
import uvicorn

app = FastAPI(title="LegoPersona Face Analysis API")

# Allow cors from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/extract-attributes")
async def extract_attributes(image_file: UploadFile = File(...)):
    image_bytes = await image_file.read()
    
    raw_json = analyze_face(image_bytes)
    
    file_paths = map_json_to_lego_files(raw_json)
    
    return {
        "attributes": raw_json,
        "lego_files": file_paths
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)