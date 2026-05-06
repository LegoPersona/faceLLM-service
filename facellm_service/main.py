from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_handler import analyze_face, select_best_matches
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
    print(f"[extract-attributes] file={image_file.filename}, type={image_file.content_type}")

    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await image_file.read()
        return analyze_face(image_bytes)
    except Exception as e:
        print(f"ERROR [extract-attributes]: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class FeatureSelection(BaseModel):
    description: str
    candidates: list[str]

class RerankRequest(BaseModel):
    features: dict[str, FeatureSelection]


@app.post("/api/v1/rerank")
async def rerank(request: RerankRequest):
    print(f"[rerank] features={list(request.features.keys())}")
    try:
        features_raw = {
            feature: {"description": data.description, "candidates": data.candidates}
            for feature, data in request.features.items()
        }
        return select_best_matches(features_raw)
    except Exception as e:
        print(f"ERROR [rerank]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
