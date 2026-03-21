from fastapi import FastAPI
import uvicorn

app = FastAPI(title="FaceLLM Service", description="Microservice for LegoPersona to extract facial attributes")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FaceLLM"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)