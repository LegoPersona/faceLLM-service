# 🤖🧱 FaceLLM-Service

## 🎯 Purpose
The FaceLLM Service is a dedicated microservice within the LEGOPersona architecture. Its primary responsibility is to interact with the FaceLLM model (gemini-1.5-flash) to analyze images of people's faces.
It receives an uploaded image file, processes it, and outputs a strictly formatted JSON object containing predefined facial attributes (such as hair color, skin tone, glasses, and beard). This extracted data is subsequently used by the text embedding service to match the user's features with the most similar LEGO modules

## 🛠️ Tech Stack
* fastapi
* uvicorn
* python-multipart
* google-generativeai
* Pillow

## 📂 Project Structure
```text
facellm_service/
├── main.py               # FastAPI application entry point and routes
├── model_handler.py      # Logic for loading Idiap/FaceLLM-8B and generating prompts
└── README.md             # Service documentation
```

## 🚀 How to Run (Local)
1. pip install all tech-stack (requirements.txt)
2. run
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
3. to swagger: http://localhost:8000/docs
