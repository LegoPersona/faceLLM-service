# 🤖🧱 FaceLLM-Service

## 🎯 Purpose
The FaceLLM Service is a dedicated microservice within the LEGOPersona architecture. Its primary responsibility is to interact with the FaceLLM model (Gemini 2.5 Flash) to analyze images of people's faces.
It receives an uploaded image file, processes it, and outputs a strictly formatted JSON object containing predefined facial attributes (such as hair color, skin tone, glasses, and beard). This extracted data is subsequently used by the text embedding service to match the user's features with the most similar LEGO modules

## 🛠️ Tech Stack
* fastapi
* uvicorn
* python-multipart
* google-genai
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

## ⚠️ Limitations & Important Notes
This project utilizes the free tier of the **Google Gemini API** (`gemini-2.5-flash`). Please be aware of the following constraints:

* **Rate Limits:** The API is restricted to a maximum of **15 requests per minute** and **1,500 requests per day**. Exceeding these limits will result in a `429 Too Many Requests` error.
* **Data Privacy:** On the free tier, images and prompts sent to the API may be used by Google to train its models. **Do not upload sensitive, personally identifiable (PII), or medical images.**
* **Safety Filters:** Google enforces strict safety guardrails. Images containing violent, explicit, or overly sensitive content will be blocked by the API, returning a safety exception rather than the expected JSON response.
