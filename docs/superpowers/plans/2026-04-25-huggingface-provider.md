# HuggingFace Inference Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `?provider=huggingface` query param to the extract-attributes endpoint, routing to `Qwen/Qwen3.5-9B:together` via HuggingFace's OpenAI-compatible router.

**Architecture:** Single conditional branch added to `model_handler.py`; `main.py` gains a `provider` query param that it passes through. Image bytes are base64-encoded into a data URL for the HF path. GenAI path is untouched.

**Tech Stack:** FastAPI, google-genai (existing), openai SDK (new, pointed at HF router), Pillow (existing), pytest + httpx (test infra)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `facellm_service/requirements.txt` | Modify | Add `openai` runtime dep |
| `requirements-dev.txt` | Create | Test deps: pytest, httpx |
| `facellm_service/main.py` | Modify | Add `provider` query param, validate it, pass to `analyze_face` |
| `facellm_service/model_handler.py` | Modify | Add HF client, `_image_to_data_url`, `_analyze_face_huggingface`, update `analyze_face` signature |
| `tests/test_api.py` | Create | Tests for provider routing, validation, both backends |

---

### Task 1: Add dependencies

**Files:**
- Modify: `facellm_service/requirements.txt`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Add `openai` to runtime requirements**

Open `facellm_service/requirements.txt` and make it read:

```
fastapi
uvicorn
python-multipart
google-genai
pillow
openai
```

- [ ] **Step 2: Create dev requirements file**

Create `requirements-dev.txt` at the repo root:

```
pytest
httpx
```

- [ ] **Step 3: Install dev deps**

```bash
pip install pytest httpx openai
```

Expected: installs without errors.

- [ ] **Step 4: Commit**

```bash
git add facellm_service/requirements.txt requirements-dev.txt
git commit -m "chore: add openai runtime dep and test dev deps"
```

---

### Task 2: Add `provider` query param to the endpoint

**Files:**
- Modify: `facellm_service/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Create test file with failing tests**

Create `tests/test_api.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "facellm_service"))

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Patch clients before importing app so module-level init doesn't fail
with patch("model_handler.genai.Client"), patch("model_handler.OpenAI"):
    from main import app

client = TestClient(app)

FAKE_IMAGE = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
    b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
    b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

def make_upload(data=FAKE_IMAGE, content_type="image/png"):
    return ("image_file", ("face.png", data, content_type))


def test_invalid_provider_returns_400():
    resp = client.post(
        "/api/v1/extract-attributes?provider=badprovider",
        files=[make_upload()],
    )
    assert resp.status_code == 400
    assert "badprovider" in resp.json()["detail"]


def test_non_image_file_returns_400():
    resp = client.post(
        "/api/v1/extract-attributes",
        files=[make_upload(b"hello", "text/plain")],
    )
    assert resp.status_code == 400


def test_default_provider_calls_genai(monkeypatch):
    # patch the name bound in main's namespace (from model_handler import analyze_face)
    import main as main_module
    monkeypatch.setattr(
        main_module,
        "analyze_face",
        lambda image_bytes, provider="genai": {"hair": "short black", "provider_used": provider},
    )
    resp = client.post(
        "/api/v1/extract-attributes",
        files=[make_upload()],
    )
    assert resp.status_code == 200
    assert resp.json()["provider_used"] == "genai"


def test_huggingface_provider_passed_through(monkeypatch):
    import main as main_module
    monkeypatch.setattr(
        main_module,
        "analyze_face",
        lambda image_bytes, provider="genai": {"hair": "short black", "provider_used": provider},
    )
    resp = client.post(
        "/api/v1/extract-attributes?provider=huggingface",
        files=[make_upload()],
    )
    assert resp.status_code == 200
    assert resp.json()["provider_used"] == "huggingface"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd faceLLM-service
pytest tests/test_api.py -v
```

Expected: tests fail because `main.py` doesn't accept `provider` param yet.

- [ ] **Step 3: Update `main.py`**

Replace the full contents of `facellm_service/main.py` with:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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

VALID_PROVIDERS = {"genai", "huggingface"}

@app.post("/api/v1/extract-attributes")
async def extract_attributes(
    image_file: UploadFile = File(...),
    provider: str = Query("genai"),
):
    if provider not in VALID_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{provider}'. Valid options: {sorted(VALID_PROVIDERS)}",
        )
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await image_file.read()
        return analyze_face(image_bytes, provider)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_api.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add facellm_service/main.py tests/test_api.py
git commit -m "feat: add provider query param to extract-attributes endpoint"
```

---

### Task 3: Implement HuggingFace branch in `model_handler.py`

**Files:**
- Modify: `facellm_service/model_handler.py`
- Modify: `tests/test_api.py` (add backend-level tests)

- [ ] **Step 1: Add backend tests to `tests/test_api.py`**

Append these tests to the end of `tests/test_api.py`:

```python
import base64
import json
from unittest.mock import MagicMock, patch


def test_analyze_face_genai_returns_parsed_json():
    mock_response = MagicMock()
    mock_response.text = '{"hair": "short black", "eyes": "brown"}'

    with patch("model_handler.client") as mock_client:
        mock_client.models.generate_content.return_value = mock_response
        import model_handler
        result = model_handler.analyze_face(FAKE_IMAGE, "genai")

    assert result == {"hair": "short black", "eyes": "brown"}


def test_analyze_face_huggingface_returns_parsed_json():
    mock_choice = MagicMock()
    mock_choice.message.content = '{"hair": "long red", "eyes": "green"}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with patch("model_handler.hf_client") as mock_hf:
        mock_hf.chat.completions.create.return_value = mock_completion
        import model_handler
        result = model_handler.analyze_face(FAKE_IMAGE, "huggingface")

    assert result == {"hair": "long red", "eyes": "green"}


def test_image_to_data_url_format():
    import model_handler
    url = model_handler._image_to_data_url(FAKE_IMAGE)
    assert url.startswith("data:image/")
    assert ";base64," in url
    # verify the base64 portion decodes without error
    b64_part = url.split(";base64,")[1]
    base64.b64decode(b64_part)
```

- [ ] **Step 2: Run new tests — expect FAIL**

```bash
pytest tests/test_api.py::test_analyze_face_genai_returns_parsed_json \
       tests/test_api.py::test_analyze_face_huggingface_returns_parsed_json \
       tests/test_api.py::test_image_to_data_url_format -v
```

Expected: FAIL — `hf_client` and `_image_to_data_url` don't exist yet, `analyze_face` doesn't accept `provider`.

- [ ] **Step 3: Rewrite `model_handler.py`**

Replace the full contents of `facellm_service/model_handler.py` with:

```python
import base64
import json
import os
import re
import io

from PIL import Image
from google import genai
from openai import OpenAI

GOOGLE_API_KEY = "AIzaSyAcQ7epIN7R0b8HSKKgYk9gw5J4Z18uUDk"
HF_MODEL = "Qwen/Qwen3.5-9B:together"

client = genai.Client(api_key=GOOGLE_API_KEY)

hf_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN", ""),
)


def generate_prompt():
    return """Analyze the person in the image and describe their physical features to build a Lego character avatar.

For each feature, write a short 2-4 word phrase capturing only the most visually distinctive attributes. These descriptions will be used for semantic vector search to find matching Lego pieces.

Return ONLY a valid JSON object with these exact fields:

- hair: combined style, length, and color (e.g. "short black side part")
- eyebrows: combined shape and color (e.g. "thick dark brown")
- eyes: iris color
- nose: nose shape
- beard: combined style and color, or "none" if no facial hair (e.g. "full black beard")
- shirt: color and basic pattern if any
- pants: color

Output only the raw JSON. No markdown, no explanations."""


def _image_to_data_url(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    fmt = image.format or "JPEG"
    mime = f"image/{fmt.lower()}"
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


def _parse_json(text: str) -> dict:
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    return json.loads(text)


def _analyze_face_genai(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes))
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[generate_prompt(), image],
    )
    print(f"DEBUG - Raw GenAI Response:\n{response.text}")
    return _parse_json(response.text)


def _analyze_face_huggingface(image_bytes: bytes) -> dict:
    data_url = _image_to_data_url(image_bytes)
    completion = hf_client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": generate_prompt()},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    text_response = completion.choices[0].message.content
    print(f"DEBUG - Raw HF Response:\n{text_response}")
    return _parse_json(text_response)


def analyze_face(image_bytes: bytes, provider: str = "genai") -> dict:
    try:
        if provider == "huggingface":
            return _analyze_face_huggingface(image_bytes)
        return _analyze_face_genai(image_bytes)
    except Exception as e:
        raise RuntimeError(f"Error during AI analysis: {str(e)}")
```

- [ ] **Step 4: Run all tests — expect PASS**

```bash
pytest tests/test_api.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add facellm_service/model_handler.py tests/test_api.py
git commit -m "feat: add HuggingFace inference provider via Qwen3.5-9B:together"
```

---

### Task 4: Smoke test manually

- [ ] **Step 1: Start the service**

```bash
cd facellm_service
HF_TOKEN=<your_token> python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- [ ] **Step 2: Test HF provider via Swagger**

Open `http://localhost:8000/docs`, call `POST /api/v1/extract-attributes?provider=huggingface` with a real face image. Confirm JSON response with all 7 fields (hair, eyebrows, eyes, nose, beard, shirt, pants).

- [ ] **Step 3: Test GenAI provider still works**

Same endpoint without `?provider=` param. Confirm response is identical in structure.

- [ ] **Step 4: Test invalid provider**

```bash
curl -X POST "http://localhost:8000/api/v1/extract-attributes?provider=openai" \
  -F "image_file=@face.jpg"
```

Expected: `{"detail": "Unknown provider 'openai'. Valid options: ['genai', 'huggingface']"}`
