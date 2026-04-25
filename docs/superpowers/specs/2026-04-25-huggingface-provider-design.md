# HuggingFace Inference Provider — Design Spec

**Date:** 2026-04-25  
**Status:** Approved

## Problem

The Google Gemini free tier is rate-limited to 15 requests/minute and 1,500/day. A second inference provider is needed so traffic can be routed to HuggingFace when GenAI is saturated.

## Approach

Option A: simple conditional branch inside `model_handler.py`. No new files, no abstraction layer.

## API Layer (`main.py`)

Add an optional `provider` query parameter to the existing endpoint:

```
POST /api/v1/extract-attributes?provider=genai        # default
POST /api/v1/extract-attributes?provider=huggingface
```

- Default value: `"genai"` (backwards-compatible, no change for existing callers)
- Invalid provider value → `HTTP 400`
- `analyze_face(image_bytes, provider)` receives the provider string

## Model Handler (`model_handler.py`)

Two clients initialized at module load:

| Client | SDK | Target |
|--------|-----|--------|
| `genai_client` | `google-genai` | Gemini 2.5 Flash (unchanged) |
| `hf_client` | `openai.OpenAI` | `https://router.huggingface.co/v1` |

### HuggingFace path

- Model: `Qwen/Qwen3.5-9B:together`
- Image bytes → PIL detects format → base64 → `data:<mime>;base64,...` data URL (e.g. `image/jpeg`, `image/png`)
- Sent as `image_url` content block alongside the existing prompt text
- Response parsed identically: regex extracts `{...}` JSON block, then `json.loads`

### Shared prompt

`generate_prompt()` is unchanged and shared by both providers.

### Error handling

Both paths raise `RuntimeError` on failure; the existing `HTTPException(500)` in `main.py` catches it.

## Dependencies

- Add `openai` to `requirements.txt`

## Configuration

| Variable | Required | Notes |
|----------|----------|-------|
| `HF_TOKEN` | Only when `provider=huggingface` | Not needed at startup |
| `GOOGLE_API_KEY` | Always (current behavior) | Hardcoded for now |

## Out of Scope

- Automatic fallback on 429 (not requested)
- Moving `GOOGLE_API_KEY` to env (pre-existing)
- Adding a 3rd provider
