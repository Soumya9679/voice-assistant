# SIH26003 — Voice Assistant ML Pipeline

Multilingual speech-to-intent-to-action ML service for the **SIH26003** cognitive gaming & memory assistance platform.

## Architecture

```text
User Audio
   │
   ▼  asr/ (ARTPARK-IISc/SraVaani-1.0, CUDA + Gradio Space)
Transcribed Text
   │
   ▼  intent/ (paraphrase-multilingual-MiniLM-L12-v2 + cosine prototypes)
Intent + Similarity Score  ──(< 0.47)──▶  UNKNOWN
   │
   ▼  actions/ (allow-listed action registry, threshold >= 0.47)
Safe Application Action
```

## Project Structure

```text
voice_assistant/
├── .gitignore
├── README.md
├── __init__.py
├── test_pipeline.py                   # End-to-end ASR -> Intent -> Action helper
├── actions/
│   ├── __init__.py
│   ├── action_registry.py             # Allow-listed intent -> action mapping & 0.47 threshold
│   └── executor.py                    # Validates intent & returns safe action payload
├── api/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app (GET /, POST /voice/process)
│   └── schemas.py                     # Pydantic request/response models
├── asr/
│   ├── __init__.py
│   ├── app.py                         # Gradio interface for Hugging Face Spaces (@spaces.GPU)
│   ├── asr.py                         # Loads ARTPARK-IISc/SraVaani-1.0 on CUDA
│   └── requirements.txt               # ASR/Gradio dependencies
└── intent/
    ├── __init__.py
    ├── intent.py                      # Semantic prototype classifier (CPU)
    ├── voice_intent_classifier.json   # Classifier threshold config (0.47)
    ├── voice_intent_dataset.csv       # 325-sample labeled utterance dataset
    ├── voice_intent_metadata.json     # Dataset & evaluation metrics
    └── voice_intent_prototypes.json   # Precomputed prototype embeddings
```

## Model & Evaluation Summary

- **ASR Model:** `ARTPARK-IISc/SraVaani-1.0` (`asr/asr.py`, configured for CUDA; uses optional `HF_TOKEN` env var)
- **Intent Model:** `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, CPU)
- **Safety Threshold:** `0.47` cosine similarity (predictions below `0.47` are rejected as `UNKNOWN` and `action: None`)
- **Dataset (`intent/voice_intent_dataset.csv`):** 325 examples across 15 intents (227 train / 49 validation / 49 test)
- **Metrics (`intent/voice_intent_metadata.json`):**
  - Validation Accuracy: `0.9592` (Macro F1: `0.9556`)
  - Test Accuracy: `0.9388` (Macro F1: `0.8942`)

### Allowed Actions (`actions/action_registry.py`)

`START_MEMORY_GAME`, `START_ATTENTION_GAME`, `START_RECALL_GAME`, `STOP_GAME`, `REPEAT`, `NEXT`, `HELP`, `SHOW_SCORE`, `SHOW_HISTORY`, `SET_REMINDER`, `SHOW_REMINDERS`, `GO_HOME`, `GO_BACK`, `CHANGE_LANGUAGE`

*(Unrecognized intents or `UNKNOWN` return `action: null` and `allowed: false`.)*

## Setup & Running

### 1. Intent + FastAPI Service (Text → Intent → Safe Action)

Install dependencies in a virtual environment:

```bash
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

pip install sentence-transformers numpy fastapi uvicorn pydantic
```

Run the intent classifier standalone:

```bash
python intent/intent.py
```

Run the FastAPI service from the **parent directory** of `voice_assistant/` (required because `api/main.py` uses package-relative imports):

```bash
uvicorn voice_assistant.api.main:app --host 127.0.0.1 --port 8000
```

### 2. ASR Service (Audio → Text, GPU / Hugging Face Spaces)

Install ASR dependencies from `asr/requirements.txt` (plus `spaces` when running on HF Spaces) in a CUDA-enabled environment:

```bash
pip install -r asr/requirements.txt
```

Set `HF_TOKEN` if required by Hugging Face Hub, then launch the Gradio interface:

```bash
python asr/app.py
```

## API Endpoints

### `GET /`

```json
{
  "status": "ok",
  "service": "voice-assistant"
}
```

### `POST /voice/process`

**Request:**

```json
{
  "text": "start the memory game"
}
```

**Example Response:**

```json
{
  "text": "start the memory game",
  "intent": "START_MEMORY_GAME",
  "similarity": 0.9377,
  "action": "start_memory_game",
  "allowed": true
}
```
