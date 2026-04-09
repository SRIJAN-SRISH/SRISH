# SRISH — Tomato Leaf Expert 🍅

An intelligent, agentic chatbot and computer vision system for tomato crop disease detection. SRISH combines a **fine-tuned Llama 3.2 language model**, **YOLOv8 object detection**, and **Gemini-powered intent routing** into a single Streamlit web application — designed specifically for Indian farmers, with full Hinglish language support.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Leaf Verification | A gatekeeper YOLO classifier confirms the uploaded image is actually a tomato leaf before any analysis |
| 🎯 Disease Detection | A custom-trained YOLOv8 model detects Leaf Miner damage with bounding box annotations |
| 🧠 Gemini Orchestrator | Gemini 1.5 Flash acts as a zero-keyword intent router — classifying every message into disease, general, greeting, URL, or off-topic |
| 💬 Fine-tuned LLM Chat | A Llama 3.2 3B model fine-tuned on tomato disease data (via LoRA/SFTTrainer) answers crop questions as "Srish" |
| 📚 RAG Knowledge Base | 12 tomato diseases with symptoms, causes, and treatments stored in a structured JSON knowledge base |
| 🌐 URL Reading | Users can paste any farming article URL; Gemini extracts the relevant content and answers based on it |
| 🌿 Hinglish Support | Automatically detects Hindi/English mix and responds in natural Hinglish for farmer accessibility |
| ✂️ Image Cropping | Built-in cropper lets users isolate the leaf before analysis |
| 📷 Live Camera | Supports direct camera capture from the browser |

---

## 🏗️ System Architecture

```
User (Streamlit UI)
    │
    ├── Image Upload / Camera
    │       ├── Gatekeeper YOLO → Is this a tomato leaf?
    │       └── Disease YOLO    → Leaf Miner detection + bounding boxes
    │
    └── Chat Message
            ├── Gemini Orchestrator → intent classification (zero keywords)
            │       ├── off_topic   → polite refusal (direct, no LLM)
            │       ├── greeting    → direct reply
            │       ├── url_query   → Gemini reads URL → context → Ollama
            │       ├── tomato_disease + KB match → RAG context → Ollama
            │       ├── tomato_disease (no match)  → clarifying questions
            │       └── tomato_general → Ollama answers directly
            │
            └── Ollama (SrishBot) → Fine-tuned Llama 3.2 3B via LoRA
```

---

## 📁 Project Structure

```
srish/
├── agentic_bot.py          # Main Streamlit application
├── jason_formation.py      # Script that builds tomato_data_clean.json
├── tomato_data_clean.json  # RAG knowledge base (12 tomato diseases)
├── Merger.jsonl            # Fine-tuning dataset (merged conversations)
├── Modelfile               # Ollama Modelfile for SrishBot persona
├── yolo.pt                 # YOLOv8 disease detection weights — NOT committed (see below)
├── keeper.pt               # YOLOv8 gatekeeper weights — NOT committed (see below)
├── llama-3.2-3b-instruct.Q4_K_M.gguf  # Base LLM — NOT committed (1.87 GB)
├── requirements.txt
└── .gitignore
```

> **Why are `.pt` and `.gguf` files missing?**
> Model weight files are excluded from this repository because they exceed GitHub's 100 MB file size limit and contain trained parameters that are large binary files. See **Model Setup** below for how to obtain or reproduce them.

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/srish.git
cd srish
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Google Gemini API key

Create a `.streamlit/secrets.toml` file:

```toml
GOOGLE_API_KEY = "your-gemini-api-key-here"
```

Or set it as an environment variable:

```bash
export GOOGLE_API_KEY="your-key-here"
```

### 4. Model Setup

**Llama base model (required for chat):**

Download the GGUF quantized model from Hugging Face:
```
meta-llama/Llama-3.2-3B-Instruct — Q4_K_M quantization
```
Place it in the project root as `llama-3.2-3b-instruct.Q4_K_M.gguf`.

**Create the fine-tuned Ollama model:**
```bash
ollama create SrishBot -f Modelfile
```

**YOLO weights:**
- `yolo.pt` — custom-trained YOLOv8 for Leaf Miner detection (trained on annotated tomato leaf dataset)
- `keeper.pt` — YOLOv8 classification gatekeeper (Tomato_Leaf vs. non-leaf)

These were trained separately and must be placed in the project root. Contact the author if needed.

### 5. Run the app

```bash
streamlit run agentic_bot.py
```

---

## 🧠 Fine-Tuning Details

The Llama 3.2 3B model was fine-tuned to create the **SrishBot** persona using:

- **Method:** LoRA (Low-Rank Adaptation) via the `SFTTrainer` from Hugging Face TRL
- **Platform:** Google Colab (T4 GPU)
- **Dataset:** `Merger.jsonl` — a custom dataset of tomato disease Q&A conversations in both English and Hinglish
- **Base model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Quantization:** Q4_K_M (GGUF format via `llama.cpp`)
- **Deployment:** Served locally via Ollama with a custom `Modelfile` defining the Srish persona

The fine-tuned model handles domain-specific agricultural language and maintains the Srish character (25-year-old botanist, warm Hinglish tone) across conversations.

---

## 📚 Knowledge Base

`tomato_data_clean.json` contains structured data on **12 tomato diseases**:

| # | Disease |
|---|---|
| 1 | Bacterial Canker |
| 2 | Bacterial Speck |
| 3 | Bacterial Spot |
| 4 | Bacterial Wilt |
| 5 | Early Blight |
| 6 | Fusarium Wilt |
| 7 | Late Blight |
| 8 | Leaf Mold |
| 9 | Septoria Leaf Spot |
| 10 | Target Spot |
| 11 | Tomato Mosaic |
| 12 | Tomato Yellow Leaf Curl |

Each entry includes causative agent, distribution, signs & symptoms, conditions for spread, and cure/management — used as RAG context when Gemini identifies a disease match.

---

## 🔑 Default Credentials / Config

| Parameter | Value |
|---|---|
| Ollama model name | `SrishBot` |
| Gemini model | `gemini-1.5-flash` |
| LLM temperature | `0.0` (deterministic) |
| Leaf confidence threshold | `65%` |
| Disease detection threshold | `25%` |

---

## 🛠️ Tech Stack

- **Frontend** — Streamlit, custom CSS animations
- **Vision** — YOLOv8 (Ultralytics), OpenCV, Pillow
- **LLM** — Llama 3.2 3B (LoRA fine-tuned, served via Ollama)
- **Orchestration** — Gemini 1.5 Flash (intent routing, URL reading)
- **RAG** — Custom JSON knowledge base (`tomato_data_clean.json`)
- **Fine-tuning** — HuggingFace TRL / SFTTrainer, LoRA, Google Colab

---

## 📜 License

Shared for portfolio and academic review. Not licensed for redistribution or commercial use.
