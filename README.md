# SRIJAN-SRISH

This repository hosts two independent projects:

- 🍅 **[SRISH — Tomato Leaf Expert](#-srish--tomato-leaf-expert-)** — an agentic AI chatbot + computer vision system for tomato crop disease detection.
- 🌱 **[AgroSmart — Precision Agriculture System](#-agrosmart--precision-agriculture-system-)** — a distributed embedded-systems platform for automated soil monitoring and irrigation.

They are unrelated in codebase and purpose — see each project's own section below and its dedicated folder for details.

---

# 🍅 SRISH — Tomato Leaf Expert

An intelligent, agentic chatbot and computer vision system for tomato crop disease detection. SRISH combines a **fine-tuned Llama 3.2 language model**, **YOLOv8 object detection**, and **Gemini-powered intent routing** into a single Streamlit web application — designed specifically for Indian farmers, with full Hinglish language support.

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

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/SRIJAN-SRISH/SRISH.git
cd SRISH
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

## 🧠 Fine-Tuning Details

The Llama 3.2 3B model was fine-tuned to create the **SrishBot** persona using:

- **Method:** LoRA (Low-Rank Adaptation) via the `SFTTrainer` from Hugging Face TRL
- **Platform:** Google Colab (T4 GPU)
- **Dataset:** `Merger.jsonl` — a custom dataset of tomato disease Q&A conversations in both English and Hinglish
- **Base model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Quantization:** Q4_K_M (GGUF format via `llama.cpp`)
- **Deployment:** Served locally via Ollama with a custom `Modelfile` defining the Srish persona

The fine-tuned model handles domain-specific agricultural language and maintains the Srish character (25-year-old botanist, warm Hinglish tone) across conversations.

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

## 🔑 Default Credentials / Config

| Parameter | Value |
|---|---|
| Ollama model name | `SrishBot` |
| Gemini model | `gemini-1.5-flash` |
| LLM temperature | `0.0` (deterministic) |
| Leaf confidence threshold | `65%` |
| Disease detection threshold | `25%` |

## 🛠️ Tech Stack

- **Frontend** — Streamlit, custom CSS animations
- **Vision** — YOLOv8 (Ultralytics), OpenCV, Pillow
- **LLM** — Llama 3.2 3B (LoRA fine-tuned, served via Ollama)
- **Orchestration** — Gemini 1.5 Flash (intent routing, URL reading)
- **RAG** — Custom JSON knowledge base (`tomato_data_clean.json`)
- **Fine-tuning** — HuggingFace TRL / SFTTrainer, LoRA, Google Colab

## 📜 License (SRISH)

Shared for portfolio and academic review. Not licensed for redistribution or commercial use.

---

# 🌱 AgroSmart — Precision Agriculture System

A closed-loop, distributed precision-agriculture system for automated soil monitoring and irrigation, built around three networked embedded nodes communicating over LoRa. AgroSmart continuously samples soil and environmental conditions, computes exact irrigation needs using field-specific agronomic physics, and drives pump/valve hardware safely and automatically — logging every decision to SD card and the cloud.

Originally developed as a Smart India Hackathon (SIH) 2025 submission and continued as a department-level Third Year student project, supervised by three faculty members with a team of six students.

## 🏗️ System Overview

```
┌────────────────────┐        LoRa 433 MHz        ┌──────────────────────────┐        LoRa 433 MHz        ┌────────────────────┐
│   Node A            │ ─────────────────────────► │   Node B                  │ ─────────────────────────► │   Node C            │
│   Field Telemetry   │   MasterSensorRecord       │   Gateway & Decision      │   NodeCCommand              │   Physical Actuator │
│                      │ ◄───────────────────────── │   Engine                  │ ◄───────────────────────── │                      │
└────────────────────┘                             └──────────────────────────┘        NodeCFeedback        └────────────────────┘
      ESP32                                          ESP32 (dual-core, FreeRTOS)                                  ESP32
   Soil + GPS sensing                                Agronomic physics engine                              Pump/valve control
                                                      Cloud sync + SD logging                                Flow + current sensing
                                                              │
                                                              ▼
                                                     ┌──────────────────┐
                                                     │  NodeB Display    │
                                                     │  (Pi Pico, ili9341)│
                                                     └──────────────────┘
```

All inter-node communication uses packed binary C++ structs (`__attribute__((packed))`) wrapped in a common frame:

```
LoRaHeader (8 B: magic 0xAB, protocol version, sender ID, receiver ID, packet type)
  + payload struct (MasterSensorRecord / NodeCCommand / NodeCFeedback / NodeCHeartbeat)
  + CRC16 (2 B, Modbus polynomial)
```

Validation on receipt: size check → magic byte check → receiver-address check → CRC16 check. An `FLAG_EMERGENCY` bit can trigger an immediate `emergencyStop()` on Node C with a safe pressure-bleed sequence.

## 📡 Node A — Field Telemetry (`/Agrosmart_NodeA`)

Deployed in the crop root zone for continuous environmental sampling and spatial tracking.

- **7-in-1 soil parameter polling** over RS-485 Modbus RTU: Volumetric Water Content (VWC), soil temperature, Electrical Conductivity (EC), pH, Nitrogen, Phosphorus, Potassium.
- **Geospatial tracking** via NEO-M8N GPS module (NMEA parsing) — latitude, longitude, fix validity.
- **Diagnostic bitmask health flagging**: soil fault, GPS fault, comm fault (bits `0x01`, `0x02`, `0x04`).
- Packs data into a `MasterSensorRecord`, wraps it in a `LoRaSensorPacket` (magic byte `0xAB`, sender `0x0A`, receiver `0x02`), appends CRC16, and transmits.

**Hardware:** ESP32 · MAX485 TTL-to-RS485 converter · 7-in-1 RS-485 soil probe · NEO-M8N GPS · SX1278 LoRa transceiver (433 MHz, SF9, BW 125 kHz)

**Key files:** `Agrosmart_NodeA.ino`, `config.h`, `pins.h`, `enums.h`, `structs.h`, `crc16.h`, `lora_comms.cpp/.h`, `lora_protocol.h`, `modbus_core.cpp/.h`, `leds.cpp/.h`

## 🧠 Node B — Master Gateway & Agronomic Decision Engine (`/Agrosmart_nodeB`)

Multi-threaded, FreeRTOS-driven gateway fusing Node A telemetry, local environmental sensors, and Node C status into irrigation decisions.

- **Six concurrent FreeRTOS tasks** communicating via thread-safe queues: `lora_task`, `environment_task`, `decision_task`, `sd_task`, `cloud_task`, `health_task` (plus `led_task`, `uart_task`, `mock_task` for testing/display).
- **19-state master state machine** governing the full irrigation decision lifecycle, from `STATE_BOOT` through validation, health checks, environmental correction, decision, safety checks, dose calculation, command dispatch, feedback wait, stabilization, remeasure, logging, cloud sync, down to `STATE_SAFE_IDLE`, `STATE_FAULT`, and `STATE_DEGRADED_MODE`.
- **Agronomic physics engine** (`decision_task.cpp`) computing VPD, Total Available Water (TAW), Readily Available Water (RAW), soil moisture deficit, and gross irrigation volume required (in liters), based on a `FarmProfile` (field capacity, wilting point, MAD, root depth, field area).
- **Multi-tiered safety lock array** (`DecisionReason`): rain lock, daily volume limit, low battery, sensor/LoRa fault, stabilization delay.
- **Cloud sync** (`cloud_task.cpp`) to a Supabase backend, plus **SD card logging** (`sd_task.cpp`) for offline-durable records.
- **Security layer** (`srijan_security.cpp/.h`) for request/data integrity.
- **Secondary display unit** (`NodeB_Display_Pi_Pico/`) — a Raspberry Pi Pico driving an ILI9341 TFT for live status readout.

**Hardware:** ESP32 (dual-core) · SX1278 LoRa transceiver (node address `0x02`) · BMP280 (pressure/temp/humidity) · DS3231 RTC · SEN0575 rain gauge · MicroSD module · 4G/WiFi modem

**Key files:** `Agrosmart_nodeB.ino`, `config.h`, `pins.h`, `enums.h`, `structs.h`, `queues.h`, `tasks.h`, `lora_task.cpp`, `lora_protocol.h`, `decision_task.cpp`, `environment_task.cpp/.h`, `health_task.cpp/.h`, `cloud_task.cpp/.h`, `sd_task.cpp`, `led_task.cpp/.h`, `uart_task.cpp`, `uart_protocol.h`, `config_manager.cpp/.h`, `srijan_security.cpp/.h`, `mock_task.cpp/.h`, `logger.h`, `sync.h`, plus architecture docs (`architecture.md`, `AgroTerm_Architecture.md`, `SYSTEM_ARCHITECTURE_DEEP_DIVE.md`, `NODE_B_EXHAUSTIVE_GUIDE.md`, `nodeC_architecture.md`, `PROJECT_REPORT.md`) and `Cloud_Backend/srijan_supabase.md`.

> ⚠️ **Before publishing:** `config.h` must have real WiFi credentials and Supabase URL/API key replaced with placeholders — see AgroSmart Setup Notes below.

## ⚙️ Node C — Physical Actuator (`/Agrosmart_NodeC`)

Non-blocking `millis()`-based state machine (no RTOS) for deterministic hardware control of the irrigation pump and valve.

- **Water-hammer prevention sequencing**: valve opens 1500 ms before pump start (pre-charge); pump stops 2000 ms before valve closes (bleed).
- **Dual-threshold fault engine** via ACS712-30A current sensor: dry-run/blockage detection (<0.3 A), mechanical jam detection (>5.0 A), 500 ms startup inrush grace window, 3-consecutive-reading trip persistence.
- **Three-tier safety clamping** on runtime/volume, with absolute ceilings (3600 s / 3000 L).
- **Volumetric integration** via YF-S201 flow meter (interrupt-driven pulse counting).
- **Telemetry**: INA219 voltage monitor, 60-second `NodeCHeartbeat` packets.

**Hardware:** ESP32 · SX1278 LoRa transceiver (node address `0x0C`) · 2-channel opto-isolated relay board (active-LOW, JD-VCC jumper removed) · ACS712-30A current sensor · YF-S201 flow meter · INA219 voltage monitor · 1N4007 flyback diodes · LM2596 buck converter · resistor-divider logic-level shifting · 0.1 µF ceramic capacitor low-pass filter on the flow sensor line

**Key files:** `AgroSmart_NodeC.ino`, `config.h`, `pins.h`, `enums.h`, `structs.h`, `lora_protocol.h`, `actuators.cpp/.h`, `sensors.cpp/.h`, `comms.cpp/.h`, `diagnostics.cpp/.h`

## 🔄 Operational Lifecycle (example walkthrough)

1. Node A samples soil at 14% VWC, gets a GPS fix, transmits a `MasterSensorRecord`.
2. Node B validates and queues the packet.
3. `decision_task` computes TAW/RAW/deficit, arriving at a 250.0 L gross dose requirement.
4. Safety gates checked (rain = 0.0 mm, daily volume under limit, battery = 12.4 V, health flag OK) — decision passes.
5. Node B dispatches a `NodeCCommand` (target 250.0 L, max runtime 3000 s, zone 1).
6. Node C clamps values and calls `startPumpCycle()`.
7. Hydraulic sequencing: valve opens → 1500 ms pre-charge delay → pump starts.
8. Live monitoring: current sampled every 100 ms (nominal 2.1 A), flow meter accumulates volume via GPIO 34 interrupts, watchdog reset each loop.
9. Cycle termination at 250.0 L: pump off → 2000 ms bleed delay → valve closes.
10. Node C sends `NodeCFeedback` (delivered volume, status) back to Node B, which updates its daily ledger, logs to SD, and queues the record for cloud sync.

## 🛠️ Tech Stack (AgroSmart)

- **Firmware** — C++ (Arduino framework), FreeRTOS (Node B), bare `millis()` state machine (Node C)
- **MCU** — ESP32 (all three nodes), Raspberry Pi Pico (Node B display, MicroPython)
- **RF** — SX1278 LoRa (433 MHz)
- **Sensing** — RS-485 Modbus RTU (soil), NEO-M8N GPS (NMEA), ACS712/INA219 (current/voltage), YF-S201 (flow), BMP280 (env), SEN0575 (rain)
- **Cloud** — Supabase (REST API)
- **Storage** — MicroSD (local durable logging)

## ⚙️ AgroSmart Setup Notes

Each node folder contains its own `config.h` with hardware pins, timing constants, and (for Node B) network/cloud credentials.

**Before building or publishing Node B**, replace the placeholders in `Agrosmart_nodeB/config.h`:

```cpp
#define WIFI_SSID "your_wifi_ssid_here"
#define WIFI_PASS "your_wifi_password_here"

#define SUPABASE_URL "your_supabase_url_here"
#define SUPABASE_ANON_KEY "your_supabase_anon_key_here"
```

Flash each node's `.ino` via the Arduino IDE / PlatformIO with the ESP32 board package selected. Node B's optional display runs separately on a Pi Pico via MicroPython (`NodeB_Display_Pi_Pico/main.py`).

## 📜 Project Status (AgroSmart)

- **Origin:** Smart India Hackathon (SIH) 2025 entry (did not qualify)
- **Continuation:** Department-level student project, Third Year classification
- **Team:** 6 students
- **Supervision:** 3 faculty supervisors

## 📜 License (AgroSmart)

Shared for portfolio and academic review. Not licensed for redistribution or commercial use.
