# AutoML Platform

A full-stack AutoML platform for image data — supporting **image classification**, **object detection**, and **image segmentation**. Upload your dataset, pick a model, train with real-time monitoring, evaluate results, explain predictions, and chat with an AI assistant that understands your entire project context.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-19-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Dataset Management** — Drag-and-drop ZIP upload with automatic analysis: class distribution, image statistics, quality checks (corrupt files, class imbalance, small images)
- **20 Model Architectures** — ResNet, EfficientNet, MobileNet, VGG, ViT, ConvNeXt, YOLOv8, Faster R-CNN, SSD, DeepLabV3, U-Net, FCN
- **Real-Time Training** — Live loss/accuracy charts via WebSocket, early stopping, mixed-precision (FP16), learning rate schedulers
- **Hyperparameter Optimization** — Optuna-powered search with TPE sampler and median pruning
- **Evaluation Dashboard** — Accuracy, precision, recall, F1, confusion matrix, per-class metrics
- **Explainability** — GradCAM, Integrated Gradients, and GradientSHAP heatmaps via Captum
- **Live Inference** — Upload an image and get top-5 predictions with confidence bars
- **AI Chat Assistant** — OpenAI-compatible LLM with full project context (dataset stats, training history, evaluation results, live metrics), markdown rendering, and stop button
- **GPU/CPU Monitoring** — Real-time system resource usage and VRAM recommendations

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 19, TypeScript, Vite 7, Tailwind CSS 4, Zustand, Recharts, React Markdown |
| **Backend** | FastAPI, Python 3.10+, SQLAlchemy (async SQLite), Pydantic v2 |
| **ML** | PyTorch, Torchvision, Ultralytics (YOLOv8), Segmentation Models PyTorch, Captum, Optuna |
| **Chat** | OpenAI-compatible API (works with OpenAI, HuggingFace, Ollama, vLLM, LMStudio) |
| **Real-Time** | WebSocket streaming for training metrics, chat responses, and HPO progress |

---

## Supported Architectures

### Classification (9 models)

| Model | Parameters | VRAM | Speed |
|-------|-----------|------|-------|
| ResNet-18 | 11.7M | 800MB | Fast |
| ResNet-50 | 25.6M | 1.5GB | Medium |
| EfficientNet-B0 | 5.3M | 600MB | Fast |
| EfficientNet-B3 | 12.2M | 1.2GB | Medium |
| MobileNetV3-Small | 2.5M | 400MB | Very Fast |
| MobileNetV3-Large | 5.5M | 700MB | Fast |
| VGG-16 | 138.4M | 2.5GB | Slow |
| ViT-B/16 | 86.6M | 2.2GB | Medium |
| ConvNeXt-Tiny | 28.6M | 1GB | Medium |

### Object Detection (6 models)

| Model | Parameters | VRAM |
|-------|-----------|------|
| YOLOv8-Nano | 3.2M | 600MB |
| YOLOv8-Small | 11.2M | 1GB |
| YOLOv8-Medium | 25.9M | 1.8GB |
| YOLOv8-Large | 43.7M | 2.8GB |
| Faster R-CNN (ResNet50-FPN) | 41.8M | 2.5GB |
| SSD300 (VGG16) | 35.6M | 2GB |

### Segmentation (5 models)

| Model | Parameters | VRAM |
|-------|-----------|------|
| DeepLabV3 (ResNet50) | 42.0M | 2GB |
| DeepLabV3 (ResNet101) | 61.0M | 3GB |
| U-Net (ResNet34) | 24.4M | 1.5GB |
| U-Net (EfficientNet-B3) | 14.0M | 1.2GB |
| FCN (ResNet50) | 35.3M | 1.8GB |

---

## Dataset Format

### Classification
```
data.zip
  └── class_name_1/
  │     ├── image1.jpg
  │     └── image2.png
  └── class_name_2/
        ├── image3.jpg
        └── image4.png
```
Optionally split into `train/`, `val/`, `test/` subdirectories.

### Object Detection
Supported annotation formats:
- **COCO JSON** — `annotations.json` with `images/` directory
- **YOLO** — `labels/` directory with `.txt` files alongside `images/`
- **Pascal VOC** — `Annotations/` directory with `.xml` files alongside `JPEGImages/`

### Segmentation
```
data.zip
  ├── images/
  │     ├── img1.jpg
  │     └── img2.png
  └── masks/
        ├── img1.png
        └── img2.png
```
Or COCO JSON with polygon annotations.

---

## Getting Started

### Prerequisites

- **Python 3.10+** (3.12 recommended)
- **Node.js 18+**
- **Git**

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/automl-platform.git
cd automl-platform
```

### 2. Backend setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For CPU-only PyTorch (if no NVIDIA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Create .env from template
cp .env.example .env
# Edit .env with your LLM API key (optional, for chat assistant)
```

### 3. Frontend setup

```bash
cd frontend
npm install
```

### 4. Run the application

Start both servers (in separate terminals):

```bash
# Terminal 1 — Backend (port 8000)
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend (port 3000)
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Environment Variables

Create a `backend/.env` file (see `backend/.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key for the LLM provider | (empty) |
| `LLM_MODEL` | Model name to use for chat | `gpt-4o` |

Works with any OpenAI-compatible provider: **OpenAI**, **HuggingFace Inference**, **Ollama** (`http://localhost:11434/v1`), **vLLM**, **LMStudio**, **Azure OpenAI**, etc.

---

## Project Structure

```
automl-platform/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Settings (reads .env)
│   ├── database.py             # SQLAlchemy async engine
│   ├── requirements.txt
│   ├── .env.example
│   ├── api/                    # REST API endpoints
│   │   ├── router.py           # Route aggregator
│   │   ├── projects.py         # CRUD for projects
│   │   ├── datasets.py         # Dataset upload & analysis
│   │   ├── experiments.py      # Experiment management
│   │   ├── training.py         # Training start/stop/status
│   │   ├── evaluation.py       # Model evaluation
│   │   ├── explanations.py     # Explainability generation
│   │   ├── inference.py        # Live prediction
│   │   ├── hpo.py              # Hyperparameter optimization
│   │   └── settings.py         # LLM configuration
│   ├── ws/                     # WebSocket endpoints
│   │   ├── training_ws.py      # Real-time training metrics
│   │   ├── chat_ws.py          # Streaming chat
│   │   └── hpo_ws.py           # HPO trial progress
│   ├── services/               # Business logic
│   ├── ml/                     # ML pipeline
│   │   ├── architectures/      # Model registry
│   │   ├── data/               # Dataset loaders
│   │   ├── trainers/           # Training loops
│   │   ├── evaluators/         # Evaluation metrics
│   │   └── explainer.py        # GradCAM / IG / SHAP
│   ├── models/                 # SQLAlchemy ORM models
│   ├── schemas/                # Pydantic validation schemas
│   └── storage/                # Runtime data (uploads, checkpoints, DB)
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx             # Route definitions
│       ├── pages/              # 8 page components
│       ├── components/         # Layout, chat panel
│       ├── api/                # Axios API client
│       ├── stores/             # Zustand state management
│       └── hooks/              # WebSocket hooks
├── .gitignore
└── README.md
```

---

## API Endpoints

### REST API (`/api`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/projects` | Create a new project |
| GET | `/api/projects` | List all projects |
| GET | `/api/projects/:id` | Get project details |
| DELETE | `/api/projects/:id` | Delete project |
| POST | `/api/projects/:id/dataset` | Upload dataset (ZIP) |
| GET | `/api/projects/:id/dataset` | Get dataset analysis |
| GET | `/api/models?task_type=...` | List architectures |
| GET | `/api/models/recommend` | Get architecture recommendation |
| POST | `/api/projects/:id/experiments` | Create experiment |
| POST | `/api/experiments/:id/train` | Start training |
| POST | `/api/experiments/:id/train/stop` | Stop training |
| POST | `/api/experiments/:id/evaluate` | Run evaluation |
| GET | `/api/experiments/:id/evaluation` | Get evaluation results |
| POST | `/api/experiments/:id/explain` | Generate explanation |
| POST | `/api/experiments/:id/predict` | Run inference |
| POST | `/api/experiments/:id/hpo` | Start HPO |
| GET | `/api/resources` | System resource stats |
| GET | `/api/settings` | Get LLM config |
| PUT | `/api/settings` | Update LLM config |

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `ws://host/ws/training/:experiment_id` | Real-time epoch metrics |
| `ws://host/ws/chat/:project_id` | Streaming chat responses |
| `ws://host/ws/hpo/:experiment_id` | HPO trial progress |

---

## License

MIT
