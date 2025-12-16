# OCR + LLM Demo with NVIDIA NIM

A document processing pipeline that extracts text from PDFs using NVIDIA NV Ingest and performs entity extraction using Nemotron LLM.

## Architecture

```
PDF/Image → NV Ingest (OCR) → Nemotron 49B LLM (Entity Extraction) → Structured Output
```

## Features

- **Document OCR**: Extract text from PDFs and images using NVIDIA NV Ingest
- **Entity Extraction**: Extract key-value pairs using Nemotron 49B v1.5 LLM
- **Web UI**: Streamlit-based interface for easy testing
- **GPU Accelerated**: Runs on NVIDIA A100 GPUs

## Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU (tested on A100 80GB)
- NVIDIA NGC API Key

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd ocr-demo

# Copy environment template and add your NGC API key
cp env.sample .env
# Edit .env and set NGC_API_KEY=your-key-here
```

### 2. Start Services

```bash
# Start Redis and NV Ingest
docker compose up -d redis nv-ingest

# Wait for services to be ready
sleep 30
curl http://localhost:8080/v1/health/ready
```

### 3. Run the UI

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

### 4. Access the App

Open `http://localhost:8501` in your browser.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NGC_API_KEY` | NVIDIA NGC API Key | Required |
| `INGEST_URL` | NV Ingest endpoint | `http://localhost:8080/v1/convert` |
| `LLM_API_URL` | NVIDIA LLM API endpoint | `https://integrate.api.nvidia.com/v1/chat/completions` |

### Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `redis` | 6379 | Message broker for NV Ingest |
| `nv-ingest` | 8080 | Document OCR/extraction |
| `nemotron-parse` | 8002 | Vision-based parsing (optional) |
| `retriever` | 8001 | Page element detection (optional) |

## Usage

1. Upload a PDF or image file
2. The pipeline will:
   - Extract text using NV Ingest OCR
   - Send to Nemotron 49B for entity extraction
   - Display structured key-value pairs

## API Reference

### NV Ingest

```bash
# Submit document
curl -X POST http://localhost:8080/v1/convert \
  -F "files=@document.pdf" \
  -F "job_id=test-123" \
  -F "extract_text=true"

# Check status
curl http://localhost:8080/v1/status/test-123
```

### NVIDIA LLM API

```bash
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
  -H "Authorization: Bearer $NGC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "messages": [{"role": "user", "content": "Extract entities from..."}],
    "max_tokens": 1000
  }'
```

## Project Structure

```
ocr-demo/
├── app.py              # Streamlit application
├── docker-compose.yaml # Docker services configuration
├── requirements.txt    # Python dependencies
├── env.sample          # Environment template
├── .streamlit/         # Streamlit configuration
│   └── config.toml
└── README.md
```

## Troubleshooting

### NV Ingest not responding
```bash
# Check service health
curl http://localhost:8080/v1/health/ready

# Check logs
docker compose logs nv-ingest
```

### Redis connection errors
```bash
# Ensure Redis is running
docker compose up -d redis
docker compose logs redis
```

## License

MIT License

## Acknowledgments

- [NVIDIA NV Ingest](https://github.com/NVIDIA/nv-ingest)
- [NVIDIA NIM](https://build.nvidia.com)
- [Nemotron 49B](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5)
