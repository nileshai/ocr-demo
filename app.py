import json
import os
import time
import uuid
from typing import Any, Dict

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# NV Ingest endpoint
INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8080/v1/convert")
STATUS_URL = os.getenv("STATUS_URL", "http://localhost:8080/v1/status")

# NVIDIA API for LLM (since local container needs newer driver)
NVIDIA_API_KEY = os.getenv("NGC_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")

TIMEOUT = int(os.getenv("NIM_TIMEOUT", "300"))


def call_ingest(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Use NV Ingest /v1/convert endpoint to process document."""
    job_id = str(uuid.uuid4())
    
    # Determine document type from filename
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        mime_type = "application/pdf"
    elif ext in [".png", ".jpg", ".jpeg"]:
        mime_type = f"image/{ext.lstrip('.')}"
    else:
        mime_type = "application/pdf"
    
    files = [("files", (filename, file_bytes, mime_type))]
    data = {
        "job_id": job_id,
        "extract_text": "true",
        "extract_images": "true",
        "extract_tables": "true",
    }
    
    resp = requests.post(INGEST_URL, files=files, data=data, timeout=TIMEOUT)
    resp.raise_for_status()
    result = resp.json()
    
    # If async processing, poll for result
    if result.get("status") == "processing":
        task_id = result.get("task_id", job_id)
        return poll_for_result(task_id)
    
    return result


def poll_for_result(task_id: str, max_wait: int = 120) -> Dict[str, Any]:
    """Poll for job completion."""
    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(f"{STATUS_URL}/{task_id}", timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "completed":
                return data
        time.sleep(2)
    raise TimeoutError(f"Job {task_id} did not complete in {max_wait}s")


def call_llm_api(prompt: str, api_key: str) -> str:
    """Call NVIDIA API for LLM inference."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "messages": [
            {"role": "system", "content": "You are an assistant that extracts and formats key-value entities from document text. Be concise and structured. Output only the final result."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
    }
    resp = requests.post(LLM_API_URL, headers=headers, json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # Remove thinking tokens - extract content after </think> tag
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    # Also remove any remaining <think> tags
    import re
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    return content


def extract_text_from_result(result: Any) -> str:
    """Extract text content from nv-ingest result."""
    if isinstance(result, dict):
        # Check for result array (from /v1/convert)
        if "result" in result and isinstance(result["result"], list):
            texts = []
            for item in result["result"]:
                if isinstance(item, dict) and "content" in item:
                    texts.append(item["content"])
            if texts:
                return "\n".join(texts)
        
        # Try common keys
        for key in ["text", "content", "extracted_text", "data"]:
            if key in result:
                val = result[key]
                if isinstance(val, str):
                    return val
                return extract_text_from_result(val)
        
        # Collect all text from nested structures
        texts = []
        for v in result.values():
            t = extract_text_from_result(v)
            if t:
                texts.append(t)
        return "\n".join(texts)
    elif isinstance(result, list):
        texts = []
        for item in result:
            t = extract_text_from_result(item)
            if t:
                texts.append(t)
        return "\n".join(texts)
    elif isinstance(result, str):
        return result
    return ""


def main() -> None:
    st.set_page_config(page_title="OCR + LLM Demo", layout="wide")
    st.title("🔍 OCR + Nemotron Demo")
    st.caption("NV Ingest (OCR) ➜ Nemotron LLM (Entity Extraction)")

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.text_input("NV Ingest URL", value=INGEST_URL, key="ingest_url")
        api_key = st.text_input("NVIDIA API Key", value=NVIDIA_API_KEY, type="password", key="api_key")
        st.markdown("---")
        custom_prompt = st.text_area(
            "LLM Prompt",
            value="Extract all key-value entities from the following invoice/document. Format as a structured markdown table with columns: Field, Value.",
            height=100,
        )
        st.markdown("---")
        st.markdown("### 📊 Service Status")
        
        # Check NV Ingest
        try:
            r = requests.get("http://localhost:8080/v1/health/ready", timeout=5)
            if r.status_code == 200:
                st.success("✅ NV Ingest: Ready")
            else:
                st.error("❌ NV Ingest: Not ready")
        except:
            st.error("❌ NV Ingest: Offline")

    uploaded = st.file_uploader("📄 Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded and st.button("🚀 Run Pipeline", type="primary"):
        try:
            # Step 1: NV Ingest OCR
            st.write("### Step 1: Document Processing")
            with st.spinner("📄 Extracting text with NV Ingest..."):
                ingest_result = call_ingest(uploaded.read(), uploaded.name)
            st.success("✅ Document processed successfully!")

            # Show raw result in expander
            with st.expander("📋 Raw Ingest Result (JSON)"):
                st.json(ingest_result)

            # Extract text
            extracted_text = extract_text_from_result(ingest_result)
            if not extracted_text.strip():
                extracted_text = json.dumps(ingest_result, indent=2)

            st.write("### Step 2: Extracted Text")
            st.text_area("Document Content", value=extracted_text, height=250, disabled=True)

            # Step 2: LLM Entity Extraction
            st.write("### Step 3: Entity Extraction with LLM")
            
            if not api_key:
                st.warning("⚠️ No NVIDIA API key provided. Enter your NGC API key in the sidebar to use LLM.")
            else:
                with st.spinner("🤖 Analyzing with Nemotron LLM..."):
                    try:
                        full_prompt = f"{custom_prompt}\n\nDocument content:\n{extracted_text[:8000]}"
                        llm_response = call_llm_api(full_prompt, api_key)
                        st.success("✅ LLM analysis complete!")
                        st.markdown("#### Extracted Entities")
                        st.markdown(llm_response)
                    except requests.HTTPError as e:
                        st.error(f"LLM API error: {e.response.status_code} - {e.response.text[:500]}")
                    except Exception as e:
                        st.error(f"LLM error: {e}")

        except requests.HTTPError as e:
            st.error(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
        except TimeoutError as e:
            st.error(f"❌ Timeout: {e}")
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown(
        "**Pipeline:** This demo uses **NV Ingest** for document OCR/extraction and **NVIDIA API** for LLM entity extraction."
    )


if __name__ == "__main__":
    main()
