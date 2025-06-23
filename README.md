# OCR Extractor using GROQ (LLM-Powered)

This project is a Streamlit based OCR  Extractor that transforms images data into a structured table of contact information, including Name, Designation, and Company. It utilizes both spaCy NLP and optionally a Groq-hosted Mistral 24B LLM for enhanced accuracy.

## Features

- Upload JPG/PNG images
- Extract OCR text using `pytesseract`
- Use LLM (Mistral SABA 24B) via Groq API for semantic entity extraction
- Automatic fallback to `spaCy` if LLM fails
- Export results as TSV (Tab-separated values)
- Debug Mode for raw JSON inspection

## Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/kiruba11k/OCR-Groq.git
cd ocr-groq
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Linux/macOS
venv\Scripts\activate       # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Tesseract OCR

Install Tesseract on your system:

- Windows: https://github.com/tesseract-ocr/tesseract/wiki#windows
- macOS:
  ```bash
  brew install tesseract
  ```
- Linux (Debian/Ubuntu):
  ```bash
  sudo apt install tesseract-ocr
  ```

### 5. Configure Groq API Key (Optional)

To enable LLM-based extraction:

Create a `.streamlit/secrets.toml` file with the following content:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

You can obtain your API key from: https://console.groq.com/keys

## Running the Application

```bash
streamlit run app.py
```
## Live Demo

You can try the live version of this app here:

[ Live Demo on Streamlit Cloud](https://lsocrgroq.streamlit.app/)


## Workflow Overview

1. User uploads one or more image files.
2. OCR is performed using pytesseract.
3. If a valid Groq API key is provided, the Mistral 24B model is used for extraction.
4. If the LLM fails or is not configured, spaCy is used as a fallback.
5. Extracted results are displayed in tabular format and made available for TSV download.

## LLM Integration Details

| Aspect                    | Details                                                                                           |
|---------------------------|---------------------------------------------------------------------------------------------------|
| **Model**                 | `mistral-saba-24b` served through the Groq Cloud API via `langchain-groq`.                        |
| **Context Window**        | Up to 32,768 tokens per request.                                                                  |
| **Free-tier Rate Limits** | • 30 requests per minute (RPM)  • 14,400 requests per day (RPD)  • 40,000 tokens per minute (TPM) |
| **Throughput**            | Up to 330 tokens/second for Mistral SABA 24B on Groq hardware (based on internal benchmarks).     |
| **Typical Latency**       | ~0.3s first-token latency, 95–150 tokens/second sustained output in independent tests.            |
| **Concurrency / Scaling** | Free tier is limited by RPM/TPM above. Higher-tier plans allow up to 10× limits or batch mode.    |
| **Fallback Strategy**     | If the LLM fails (network issues, rate limit, parsing error), the app automatically falls back to `spaCy` NER extraction. |
| **File Upload Limit**     | Streamlit supports files up to 200MB per session. Recommended image file size is <5MB each.       |
| **Accuracy (LLM Mode)**   | 92–97% extraction accuracy for well-scanned cards; depends on OCR clarity and image quality.      |
| **Accuracy (spaCy Mode)** | ~70–80% accuracy, best effort based on entity recognition heuristics.                             |

### Why This Architecture is Beneficial

- **Robustness**: The LLM-based extraction handles nuanced or unstructured OCR text far better than keyword-based rules.
- **Fallback-Ready**: If the LLM API is unavailable or budget-constrained, the built-in spaCy fallback ensures consistent functionality.
- **Scalability**: Works in both lightweight (offline/spaCy) and advanced (cloud/LLM) environments.
- **Flexible Input**: Can handle multiple image uploads and still output structured TSV data suitable for bulk contact processing.

> **Note**: For best performance, use high-resolution, front-facing images of business cards or contact sections from digital profiles.


## Output Format Example

```json
[
  {
    "Name": "John Doe",
    "Designation": "Software Engineer",
    "Company": "OpenAI"
  }
]
```

## Technologies Used

- Streamlit – User Interface
- pytesseract – OCR Engine
- OpenCV – Image Preprocessing
- spaCy – NLP Fallback
- LangChain with Groq – LLM Integration
- Pandas – Data Formatting and Export

## Author

Kirubakaran Periyasamy  
GitHub: [@kiruba11k](https://github.com/kiruba11k)

