import streamlit as st
import pytesseract
import cv2
import numpy as np
import pandas as pd
import json
import re
from groq import Groq

# ── API setup ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"

# ── Image preprocessing ────────────────────────────────────────────────────────

def _preprocess(img_bgr):
    """Return a list of preprocessed grayscale images to try."""
    variants = []

    # 1. Upscale if small
    h, w = img_bgr.shape[:2]
    scale = max(1, int(np.ceil(1600 / max(h, w))))
    if scale > 1:
        img_bgr = cv2.resize(img_bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=15)

    # 3. CLAHE contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    variants.append(contrast)

    # 4. Otsu threshold
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # 5. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    variants.append(adaptive)

    # 6. Inverted Otsu (white text on dark background)
    _, otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(otsu_inv)

    return variants


def _run_tesseract(img_gray, psm):
    cfg = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(img_gray, config=cfg)


def extract_lines(img_file):
    """Run Tesseract with multiple preprocessing strategies; pick best result."""
    arr = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    variants = _preprocess(img)
    psm_modes = [6, 11, 4]  # block, sparse, single-column

    best_lines = []
    best_score = -1

    for variant in variants:
        for psm in psm_modes:
            try:
                raw = _run_tesseract(variant, psm)
                lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
                # Score = number of lines that look like real words
                score = sum(1 for ln in lines if re.search(r"[A-Za-z]{3,}", ln))
                if score > best_score:
                    best_score = score
                    best_lines = lines
            except Exception:
                continue

    return best_lines


# ── OCR noise filter ───────────────────────────────────────────────────────────

def _is_noise(line):
    """Return True if the line is likely OCR garbage, not real text."""
    # Too short
    if len(line) <= 2:
        return True
    # Mostly non-alphanumeric (symbols, boxes, noise chars)
    alnum = sum(c.isalnum() or c.isspace() for c in line)
    if alnum / len(line) < 0.5:
        return True
    # No real word (3+ letters)
    if not re.search(r"[A-Za-z]{3,}", line):
        return True
    # Looks like a URL / email / phone noise fragment
    if re.match(r"^(https?://|www\.|[+\d\s\-().]{7,}$)", line):
        return True
    return False


def clean_lines(lines):
    return [ln for ln in lines if not _is_noise(ln)]


# ── LLM extraction ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise contact-card data extractor.

Your job:
- Read OCR text that may contain noise, UI artifacts, icons, random symbols, or fragments.
- Identify every real person's contact information in the text.
- For each person extract ONLY: Name, Designation (job title/role), Company (organisation name).

Rules:
- Ignore ALL OCR noise: garbled characters, single letters, random symbols, app UI labels (like "Chat", "Search", "Home", "Settings", "Back", "Menu", battery/signal icons, timestamps, etc.).
- Do NOT hardcode or guess designations or company names — only extract what is explicitly present.
- If a field is not clearly present, use an empty string "".
- Output ONLY a valid JSON array, no markdown, no explanation, nothing else.

Output format:
[{"Name": "...", "Designation": "...", "Company": "..."}]"""


def llm_extract(lines):
    cleaned = clean_lines(lines)
    if not cleaned:
        return []

    ocr_text = "\n".join(cleaned)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"OCR text:\n{ocr_text}"},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        output = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output, flags=re.DOTALL).strip()

        # Extract JSON array even if there's surrounding text
        match = re.search(r"\[.*\]", output, re.DOTALL)
        if match:
            output = match.group(0)

        data = json.loads(output)
        if isinstance(data, dict):
            data = [data]

        # Filter out fully empty entries
        return [d for d in data if any(v.strip() for v in d.values())]

    except json.JSONDecodeError:
        return _spacy_fallback(cleaned)
    except Exception as e:
        st.warning(f"LLM error: {e}")
        return _spacy_fallback(cleaned)


# ── spaCy fallback ─────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False


def _spacy_fallback(lines):
    if not SPACY_OK:
        return []
    joined = " ".join(lines)
    doc = nlp(joined)
    name = company = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
        if ent.label_ == "ORG" and not company:
            company = ent.text
    return [{"Name": name, "Designation": "", "Company": company}]


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="OCR Extractor", layout="centered")
st.title("OCR Contact Extractor")
st.caption("Upload screenshots or images to extract Name, Designation, and Company.")

debug = st.toggle("Debug mode", value=False)

files = st.file_uploader(
    "Upload JPG/PNG images", ["jpg", "jpeg", "png"], accept_multiple_files=True
)

if files:
    rows = []
    prog = st.progress(0.0, "Processing…")

    for i, f in enumerate(files, 1):
        lines = extract_lines(f)
        cleaned = clean_lines(lines)

        if debug:
            st.subheader(f"Raw OCR lines — {f.name}")
            st.code("\n".join(lines) if lines else "(none)")
            st.subheader(f"After noise filter — {f.name}")
            st.code("\n".join(cleaned) if cleaned else "(none)")

        entries = llm_extract(lines)

        if debug:
            st.subheader(f"LLM result — {f.name}")
            st.json(entries)

        for entry in entries:
            rows.append({"Image": f.name, **entry})

        prog.progress(i / len(files), f"Done {i}/{len(files)}")

    prog.empty()
    df = pd.DataFrame(rows, columns=["Image", "Name", "Designation", "Company"])

    if df.empty:
        st.warning("No contacts detected. Try enabling Debug mode to inspect OCR output.")
    else:
        st.success("Extraction complete")
        st.dataframe(df, use_container_width=True)

        tsv = df.to_csv(sep="\t", index=False)
        st.text_area("TSV Output", tsv, height=220)
        st.download_button("Download TSV", tsv, "contacts.tsv", "text/tab-separated-values")
else:
    st.info("Upload one or more images to begin.")
