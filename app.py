import base64
import json
import re
import warnings

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from groq import Groq

warnings.filterwarnings("ignore", category=UserWarning)

# ── API / model setup ──────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

# ── Prompts ────────────────────────────────────────────────────────────────────

_BASE_RULES = """
Rules:
- Each row of the source belongs to exactly ONE person — never mix fields across rows.
- Correct obvious OCR misspellings (e.g. "Manger" → "Manager", "Engincar" → "Engineer").
- Ignore all UI chrome: icons, battery/signal bars, timestamps, "Chat"/"Home"/"Back" labels.
- Do NOT invent or guess a field — if genuinely absent use "".
- Output ONLY a valid JSON array, no markdown fences, no explanation.

Output format (one object per person):
[{"Name": "...", "Designation": "...", "Company": "..."}]"""

VISION_PROMPT = (
    "You are a precise contact-card data extractor.\n"
    "Look at this image and extract every person's: Name, Designation (job title/role), "
    "Company (organisation name).\n" + _BASE_RULES
)

SYSTEM_PROMPT = (
    "You are a precise contact-card data extractor.\n"
    "You receive OCR text reconstructed row-by-row from an image. "
    "Each line in the input corresponds to one visual row in the original image.\n"
    "Extract every person's: Name, Designation (job title/role), Company (organisation name).\n"
    + _BASE_RULES
)

# ── Image preprocessing ────────────────────────────────────────────────────────

def _deskew(gray):
    """Straighten slight skew using moment-based angle correction."""
    try:
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 10:
            return gray
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        if angle < -45:
            angle += 90
        if abs(angle) < 0.5:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def _sharpen(gray):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(gray, -1, kernel)


def _preprocess(img_bgr):
    """Return preprocessed grayscale variants for OCR trials."""
    h, w = img_bgr.shape[:2]
    scale = max(1, int(np.ceil(2000 / max(h, w))))
    if scale > 1:
        img_bgr = cv2.resize(img_bgr, (w * scale, h * scale),
                             interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    deskewed = _deskew(denoised)
    sharpened = _sharpen(deskewed)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(sharpened)

    _, otsu = cv2.threshold(sharpened, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10,
    )
    _, otsu_inv = cv2.threshold(sharpened, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return [contrast, otsu, adaptive, otsu_inv]


# ── Spatial row extraction (Tesseract fallback) ────────────────────────────────

def _words_from_variant(variant, psm, min_conf=45):
    """Return confident word dicts with spatial info from one Tesseract pass."""
    cfg = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(
        variant, config=cfg, output_type=pytesseract.Output.DICT
    )
    words = []
    for i in range(len(data["text"])):
        conf = int(data["conf"][i])
        text = data["text"][i].strip()
        if conf >= min_conf and text:
            words.append({
                "text": text,
                "conf": conf,
                "x": data["left"][i],
                "y": data["top"][i],
                "h": data["height"][i],
            })
    return words


def _cluster_into_rows(words):
    """Group words into rows by Y-coordinate proximity, sort each row by X."""
    if not words:
        return []
    words = sorted(words, key=lambda w: w["y"])
    rows, current = [], [words[0]]
    for w in words[1:]:
        ref = current[-1]
        row_height = max(ref["h"], 8)
        if abs(w["y"] - ref["y"]) <= row_height * 0.65:
            current.append(w)
        else:
            rows.append(current)
            current = [w]
    rows.append(current)
    return [sorted(r, key=lambda w: w["x"]) for r in rows]


def extract_rows_tesseract(img_bgr):
    """
    Run Tesseract across preprocessing variants, pick the best word set,
    reconstruct spatial rows, and return them as text lines.
    """
    variants = _preprocess(img_bgr)
    best_words, best_score = [], -1

    for variant in variants:
        for psm in (6, 4, 11):
            try:
                words = _words_from_variant(variant, psm)
                score = sum(1 for w in words if re.search(r"[A-Za-z]{3,}", w["text"]))
                if score > best_score:
                    best_score, best_words = score, words
            except Exception:
                continue

    rows = _cluster_into_rows(best_words)
    lines = []
    for row in rows:
        line = " ".join(w["text"] for w in row)
        if re.search(r"[A-Za-z]{3,}", line):
            lines.append(line)
    return lines


# ── Vision extraction (primary) ────────────────────────────────────────────────

def _encode_image(raw_bytes, filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return b64, mime


def vision_extract(raw_bytes, filename):
    """Send image directly to vision LLM; returns parsed list or None on failure."""
    try:
        b64, mime = _encode_image(raw_bytes, filename)
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": VISION_PROMPT},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        return _parse_json(response.choices[0].message.content.strip())
    except Exception as e:
        return None  # caller falls through to Tesseract path


# ── Text LLM extraction (Tesseract fallback path) ──────────────────────────────

def _is_noise(line):
    if len(line) <= 2:
        return True
    alnum = sum(c.isalnum() or c.isspace() for c in line)
    if alnum / len(line) < 0.45:
        return True
    if not re.search(r"[A-Za-z]{3,}", line):
        return True
    if re.match(r"^(https?://|www\.)", line):
        return True
    return False


def text_llm_extract(lines):
    cleaned = [ln for ln in lines if not _is_noise(ln)]
    if not cleaned:
        return []
    ocr_text = "\n".join(cleaned)
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"OCR rows (one per line):\n{ocr_text}"},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        return _parse_json(response.choices[0].message.content.strip()) or []
    except Exception as e:
        st.warning(f"Text LLM error: {e}")
        return _spacy_fallback(cleaned)


# ── JSON parsing ───────────────────────────────────────────────────────────────

def _parse_json(text):
    """Extract and parse a JSON array from LLM output robustly."""
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            data = [data]
        return [d for d in data if isinstance(d, dict) and any(
            str(v).strip() for v in d.values()
        )]
    except json.JSONDecodeError:
        return None


# ── spaCy fallback ─────────────────────────────────────────────────────────────

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as _dl
        _dl("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False


def _spacy_fallback(lines):
    if not SPACY_OK:
        return []
    doc = nlp(" ".join(lines))
    name = company = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
        if ent.label_ == "ORG" and not company:
            company = ent.text
    return [{"Name": name, "Designation": "", "Company": company}]


# ── Main extraction pipeline ───────────────────────────────────────────────────

def process_image(raw_bytes, filename, debug=False):
    """
    Try vision LLM first; fall back to Tesseract + text LLM.
    Returns list of dicts with Name/Designation/Company.
    """
    # Stage 1: vision model
    result = vision_extract(raw_bytes, filename)
    if result is not None:
        if debug:
            st.caption("Extraction method: vision LLM")
        return result

    if debug:
        st.caption("Vision LLM unavailable — falling back to Tesseract")

    # Stage 2: Tesseract with spatial rows
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    lines = extract_rows_tesseract(img)

    if debug:
        st.subheader("Spatial OCR rows (Tesseract)")
        st.code("\n".join(lines) if lines else "(none)")

    result = text_llm_extract(lines)

    if debug:
        st.subheader("LLM result")
        st.json(result)

    return result or []


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="OCR Extractor", layout="centered")
st.title("OCR Extractor")
st.caption("Upload screenshots or images to extract Name, Designation and Company.")

debug = st.toggle("Debug mode", value=False)

files = st.file_uploader(
    "Upload JPG/PNG images", ["jpg", "jpeg", "png"], accept_multiple_files=True
)

if files:
    rows = []
    prog = st.progress(0.0, "Processing…")

    for i, f in enumerate(files, 1):
        raw = f.read()

        if debug:
            st.subheader(f"Processing: {f.name}")

        entries = process_image(raw, f.name, debug=debug)

        for entry in entries:
            rows.append({"Image": f.name, **entry})

        prog.progress(i / len(files), f"Done {i}/{len(files)}")

    prog.empty()

    df = pd.DataFrame(rows, columns=["Image", "Name", "Designation", "Company"])

    if df.empty:
        st.warning("No contacts detected. Enable Debug mode to inspect intermediate output.")
    else:
        st.success(f"Extracted {len(df)} contact(s)")
        st.dataframe(df, use_container_width=True)

        tsv = df.to_csv(sep="\t", index=False)
        st.text_area("TSV Output", tsv, height=220)
        st.download_button("Download TSV", tsv, "contacts.tsv", "text/tab-separated-values")
else:
    st.info("Upload one or more images to begin.")
