import streamlit as st
import pytesseract, cv2, numpy as np, pandas as pd, json, os, re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import spacy
import subprocess
import importlib.util

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]    
USE_LLM      = bool(GROQ_API_KEY)

if USE_LLM:
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,
    )


    prompt_tmpl = ChatPromptTemplate.from_template("""
You are a data-extraction assistant.

From the OCR text below, detect every person mentioned and output
**ONLY** a valid JSON array.  Each item must have:

{{
  "Name": "",
  "Designation": "",
  "Company": ""
}}

Use empty string if a field is missing.
No markdown, no explanations.

OCR text:
{ocr_text}
""")

import spacy, warnings
warnings.filterwarnings("ignore", category=UserWarning)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

DESIG_KWS = {
    "manager","director","engineer","head","officer","ceo","cto","cfo",
    "founder","consultant","lead","president","developer","investor",
    "analyst","architect","scientist","specialist","marketing","sales",
    "product","operations","principal"
}

def spacy_guess(lines):
    """Return ONE row guess (spaCy can't easily split many-people, but good fallback)."""
    joined = " ".join(lines)
    doc    = nlp(joined)
    name = company = designation = ""

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
        if ent.label_ == "ORG" and not company:
            company = ent.text

    for ln in lines:
        if any(k in ln.lower() for k in DESIG_KWS):
            designation = ln
            break

    return [{"Name": name, "Designation": designation, "Company": company}]  

st.set_page_config(" OCR ", layout="centered")
st.title(" OCR EXtractor")
st.caption("Image to TSV.")

debug = st.toggle(" Debug mode to check raw json", value=False)

def extract_lines(img_file):
    arr   = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text  = pytesseract.image_to_string(gray)
    return [ln.strip() for ln in text.split("\n") if ln.strip()]

def llm_extract(lines):
    if not USE_LLM:
        return spacy_guess(lines)

    try:
        prompt  = prompt_tmpl.invoke({"ocr_text": "\n".join(lines)})
        result  = llm.invoke(prompt)
        output  = result.content.strip()

        if debug:
            st.subheader(" Raw LLM output"); st.code(output, language="json")

        output  = re.sub(r"^```json|```$", "", output).strip()
        data    = json.loads(output)

        if isinstance(data, dict):
            data = [data]
        return data
    except Exception as e:
        st.error(f" LLM failed  falling back to spaCy ({e})")
        return spacy_guess(lines)

files = st.file_uploader(" Upload JPG/PNG images", ["jpg","jpeg","png"], accept_multiple_files=True)

if files:
    rows, prog = [], st.progress(0.0, "Processing…")

    for i, f in enumerate(files, 1):
        lines = extract_lines(f)
        if debug:
            st.subheader(f" OCR lines ({f.name})"); st.code("\n".join(lines))

        entries = llm_extract(lines)
        for entry in entries:
            if any(entry.values()):                 
                rows.append({"Image": f.name, **entry})

        prog.progress(i/len(files), f"Done {i}/{len(files)}")

    prog.empty()
    df = pd.DataFrame(rows)

    if df.empty:
        st.warning("No contacts detected.")
    else:
        st.success(" Extraction complete")
        st.dataframe(df, use_container_width=True)

        tsv = df.to_csv(sep="\t", index=False)
        st.text_area(" TSV Output", tsv, height=220)
        st.download_button(" Download TSV", tsv, "contacts.tsv", "text/tab-separated-values")
else:
    st.info(" Upload image(s) to begin.")
