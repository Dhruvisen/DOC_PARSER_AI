from docx import Document
import io

import re

def extract_text_from_binary(data: bytes) -> str:
    try:
        text = data.decode('latin-1', errors='ignore')
        matches = re.findall(r"[\w\.,\-\?:;'\(\)\/]{4,}", text)
        return "\n".join(matches)
    except Exception:
        return ""

def load_doc(doc_data) -> str:
    source = doc_data
    original_bytes = None
    
    if isinstance(doc_data, bytes):
        source = io.BytesIO(doc_data)
        original_bytes = doc_data
    else:
        try:
            with open(doc_data, "rb") as f:
                original_bytes = f.read()
        except:
            pass

    try:
        if isinstance(source, io.BytesIO):
             source.seek(0)
        doc = Document(source)
        return "\n".join(p.text for p in doc.paragraphs)
        
    except Exception:
        if original_bytes:
            print("Warning: Failed to load as .docx, attempting legacy .doc extraction.")
            extracted = extract_text_from_binary(original_bytes)
            if extracted:
                return extracted
        
        raise ValueError("Could not load document. It might be a complex legacy .doc file or corrupted.")
