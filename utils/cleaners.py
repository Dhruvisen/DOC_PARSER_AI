import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # Efficiently remove null bytes
    text = text.replace('\0', '')
    
    # Remove control characters that break JSON
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # Collapse multiple spaces/newlines into one
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()