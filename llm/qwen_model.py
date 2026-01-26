import easyocr
import io

reader = None

def get_reader():
    global reader
    if reader is None:
        print("Loading EasyOCR model on CPU...")
        reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR model loaded (CPU mode).")
    return reader

def run_vlm(image_bytes: bytes, prompt_text: str = ""):
    """
    Extract text using EasyOCR. 
    """
    try:
        reader = get_reader()
        
        result = reader.readtext(image_bytes, detail=0)
        
        full_text = "\n".join(result)
        
        return full_text
        
    except Exception as e:
        print(f"Error in EasyOCR: {e}")
        return f"Error extracting text: {e}"
