import io
import base64
from PIL import Image
from llm.qwen_model import run_vlm

async def ocr_image_from_bytes(image_bytes: bytes, llm=None, markdown: bool = False) -> str:
    """Extract text from an image using the local open source VLM (Qwen2-VL)."""
    try:
        result = run_vlm(image_bytes, "Extract all visible text from this image accurately. Return only the extracted text.")
        
        if markdown:
            return f"\n### OCR Result\n\n```\n{result}\n```\n"
        
        return result
    
    except Exception as e:
        print(f"Error processing image with VLM: {e}")
        return f"Error encountered: {e}"

def load_image(image_source) -> str:
    """
    Load image from path or bytes and run VLM.
    """
    try:
        if isinstance(image_source, bytes):
            image_bytes = image_source
        else:
            with open(image_source, "rb") as f:
                image_bytes = f.read()
        return run_vlm(image_bytes)
    except Exception as e:
        return ""
