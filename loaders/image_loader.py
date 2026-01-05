import pytesseract
from PIL import Image
import io

def ocr_image_from_bytes(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text.strip()

def load_image(path: str) -> str:
    image = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text.strip()
