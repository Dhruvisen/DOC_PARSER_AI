import fitz  
from io import BytesIO
from loaders.image_loader import load_image 

def load_pdf(pdf_bytes: bytes):
    """
    High-fidelity PDF extraction engine.
    - Extracts native text from each page.
    - Automatically detects and OCRs embedded images using Tesseract/PaddleOCR.
    - Cleans null bytes and returns both a full string and page-mapped structure.
    """
    full_text = ""
    pages_text = [] 
    pdf_stream = BytesIO(pdf_bytes)

    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            # Scrub null bytes
            raw_text = page.get_text("text") or ""
            page_text = raw_text.replace("\x00", "")
            
            combined_page_text = page_text.strip()
            images = page.get_images(full=True)

            # OCR embedded images
            for img in images:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    ocr_text = load_image(image_bytes) or ""
                    clean_ocr = ocr_text.replace("\x00", "")
                    combined_page_text += f"\n{clean_ocr}"
                except Exception:
                    continue

            pages_text.append({
                "page_number": page_index + 1,
                "content": combined_page_text 
            })
            
            full_text += f"\nPage {page_index + 1}:\n{combined_page_text}\n"
    
    return full_text.strip(), pages_text