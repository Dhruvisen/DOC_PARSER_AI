import fitz  
from io import BytesIO
from loaders.image_loader import load_image 

def load_pdf(pdf_bytes: bytes):
    full_text = ""
    pages_text = []
    pdf_stream = BytesIO(pdf_bytes)

    with fitz.open(stream=pdf_stream, filetype="pdf") as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            # Scrub null bytes from text extraction
            raw_text = page.get_text("text") or ""
            page_text = raw_text.replace("\x00", "")
            
            combined_page_text = page_text.strip()
            images = page.get_images(full=True)

            for img in images:
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # OCR processing
                    ocr_text = load_image(image_bytes) or ""
                    # FIX: Handle replacement outside the f-string to avoid SyntaxError
                    clean_ocr = ocr_text.replace("\x00", "")
                    combined_page_text += f"\n{clean_ocr}"
                except Exception:
                    continue

            pages_text.append(combined_page_text)
            full_text += f"\nPage {page_index + 1}:\n{combined_page_text}\n"
    
    print("DEBUG: Extracted PDF text length =", len(full_text))
    return full_text.strip(), pages_text