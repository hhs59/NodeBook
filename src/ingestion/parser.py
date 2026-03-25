import fitz
import os
import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Reads a PDF file and extracts all text page by page.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Oops! Could not find the file: {pdf_path}")
    
    print(f"Extracting text from: {pdf_path}...")
    
    doc = fitz.open(pdf_path)
    full_text =[]
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")
        
        cleaned_page_text = clean_text(raw_text)
        full_text.append(cleaned_page_text)
        
    print(f"Successfully extracted {len(doc)} pages.")
    
    return " ".join(full_text)

if __name__ == "__main__":
    sample_pdf_path = "../../data/raw/sample.pdf" 
    
    try:
        extracted = extract_text_from_pdf(sample_pdf_path)
        print("\n--- FIRST 200 CHARACTERS EXTRACTED ---")
        print(extracted[:200])
    except Exception as e:
        print(e)