import fitz  # PyMuPDF
import os

pdf_path = "./Lec12.pdf"
output_dir = "./12"
os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=200)
    output_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
    pix.save(output_path)

doc.close()