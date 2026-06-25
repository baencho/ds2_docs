import fitz  # PyMuPDF
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("pdf", help="PDF 파일 경로. 예: Lec15.pdf 또는 15.pdf")
parser.add_argument("-o", "--output", default=None, help="출력 폴더 경로")
parser.add_argument("--dpi", type=int, default=200, help="이미지 변환 DPI")

args = parser.parse_args()

pdf_path = Path(args.pdf)

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF 파일을 찾을 수 없음: {pdf_path}")

output_dir = Path(args.output) if args.output else pdf_path.with_suffix("")
output_dir.mkdir(parents=True, exist_ok=True)

with fitz.open(pdf_path) as doc:
    page_count = len(doc)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=args.dpi)
        output_path = output_dir / f"page_{i + 1:03d}.png"
        pix.save(output_path)
        print(f"[저장 완료] {output_path}")

print()
print(f"총 {page_count}개 페이지 변환 완료")
print(f"출력 폴더: {output_dir}")