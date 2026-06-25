import fitz  # PyMuPDF
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("pdf", help="PDF 파일 경로. 예: Lec15.pdf 또는 15.pdf")
parser.add_argument(
    "-o", "--output",
    default=None,
    help="출력 폴더 경로. 생략하면 PDF 파일명 기준으로 폴더 생성"
)
parser.add_argument(
    "--dpi",
    type=int,
    default=200,
    help="이미지 변환 DPI. 기본값: 200"
)

args = parser.parse_args()

# ===== 경로 설정 =====
pdf_path = Path(args.pdf)

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF 파일을 찾을 수 없음: {pdf_path}")

# output 폴더를 직접 지정하지 않으면 PDF 이름에서 확장자 제거한 폴더 사용
# 예: Lec15.pdf -> Lec15/
# 예: 15.pdf -> 15/
if args.output is None:
    output_dir = pdf_path.with_suffix("")
else:
    output_dir = Path(args.output)

output_dir.mkdir(parents=True, exist_ok=True)

# ===== PDF -> PNG 변환 =====
doc = fitz.open(pdf_path)

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=args.dpi)
    output_path = output_dir / f"page_{i + 1:03d}.png"
    pix.save(output_path)
    print(f"[저장 완료] {output_path}")

doc.close()

print()
print(f"총 {len(doc)}개 페이지 변환 완료")
print(f"출력 폴더: {output_dir}")