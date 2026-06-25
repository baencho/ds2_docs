from pathlib import Path
import re
from urllib.parse import unquote
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("base", help="예: 15 를 넣으면 15.md 와 15/ 폴더를 사용")
args = parser.parse_args()

base = args.base

# ===== 설정 =====
md_path = Path(f"{base}.md")
image_dir = Path(base)

# ===== 존재 확인 =====
if not md_path.exists():
    raise FileNotFoundError(f"MD 파일을 찾을 수 없음: {md_path}")

if not image_dir.exists():
    raise FileNotFoundError(f"이미지 폴더를 찾을 수 없음: {image_dir}")

# ===== md 파일 읽기 =====
md_text = md_path.read_text(encoding="utf-8")
md_text = unquote(md_text)
md_text = md_text.replace("\\", "/")

# ===== md 안에서 page_XXX.png 파일명 찾기 =====
used_files = set(
    match.lower()
    for match in re.findall(r"page_\d+\.png", md_text, flags=re.IGNORECASE)
)

# ===== 실제 폴더에 있는 png 중 md에 없는 것 찾기 =====
all_pngs = list(image_dir.glob("page_*.png"))

delete_candidates = []

for png_path in sorted(all_pngs):
    if png_path.name.lower() not in used_files:
        delete_candidates.append(png_path)

# ===== 결과 출력 =====
print(f"MD 파일: {md_path}")
print(f"이미지 폴더: {image_dir}")
print(f"MD에서 발견한 사용 이미지 개수: {len(used_files)}")
print(f"폴더 내 이미지 개수: {len(all_pngs)}")
print(f"삭제 대상 파일 개수: {len(delete_candidates)}")
print()

print("=== 삭제 대상 ===")
for path in delete_candidates:
    print(path)

print()

# ===== 실제 삭제 =====
confirm = input("위 삭제 대상 파일들을 실제로 삭제할까요? (y/N): ").strip().lower()

if confirm == "y":
    for path in delete_candidates:
        path.unlink()
        print(f"[삭제 완료] {path}")

    print(f"\n총 {len(delete_candidates)}개 파일 삭제 완료")
else:
    print("\n삭제 취소됨")