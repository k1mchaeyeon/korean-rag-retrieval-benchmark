#!/bin/bash
# 처음부터 끝까지 한 번에: 가상환경 → 설치 → 데이터 → 실험(hash 기준선 + e5)
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python src/data_prep.py
python src/evaluate.py --embedder hash
python src/evaluate.py --embedder e5
echo "끝. results 폴더의 results_e5.md, results_hash.md를 확인하세요."
