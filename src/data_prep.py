"""KorQuAD 1.0 dev 세트를 받아 '문서 묶음'과 '평가용 질문'으로 바꾼다.

- 문서: 위키 문서 한 편(같은 제목의 문단들을 이어 붙인 것) = 사내 문서 한 건이라고 생각한다.
- 평가 질문: 질문 + 정답 문자열 + 정답이 문서 안 어디에 있는지(문자 위치).
  검색이 정답 위치를 포함한 조각(청크)을 찾아오면 '맞힘'으로 본다.

데이터 출처: KorQuAD 1.0 (https://korquad.github.io), CC BY-ND 2.0 KR.
원본 파일은 저장소에 올리지 않고 이 스크립트가 내려받는다.
"""
import json
import random
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/korquad/korquad.github.io/master/dataset/KorQuAD_v1.0_dev.json"
DATA = Path(__file__).resolve().parent.parent / "data"
SEP = "\n\n"  # 문단 사이 구분자


def download(path: Path) -> None:
    if path.exists():
        return
    print(f"내려받는 중: {URL}")
    urllib.request.urlretrieve(URL, path)


def build(n_questions: int = 500, seed: int = 42) -> None:
    DATA.mkdir(exist_ok=True)
    raw_path = DATA / "KorQuAD_v1.0_dev.json"
    download(raw_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    docs, questions = [], []
    for doc_id, article in enumerate(raw["data"]):
        text, offset = "", 0
        for para in article["paragraphs"]:
            if text:
                text += SEP
                offset = len(text)
            ctx = para["context"]
            text += ctx
            for qa in para["qas"]:
                ans = qa["answers"][0]
                start = offset + ans["answer_start"]
                questions.append({
                    "qid": qa["id"],
                    "question": qa["question"],
                    "answer": ans["text"],
                    "doc_id": doc_id,
                    "start": start,
                    "end": start + len(ans["text"]),
                })
        docs.append({"doc_id": doc_id, "title": article["title"], "text": text})

    random.Random(seed).shuffle(questions)
    sample = questions[:n_questions]

    with open(DATA / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(DATA / "eval.jsonl", "w", encoding="utf-8") as f:
        for q in sample:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    total_chars = sum(len(d["text"]) for d in docs)
    print(f"문서 {len(docs)}편, 총 {total_chars:,}자, 평가 질문 {len(sample)}개 (전체 {len(questions)}개 중)")


if __name__ == "__main__":
    build()
