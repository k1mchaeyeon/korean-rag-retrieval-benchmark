"""질문을 넣으면 근거 문단을 찾아 보여 주는 화면.

실행: streamlit run app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from chunking import chunk_all  # noqa: E402
from embedder import get_embedder  # noqa: E402
from retriever import Retriever  # noqa: E402

st.set_page_config(page_title="한국어 문서 Q&A 검색", layout="wide")


@st.cache_resource(show_spinner="문서를 색인하는 중…")
def load(embedder_name: str, size: int) -> Retriever:
    docs = [json.loads(l) for l in open(ROOT / "data" / "docs.jsonl", encoding="utf-8")]
    return Retriever(chunk_all(docs, size), get_embedder(embedder_name), collection=f"app_{embedder_name}_{size}")


@st.cache_data
def samples() -> list[dict]:
    return [json.loads(l) for l in open(ROOT / "data" / "eval.jsonl", encoding="utf-8")][:30]


def highlight(text: str, answer: str | None) -> str:
    if answer and answer in text:
        return text.replace(answer, f"**:blue[{answer}]**", 1)
    return text


st.title("한국어 문서 Q&A 검색")
st.caption("KorQuAD 1.0 위키 문서 140편에서 질문의 근거 문단을 찾습니다. 데이터: KorQuAD 1.0 (CC BY-ND 2.0 KR)")

with st.sidebar:
    embedder_name = st.radio("임베딩", ["e5", "hash"], help="e5: 다국어 AI 임베딩 / hash: 글자 묶음 기준선")
    size = st.select_slider("청크 크기(자)", options=[300, 600, 1000], value=600)
    method = st.radio("검색 방식", ["hybrid", "vector", "bm25"], format_func={"hybrid": "하이브리드", "vector": "벡터(뜻)", "bm25": "BM25(키워드)"}.get)
    k = st.slider("가져올 문단 수", 1, 10, 3)

retriever = load(embedder_name, size)

pick = st.selectbox("예시 질문 고르기 (평가 세트에서)", ["직접 입력"] + [q["question"] for q in samples()])
question = st.text_input("질문", value="" if pick == "직접 입력" else pick)
gold = next((q for q in samples() if q["question"] == question), None)

if question:
    results = retriever.search(question, method, k=k)
    if gold:
        hit_rank = next((i + 1 for i, c in enumerate(results) if c.doc_id == gold["doc_id"] and c.start <= gold["start"] and gold["end"] <= c.end), None)
        st.info(f"정답: {gold['answer']} · " + (f"{hit_rank}번째 문단에서 찾음" if hit_rank else f"상위 {k}개 안에 없음"))
    for i, c in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {c.title}**  ·  청크 {c.chunk_id}")
            st.markdown(highlight(c.text, gold["answer"] if gold else None))
