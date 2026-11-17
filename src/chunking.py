"""문서를 검색 단위(청크)로 자른다.

글자 수 기준으로 자르되, 가능하면 문장 끝('. ' 등)에서 끊고,
앞 청크와 조금 겹치게(overlap) 해서 문장이 잘려 정보가 사라지는 걸 줄인다.
각 청크는 원문에서의 시작·끝 위치를 갖고 있어서, 평가 때 정답 위치와 비교할 수 있다.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: int
    title: str
    text: str
    start: int
    end: int


BREAKS = (". ", "다. ", "\n\n", "\n", "? ", "! ")


def split_doc(doc: dict, size: int, overlap: int) -> list[Chunk]:
    text, chunks, pos = doc["text"], [], 0
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            # 너무 앞에서 끊기지 않도록 청크의 뒤쪽 절반 안에서 문장 끝을 찾는다
            window_start = pos + size // 2
            window = text[window_start:end]
            cuts = [window.rfind(b) + len(b) for b in BREAKS if window.rfind(b) >= 0]
            if cuts:
                end = window_start + max(cuts)
        piece = text[pos:end].strip()
        if piece:
            chunks.append(Chunk(f"{doc['doc_id']}-{len(chunks)}", doc["doc_id"], doc["title"], piece, pos, end))
        if end >= len(text):
            break
        pos = max(end - overlap, pos + 1)
    return chunks


def chunk_all(docs: list[dict], size: int, overlap: int | None = None) -> list[Chunk]:
    overlap = size // 5 if overlap is None else overlap
    out = []
    for d in docs:
        out.extend(split_doc(d, size, overlap))
    return out
