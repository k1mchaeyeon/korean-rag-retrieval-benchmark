"""세 가지 검색 방식.

1) BM25: 질문에 나온 단어가 많이, 드물게 나오는 청크를 높게 친다(키워드 검색).
   한국어는 조사가 붙어 띄어쓰기 단위가 잘 안 맞아서, 글자 2개 묶음(bigram)으로 쪼개 쓴다.
2) 벡터: 질문과 청크를 임베딩해서 Qdrant에서 가장 가까운(코사인 유사도) 청크를 찾는다(뜻 검색).
3) 하이브리드: 두 결과의 '순위'를 합친다(Reciprocal Rank Fusion). 점수 단위가 달라도 합칠 수 있다.
"""
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from chunking import Chunk


def bigrams(text: str) -> list[str]:
    toks = []
    for word in text.split():
        toks.extend(word[i:i + 2] for i in range(max(1, len(word) - 1)))
    return toks


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder, collection: str, qdrant_path: Path | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25Okapi([bigrams(c.text) for c in chunks])

        # Qdrant를 서버 없이 '로컬 모드'로 쓴다. path를 주면 디스크에 저장, 없으면 메모리.
        self.qdrant = QdrantClient(path=str(qdrant_path)) if qdrant_path else QdrantClient(":memory:")
        self.collection = collection
        if self.qdrant.collection_exists(collection):
            self.qdrant.delete_collection(collection)
        self.qdrant.create_collection(collection, vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE))
        vecs = embedder.embed_docs([c.text for c in chunks])
        self.qdrant.upload_points(collection, [
            PointStruct(id=i, vector=v.tolist(), payload={"chunk_id": c.chunk_id, "title": c.title})
            for i, (c, v) in enumerate(zip(chunks, vecs))
        ], batch_size=256)

    def bm25_search(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(bigrams(query))
        return list(scores.argsort()[::-1][:k])

    def vector_search(self, query: str, k: int) -> list[int]:
        qv = self.embedder.embed_query(query)
        hits = self.qdrant.query_points(self.collection, query=qv.tolist(), limit=k).points
        return [h.id for h in hits]

    def hybrid_search(self, query: str, k: int, pool: int = 50, rrf_k: int = 60) -> list[int]:
        score: dict[int, float] = {}
        for ranked in (self.bm25_search(query, pool), self.vector_search(query, pool)):
            for rank, idx in enumerate(ranked):
                score[idx] = score.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(score, key=score.get, reverse=True)[:k]

    def search(self, query: str, method: str, k: int = 5) -> list[Chunk]:
        fn = {"bm25": self.bm25_search, "vector": self.vector_search, "hybrid": self.hybrid_search}[method]
        return [self.chunks[i] for i in fn(query, k)]
