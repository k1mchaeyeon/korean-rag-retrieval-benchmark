"""문장을 벡터(숫자 묶음)로 바꾸는 임베딩 모델.

- e5: intfloat/multilingual-e5-small. 한국어를 포함한 다국어 모델이고 CPU에서도 돈다.
  e5 계열은 질문 앞에 "query: ", 문서 앞에 "passage: "를 붙여야 성능이 제대로 나온다.
- hash: 글자 2~3개 묶음(n-gram)을 해시해서 세는 단순한 방식. 모델 다운로드 없이 돌아가서
  인터넷이 막힌 환경에서 코드가 도는지 확인할 때와, 'AI 임베딩이 단순 글자 매칭보다 얼마나 나은가'를
  비교하는 기준선으로 쓴다.
"""
import hashlib

import numpy as np


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = text.replace(" ", "")
        for n in (2, 3):
            for i in range(len(t) - n + 1):
                h = int(hashlib.md5(t[i:i + n].encode()).hexdigest()[:8], 16)
                v[h % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class E5Embedder:
    name = "e5"

    def __init__(self, model: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_docs(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([f"passage: {t}" for t in texts], batch_size=32,
                                 normalize_embeddings=True, show_progress_bar=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(f"query: {text}", normalize_embeddings=True)


def get_embedder(name: str):
    return E5Embedder() if name == "e5" else HashEmbedder()
