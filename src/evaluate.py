"""검색 성능 실험.

청크 크기 x 검색 방식 조합마다 평가 질문을 던져서,
정답 위치를 포함한 청크를 몇 번째 안에 찾아오는지 잰다.

- Hit@1 / Hit@5: 1등 안에, 5등 안에 정답 청크가 있었던 질문 비율
- MRR@10: 정답 청크 순위의 역수 평균(1등이면 1, 2등이면 0.5 ...). 높을수록 위쪽에서 찾는다는 뜻
- 지연시간: 질문 하나 검색에 걸린 평균 시간(ms)

실행 예)  python src/evaluate.py --embedder e5
          python src/evaluate.py --embedder hash --sizes 300 600   (모델 없이 빠르게 확인)
"""
import argparse
import csv
import json
import time
from pathlib import Path

from chunking import chunk_all
from embedder import get_embedder
from retriever import Retriever

ROOT = Path(__file__).resolve().parent.parent


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in open(ROOT / "data" / name, encoding="utf-8")]


def is_hit(chunk, q) -> bool:
    return chunk.doc_id == q["doc_id"] and chunk.start <= q["start"] and q["end"] <= chunk.end


def run(embedder_name: str, sizes: list[int], methods: list[str], limit: int | None) -> list[dict]:
    docs, qs = load("docs.jsonl"), load("eval.jsonl")
    if limit:
        qs = qs[:limit]
    embedder = get_embedder(embedder_name)
    rows = []
    for size in sizes:
        chunks = chunk_all(docs, size)
        t0 = time.time()
        r = Retriever(chunks, embedder, collection=f"{embedder_name}_{size}")
        build_s = time.time() - t0
        print(f"\n[청크 {size}자] 청크 {len(chunks)}개, 색인 {build_s:.1f}초")
        for m in methods:
            h1 = h5 = mrr = 0.0
            t0 = time.time()
            for q in qs:
                found = r.search(q["question"], m, k=10)
                ranks = [i for i, c in enumerate(found) if is_hit(c, q)]
                if ranks:
                    rank = ranks[0]
                    h1 += rank == 0
                    h5 += rank < 5
                    mrr += 1 / (rank + 1)
            n = len(qs)
            row = {"embedder": embedder_name, "chunk_size": size, "n_chunks": len(chunks), "method": m,
                   "hit@1": round(h1 / n, 3), "hit@5": round(h5 / n, 3), "mrr@10": round(mrr / n, 3),
                   "ms_per_query": round((time.time() - t0) / n * 1000, 1)}
            print(row)
            rows.append(row)
    return rows


def save(rows: list[dict], tag: str) -> None:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"results_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    lines = ["| 임베딩 | 청크 크기 | 청크 수 | 검색 방식 | Hit@1 | Hit@5 | MRR@10 | ms/질문 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['embedder']} | {r['chunk_size']} | {r['n_chunks']} | {r['method']} | {r['hit@1']:.3f} | {r['hit@5']:.3f} | {r['mrr@10']:.3f} | {r['ms_per_query']} |")
    (out / f"results_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: results/results_{tag}.csv, results/results_{tag}.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="e5", choices=["e5", "hash"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[300, 600, 1000])
    ap.add_argument("--methods", nargs="+", default=["bm25", "vector", "hybrid"])
    ap.add_argument("--limit", type=int, default=None, help="질문 수를 줄여 빠르게 돌릴 때")
    a = ap.parse_args()
    save(run(a.embedder, a.sizes, a.methods, a.limit), a.embedder)
