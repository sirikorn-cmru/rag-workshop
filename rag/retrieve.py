"""ค้นคืน chunk ที่เกี่ยวข้องกับคำถาม (เวิร์กช็อป 2 และ 4) ทำงานในเครื่องทั้งหมด ไม่เรียก API

python -m rag.retrieve "คำถาม" -k 4 [--method tfidf|hybrid] [--include-superseded] [--json]
"""
import argparse
import json
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.chunk import chunk_docs

# พจนานุกรมคำพ้อง (W4) สร้างจากคำถามที่พลาดในชุด dev เท่านั้น
SYNONYMS = {
    "หมอฟัน": "ทันตกรรม ทำฟัน",
    "รถอีวี": "รถยนต์ไฟฟ้า",
    "พาสเวิร์ด": "รหัสผ่าน",
    "ลืมรหัสผ่าน": "รีเซ็ต รหัสผ่าน Self-Service",
}


def _text(c: dict) -> str:
    return f"{c['title']} {c['section']} {c['text']}"


def _expand(query: str) -> str:
    extra = [v for k, v in SYNONYMS.items() if k in query]
    return " ".join([query, *extra])


def _tokens(text: str) -> list[str]:
    from pythainlp.tokenize import word_tokenize
    return [t for t in word_tokenize(text, engine="newmm") if t.strip()]


def _norm(x: np.ndarray) -> np.ndarray:
    top = x.max()
    return x / top if top > 0 else x


class Retriever:
    def __init__(self, chunks: list[dict] | None = None, method: str = "tfidf"):
        """method="tfidf": TF-IDF ระดับตัวอักษร (W2) · method="hybrid": TF-IDF + BM25 + คำพ้อง (W4)"""
        if method not in ("tfidf", "hybrid"):
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.chunks = chunks if chunks is not None else chunk_docs()
        docs = [_text(c) for c in self.chunks]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vec.fit_transform(docs)
        if method == "hybrid":
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([_tokens(d) for d in docs])

    def _scores(self, query: str) -> np.ndarray:
        if self.method == "tfidf":
            return cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        q = _expand(query)
        dense = cosine_similarity(self.vec.transform([q]), self.matrix)[0]
        sparse = np.asarray(self.bm25.get_scores(_tokens(q)))
        return 0.5 * _norm(dense) + 0.5 * _norm(sparse)

    def search(self, query: str, k: int = 4, include_superseded: bool = True) -> list[dict]:
        """คืน chunk k อันที่คะแนนสูงสุด เรียงจากมากไปน้อย แต่ละอันเพิ่มช่อง "score" """
        scores = self._scores(query)
        out = []
        for i in np.argsort(-scores):
            c = self.chunks[i]
            if not include_superseded and c["status"] != "active":
                continue
            out.append({**c, "score": round(float(scores[i]), 4)})
            if len(out) == k:
                break
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--method", default="hybrid", choices=["tfidf", "hybrid"])
    ap.add_argument("--include-superseded", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = Retriever(method=a.method).search(a.query, a.k, a.include_superseded)
    if a.json:
        json.dump(res, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return
    for r in res:
        print(f"{r['score']:.3f}  {r['chunk_id']:8} [{r['status']}] {r['title']} › {r['section']}")
        print("       " + r["text"].replace("\n", " ")[:120])


if __name__ == "__main__":
    main()
