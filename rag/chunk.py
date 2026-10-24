"""แบ่งเอกสารเป็น chunk พร้อม metadata (เวิร์กช็อป 1)

แต่ละ chunk เป็น dict ที่มีอย่างน้อย:
chunk_id ("<doc_id>#<ลำดับ>"), doc_id, title, section, status, effective_date, text
"""
import re

from rag.ingest import load_docs

MAX_CHARS = 600


def _split_long(text: str, max_chars: int) -> list[str]:
    """ตัดตามบรรทัด ให้แต่ละส่วนไม่เกิน max_chars"""
    parts, cur = [], ""
    for line in text.splitlines():
        while len(line) > max_chars:  # บรรทัดเดียวยาวเกิน
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if cur and len(cur) + 1 + len(line) > max_chars:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def chunk_docs(docs: list[dict] | None = None, max_chars: int = MAX_CHARS) -> list[dict]:
    docs = docs if docs is not None else load_docs()
    chunks = []
    for d in docs:
        sections = re.split(r"(?m)^## +", d["body"])
        # ส่วนแรกคือหัวเรื่อง (#) ก่อนหัวข้อ ## แรก ข้ามได้ถ้าไม่มีเนื้อหา
        for sec in sections[1:] if len(sections) > 1 else sections:
            title, _, body = sec.partition("\n")
            title, body = title.strip(), body.strip()
            if len(sections) == 1:
                title, body = d["title"], sec.strip()
            if not body:
                continue
            for piece in _split_long(body, max_chars):
                if not piece.strip():
                    continue
                chunks.append({
                    "chunk_id": f"{d['doc_id']}#{len([c for c in chunks if c['doc_id'] == d['doc_id']]) + 1}",
                    "doc_id": d["doc_id"],
                    "title": d["title"],
                    "department": d.get("department", ""),
                    "section": title,
                    "status": d["status"],
                    "version": d.get("version", ""),
                    "effective_date": d["effective_date"],
                    "text": piece.strip(),
                })
    return chunks


if __name__ == "__main__":
    cs = chunk_docs()
    print(len(cs), "chunks · ยาวสุด", max(len(c["text"]) for c in cs), "ตัวอักษร")
