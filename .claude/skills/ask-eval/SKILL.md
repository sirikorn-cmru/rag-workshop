---
name: ask-eval
description: ตอบคำถามชุดประเมินทั้งชุดด้วยขั้นตอนของ /ask แล้วให้ Python ตรวจคำตอบ
argument-hint: <dev|test> <จำนวนข้อ>
---
รับ split และจำนวนข้อจาก $ARGUMENTS

1. อ่าน data/eval/questions.csv เฉพาะคอลัมน์ qid และ question ของ split ที่ระบุ ห้ามดูคอลัมน์เฉลย
2. ทำทีละข้อตามขั้นตอนของ /ask แล้วเขียนผลลง out/answers_<split>.jsonl (หนึ่งบรรทัดต่อข้อ: qid, answer)
3. เมื่อครบ รัน `python -m rag.check_answers out/answers_<split>.jsonl`
4. สรุปข้อที่ไม่ผ่านและสาเหตุ (ขั้นค้นคืนหรือขั้นตอบ)
