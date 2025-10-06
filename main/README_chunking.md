# RAG Chunking System

Context-aware chunking cho văn bản pháp luật Việt Nam

---

## 🎯 Chunking Strategy

### **1. Clause-Based Chunking**
Mỗi **khoản** = 1 chunk (đơn vị ngữ nghĩa pháp lý hoàn chỉnh)

```
Điều 5. Các nguyên tắc cơ bản
1. Nguyên tắc thứ nhất...    → chunk: điều_5_khoan_1
2. Nguyên tắc thứ hai...     → chunk: điều_5_khoan_2
```

### **2. Token-Aware Splitting**
Nếu khoản > 180 tokens → Tự động chia nhỏ + bảo toàn context

```
điều_3_khoan_1 (250 tokens)
    ↓ Split
    ├─ điều_3_khoan_1_part_0 (120 tokens) ┐
    └─ điều_3_khoan_1_part_1 (130 tokens) ┴─ Có link với nhau
```

**Splitting logic:** Câu → Phẩy/chấm phẩy → Từ

### **3. Context Preservation**
Mỗi sub-chunk giữ 3 loại link:
- **Parent:** Link về chunk gốc
- **Siblings:** Link đến các phần anh em
- **Summary:** Tóm tắt nội dung chunk gốc

---

## 📊 Kết Quả Chunking

**Test:** Bộ luật Hình sự (`01_VBHN-VPQH_363655.txt`)

```
📊 Generated 1474 chunks
   ├─ Complete chunks: 1272 (86.3%)
   └─ Split chunks: 202 (13.7%) in 76 families
   
📈 Token Statistics:
   ├─ Average: 65.8 tokens/chunk
   ├─ Max: 180 tokens
   └─ Over limit: 0 ✅
```

---

## 📋 Chunk Structure

### **Normal Chunk**
```json
{
  "chunk_id": "điều_5_khoan_1",
  "content": "1. Nguyên tắc...",
  "token_count": 120,
  "article": "Điều 5",
  "clause": "1",
  "is_split": false
}
```

### **Split Chunk** (có context links)
```json
{
  "chunk_id": "điều_3_khoan_1_part_0",
  "content": "1. Khoan hồng...",
  "token_count": 120,
  
  "is_split": true,
  "parent_chunk_id": "điều_3_khoan_1",
  "sibling_chunk_ids": ["điều_3_khoan_1_part_1"],
  "context_summary": "1. Khoan hồng đối với..."
}
```

---

## 🚀 Tăng Hiệu Quả RAG

### **1. Semantic Integrity** 
✅ Chunk theo đơn vị ngữ nghĩa pháp lý → Không bị cắt nghĩa giữa chừng

### **2. Token Control**
✅ Max 180 tokens → Phù hợp embedding model → Không bị truncate

### **3. Context Assembly**
✅ Khi retrieve 1 sub-chunk → Tự động lấy parent + siblings → Phục hồi ngữ cảnh đầy đủ

### **4. Hierarchical Metadata**
✅ Track: Phần → Chương → Điều → Khoản → Dễ filter + navigate

**Example Flow:**
```
Query: "Khoan hồng đối với người tự thú"
    ↓
Retrieve: điều_3_khoan_1_part_0 (similarity: 0.89)
    ↓
Auto fetch: 
    ├─ Parent: điều_3_khoan_1 (full context)
    └─ Sibling: điều_3_khoan_1_part_1
    ↓
Return: Complete clause với ngữ cảnh đầy đủ ✅
```

