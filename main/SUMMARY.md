# Legal RAG System - Complete Summary

Tổng quan hệ thống RAG hoàn chỉnh cho tư vấn pháp luật Việt Nam.

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  QUERY PROCESSING (LLM-based)                                │
│  ├─ Normalization: Fix typos, standardize terminology        │
│  └─ Classification: Legal vs Non-legal                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌─────────────┐    ┌─────────────────┐
│  NON-LEGAL  │    │     LEGAL       │
│   Direct    │    │   RAG Pipeline  │
│  Generate   │    │                 │
└─────────────┘    └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   RETRIEVAL     │
                   │  (Multi-stage)  │
                   │  ├─ Embedding   │
                   │  └─ Reranking   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ CONTEXT ASSEMBLY│
                   │  (Reassemble)   │
                   │  ├─ Detect split│
                   │  ├─ Fetch parts │
                   │  └─ Merge full  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   GENERATION    │
                   │   (LLM + ctx)   │
                   │  ├─ Reasoning   │
                   │  └─ Citations   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ POST-PROCESSING │
                   │  ├─ Format      │
                   │  ├─ Confidence  │
                   │  └─ Citations   │
                   └────────┬────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │    ANSWER    │
                    └──────────────┘
```

---

## 📦 **Components**

### **1. RAGchunking.py**
- **Purpose**: Context-aware chunking
- **Strategy**: Clause-based (mỗi khoản = 1 chunk)
- **Features**:
  - Token-aware splitting (max 180 tokens)
  - Parent-child linking
  - Context preservation
- **Output**: 1,474 chunks (86.3% complete, 13.7% split)

### **2. RAGembedding.py**
- **Purpose**: Vector embedding
- **Model**: huyydangg/DEk21_hcmute_embedding
- **Features**:
  - 768-dimensional vectors
  - Vietnamese optimization
  - Persistent storage (4.32 MB)
- **Output**: context_aware_vectors.pkl

### **3. RAGretrieval.py**
- **Purpose**: Advanced retrieval
- **Strategy**: Dual scoring (40% embedding + 60% rerank)
- **Features**:
  - Multi-stage retrieval
  - Rerank model: hghaan/rerank_model
  - 8-bit quantization (50% RAM reduction)
  - Confidence scoring
- **Output**: Top-K chunks with scores

### **4. RAGassistant.py** (NEW!)
- **Purpose**: Complete RAG orchestrator
- **LLM**: Qwen3 1.7B Legal GRPO Phase 2
- **Features**:
  - Query normalization (LLM-based)
  - Query classification (2 labels)
  - Context assembly (full reassembly)
  - Answer generation
  - Interactive CLI
- **Output**: Complete legal answers

---

## 🎯 **Key Features**

### **1. Query Processing**
```python
# Normalization
"hinh phat tu hinh" → "hình phạt tử hình"
"700 trieu" → "700.000.000 đồng"

# Classification
"Điều 40 quy định gì?" → legal
"Thời tiết hôm nay?" → non-legal
```

### **2. Context Assembly** (NEW!)
```python
# Before
điều_3_khoan_1_part_0 → [INCOMPLETE]

# After
điều_3_khoan_1_part_0 + part_1 → [COMPLETE 100%]
```

### **3. Dual Scoring**
```python
combined_score = 0.4 * embedding_score + 0.6 * rerank_score
```

### **4. Confidence Metrics**
```python
🟢 Cao (≥70%)
🟡 Trung bình (60-70%)
🔴 Thấp (<60%)
```

---

## 📊 **Performance**

### **Chunking**
- Total: 1,474 chunks
- Complete: 1,272 (86.3%)
- Split: 202 (13.7%)
- Families: 76

### **Embedding**
- Model size: 768D
- Vector store: 4.32 MB
- Search speed: ~100ms

### **Retrieval**
- Initial candidates: 15
- Final results: 5
- Rerank time: ~2-3s

### **Generation**
- LLM size: 3.44 GB (1.7B params)
- Generation time: ~3-5s
- Total pipeline: ~6-9s

### **Memory Usage**
- Standard: ~4.7 GB
- With 8-bit: ~2.3 GB (50% reduction)
- With 4-bit: ~1.2 GB (75% reduction)

---

## 🚀 **Usage**

### **Quick Start**
```bash
# Setup
cd main
pip install -r ../requirements_rag.txt

# Run
python RAGassistant.py
```

### **Interactive Session**
```
❓ Câu hỏi của bạn: hình phạt tử hình áp dụng như thế nào

🔍 Processing query...
   📝 Normalizing...
   🏷️  Classifying: legal
   🔎 Retrieving: 5 chunks
   🔗 Assembling: 2 split chunks
   🤖 Generating answer...

⚖️  Trả lời:
[Detailed legal answer with citations]

📖 Căn cứ pháp lý:
• Điều 40. Tử hình
• Điều 12. Tuổi chịu trách nhiệm hình sự

📊 Độ tin cậy: Cao (85%)
```

---

## 📚 **Documentation**

### **Core Docs**
- `README_chunking.md` - Chunking strategy
- `README_embedding.md` - Embedding system
- `README_retrieval.md` - Retrieval system
- `README_assistant.md` - Complete RAG system

### **Advanced Docs**
- `README_context_assembly.md` - Context assembly feature
- `README_optimization.md` - Memory optimization

### **Test Scripts**
- `test_context_assembly.py` - Test context assembly

---

## 🔧 **Configuration**

### **Retrieval Config**
```python
RetrievalConfig(
    embedding_weight=0.4,
    reranker_weight=0.6,
    initial_search_k=15,
    final_results_k=5
)
```

### **Generation Config**
```python
# Normalization
temperature=0.1, max_tokens=256

# Classification  
temperature=0.1, max_tokens=32

# Legal Generation
temperature=0.2, max_tokens=1024
```

### **Optimization Config**
```python
# 8-bit quantization
load_in_8bit=True  # 50% RAM reduction

# 4-bit quantization
load_in_4bit=True  # 75% RAM reduction
```

---

## 🎓 **Best Practices**

### **1. Query Formulation**
- ✅ Be specific about legal concepts
- ✅ Use proper legal terminology
- ✅ Mention article numbers if known

### **2. Result Interpretation**
- ✅ Check confidence scores
- ✅ Review citations
- ✅ Verify with legal professionals

### **3. System Optimization**
- ✅ Use 8-bit quantization for balance
- ✅ Monitor RAM usage
- ✅ Adjust top-K based on needs

---

## ⚠️ **Limitations**

### **1. Domain Specific**
- Only for Vietnamese criminal law
- Based on Bộ luật Hình sự 100/2015/QH13
- Not for other legal domains

### **2. Not Legal Advice**
- For educational/research purposes
- Preliminary guidance only
- Always consult legal professionals

### **3. Model Limitations**
- LLM: 1.7B params (not GPT-4 level)
- Context: 8192 tokens max
- May miss complex reasoning

---

## 🔄 **Workflow**

### **Development Workflow**
```bash
1. Prepare data → data/01_VBHN-VPQH_363655.txt
2. Run chunking → python RAGchunking.py
3. Create embeddings → python RAGembedding.py
4. Test retrieval → python RAGretrieval.py
5. Run assistant → python RAGassistant.py
```

### **Production Workflow**
```bash
1. Load pre-built vectors
2. Initialize assistant
3. Serve queries
4. Monitor performance
5. Update as needed
```

---

## 📈 **Future Improvements**

### **Planned Features**
1. **Multi-document support**: Beyond criminal law
2. **Conversation memory**: Track multi-turn dialogues
3. **Hybrid search**: Add keyword search
4. **Fine-tuning**: Custom legal LLM
5. **Web interface**: User-friendly UI
6. **API service**: REST API endpoints

### **Optimization Opportunities**
1. **ONNX Runtime**: 2-3x faster inference
2. **Caching**: Cache common queries
3. **Parallel processing**: Batch queries
4. **Model distillation**: Smaller models

---

## 🙏 **Acknowledgments**

- **Qwen Team**: Base LLM model
- **HuggingFace**: Transformers library
- **TRL Team**: GRPO implementation
- **Model Authors**: Vietnamese legal models

---

## 📞 **Support**

### **Issues**
- Check documentation first
- Review error messages
- Test with simple queries

### **Contributing**
- Report bugs
- Suggest features
- Submit improvements

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-19  
**Status**: Production Ready ✅


