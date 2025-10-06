# RAG Embedding System

## Overview
Vector embedding system cho legal RAG sử dụng model `huyydangg/DEk21_hcmute_embedding`.

## Files
- `RAGembedding.py` - Main embedding system
- `context_aware_vectors.pkl` - Vector store (4.32 MB)
- `context_aware_vectors_stats.json` - Statistics

## Stats
- **Chunks**: 1474 unique
- **Dimension**: 768
- **Model**: huyydangg/DEk21_hcmute_embedding
- **Size**: 4.32 MB

## Usage
```python
from RAGembedding import ContextAwareVectorStore

# Load vector store
vector_store = ContextAwareVectorStore.load("context_aware_vectors.pkl")

# Search
results = vector_store.search("trách nhiệm hình sự", top_k=5)
```

## Features
- ✅ Context-aware chunking (1474 chunks)
- ✅ Vietnamese legal document optimized
- ✅ Fast similarity search
- ✅ Persistent storage
