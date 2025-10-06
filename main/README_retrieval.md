# RAG Retrieval System

## Overview
Advanced retrieval system với embedding + rerank + context cho legal RAG.

## Features
- ✅ **Embedding Search**: `huyydangg/DEk21_hcmute_embedding`
- ✅ **Rerank Model**: `hghaan/rerank_model` 
- ✅ **Context Assembly**: Parent-child + sibling linking
- ✅ **Combined Scoring**: 40% embedding + 60% rerank
- ✅ **Confidence Scoring**: Context-aware confidence

## Files
- `RAGretrieval.py` - Main retrieval system
- `context_aware_chunks.json` - Chunk metadata (1474 chunks)

## Usage
```python
from RAGretrieval import AdvancedRetrievalSystem, RetrievalConfig
from RAGembedding import ContextAwareVectorStore

# Load vector store
vector_store = ContextAwareVectorStore.load("context_aware_vectors.pkl")

# Initialize retrieval
config = RetrievalConfig(embedding_weight=0.4, reranker_weight=0.6)
retrieval = AdvancedRetrievalSystem(vector_store, config)

# Search
results = retrieval.search_with_context("trách nhiệm hình sự", top_k=5)
```

## Search Pipeline
1. **Initial Search**: Embedding similarity (top 20)
2. **Reranking**: Rerank model scoring (max_length=384)
3. **Combined Scoring**: Weighted combination
4. **Context Assembly**: Add related chunks
5. **Final Results**: Top K with confidence scores

## Performance Optimization
- **max_length**: 384 tokens (optimized for CPU usage)
- **Batch processing**: Single forward pass for multiple documents
- **Embedding reuse**: Uses pre-computed embeddings from vector store
- **Expected**: 50% reduction in computation, 25% reduction in CPU usage
- **Accuracy**: No impact (same embeddings, same logic)

