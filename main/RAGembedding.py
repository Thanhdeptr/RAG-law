#!/usr/bin/env python3
"""
RAG Embedding System - Context-Aware Vector Store
Vector store với HuggingFace embedding model (huyydangg/DEk21_hcmute_embedding)
"""

import numpy as np
import pickle
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer


# ======================== DATA CLASSES ========================

@dataclass
class SearchResult:
    """Search result with metadata"""
    chunk_id: str
    score: float
    metadata: Dict[str, Any]
    content: str


# ======================== VECTOR STORE CLASS ========================

class ContextAwareVectorStore:
    """Context-aware vector store with HuggingFace embedding"""
    
    def __init__(self, embedding_model_id: str = "huyydangg/DEk21_hcmute_embedding"):
        self.embedding_model_id = embedding_model_id
        self.embedding_model = SentenceTransformer(embedding_model_id)
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.contents: Dict[str, str] = {}
        print(f"✅ Loaded embedding model: {embedding_model_id}")
    
    # ============ EMBEDDING METHODS ============
    
    def embed_text(self, text: str) -> np.ndarray:
        """Create embedding vector for Vietnamese text"""
        try:
            # Tokenize Vietnamese text
            segmented_text = ViTokenizer.tokenize(text)
            
            embedding = self.embedding_model.encode([segmented_text])
            return np.array(embedding[0], dtype=np.float32)
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return np.zeros(768, dtype=np.float32)
    
    # ============ ADD CHUNKS METHODS ============
    
    def add_chunk(self, chunk_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a single chunk to vector store"""
        if metadata is None:
            metadata = {}
        
        print(f"Creating embedding for chunk: {chunk_id}")
        vector = self.embed_text(content)
        self.vectors[chunk_id] = vector
        self.contents[chunk_id] = content
        self.metadata[chunk_id] = metadata
    
    def add_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """Add multiple chunks efficiently"""
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('chunk_id')
            content = chunk.get('content', '')
            metadata = {k: v for k, v in chunk.items() if k not in ['content']}
            
            self.add_chunk(chunk_id, content, metadata)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks...")
        
        print(f"✅ Successfully added {len(chunks)} chunks to vector store")
    
    # ============ SEARCH METHODS ============
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[SearchResult]:
        """Search for similar chunks by query"""
        if not self.vectors:
            print("Vector store is empty!")
            return []
        
        query_vector = self.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for chunk_id, vector in self.vectors.items():
            score = self.cosine_similarity(query_vector, vector)
            if score >= min_score:
                similarities.append((chunk_id, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K results
        results = []
        for chunk_id, score in similarities[:top_k]:
            results.append(SearchResult(
                chunk_id=chunk_id,
                score=score,
                metadata=self.metadata.get(chunk_id, {}),
                content=self.contents.get(chunk_id, "")
            ))
        
        return results
    
    def get_chunk(self, chunk_id: str) -> Optional[SearchResult]:
        """Get specific chunk by ID"""
        if chunk_id not in self.vectors:
            return None
        
        return SearchResult(
            chunk_id=chunk_id,
            score=1.0,
            metadata=self.metadata.get(chunk_id, {}),
            content=self.contents.get(chunk_id, "")
        )
    
    # ============ UTILITY METHODS ============
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        if not self.vectors:
            return {"total_chunks": 0}
        
        vector_sizes = [vec.nbytes for vec in self.vectors.values()]
        total_size_mb = sum(vector_sizes) / (1024 * 1024)
        
        chunk_types = {}
        for metadata in self.metadata.values():
            chunk_type = metadata.get('type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            "total_chunks": len(self.vectors),
            "vector_dimension": len(next(iter(self.vectors.values()))),
            "total_size_mb": round(total_size_mb, 2),
            "chunk_types": chunk_types,
            "embedding_model": self.embedding_model_id
        }
    
    # ============ SAVE/LOAD METHODS ============
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        print(f"Saving vector store to {filepath}...")
        
        data = {
            'vectors': self.vectors,
            'metadata': self.metadata,
            'contents': self.contents,
            'embedding_model': self.embedding_model_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        stats_file = filepath.replace('.pkl', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.get_stats(), f, ensure_ascii=False, indent=2)
        
        print(f"✅ Vector store saved successfully")
        print(f"📊 Stats saved to: {stats_file}")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
        
        print(f"Loading vector store from {filepath}...")
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.vectors = data['vectors']
            self.metadata = data['metadata']
            self.contents = data['contents']
            self.embedding_model_id = data.get('embedding_model', self.embedding_model_id)
            
            print(f"✅ Loaded {len(self.vectors)} chunks successfully")
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False


# ======================== TEST/DEMO FUNCTION ========================

def main():
    """Test and demo embedding system"""
    
    # Initialize
    vector_store = ContextAwareVectorStore()
    chunks_file = 'context_aware_chunks.json'
    
    if not os.path.exists(chunks_file):
        print(f"❌ {chunks_file} not found. Run RAGchunking.py first.")
        return
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"📊 Loaded {len(chunks)} context-aware chunks")
    
    # Create embeddings
    vector_store.add_chunks_batch(chunks)
    vector_store.save('context_aware_vectors.pkl')
    
    # Show statistics
    stats = vector_store.get_stats()
    print("\n" + "="*60)
    print("VECTOR STORE STATISTICS:")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test searches
    print("\n" + "="*60)
    print("SEARCH TEST:")
    print("="*60)
    
    test_queries = [
        "trách nhiệm hình sự của người 15 tuổi",
        "hình phạt tử hình",
        "điều kiện miễn trách nhiệm hình sự"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = vector_store.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result.score:.3f} | {result.metadata.get('article', 'N/A')}")
            print(f"   Title: {result.metadata.get('article_title', 'N/A')}")
            print(f"   Content: {result.content[:80]}...")
            if result.metadata.get('is_split'):
                print(f"   Parent: {result.metadata.get('parent_chunk_id', 'N/A')}")
            print()


if __name__ == "__main__":
    main()
