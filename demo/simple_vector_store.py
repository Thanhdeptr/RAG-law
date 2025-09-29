#!/usr/bin/env python3
"""
Simple Vector Store for Legal RAG
In-memory vector storage với cosine similarity search
Tối ưu cho văn bản pháp luật có kích thước nhỏ-vừa
"""

import numpy as np
import pickle
import json
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import ollama


@dataclass
class SearchResult:
    """Kết quả tìm kiếm vector"""
    chunk_id: str
    score: float
    metadata: Dict[str, Any]
    content: str


class SimpleVectorStore:
    """
    Simple in-memory vector store với persistence
    Phù hợp cho datasets nhỏ-vừa (< 10k chunks)
    """
    
    def __init__(self, embedding_model: str = "nomic-embed-text", 
                 ollama_host: str = "http://192.168.10.32:11434"):
        self.embedding_model = embedding_model
        self.ollama_client = ollama.Client(host=ollama_host)
        
        # Storage
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.contents: Dict[str, str] = {}
        
        # Index for faster search (optional optimization)
        self.index_built = False
        
    def embed_text(self, text: str) -> np.ndarray:
        """Create embedding vector for text"""
        try:
            response = self.ollama_client.embeddings(
                model=self.embedding_model, 
                prompt=text
            )
            embedding = np.array(response["embedding"], dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(768, dtype=np.float32)
    
    def add_chunk(self, chunk_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a chunk to the vector store"""
        
        if metadata is None:
            metadata = {}
            
        # Create embedding
        print(f"Creating embedding for chunk: {chunk_id}")
        vector = self.embed_text(content)
        
        # Store
        self.vectors[chunk_id] = vector
        self.contents[chunk_id] = content
        self.metadata[chunk_id] = metadata
        
        # Mark index as outdated
        self.index_built = False
    
    def add_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """Add multiple chunks efficiently"""
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('chunk_id')
            content = chunk.get('content', '')
            
            # Prepare metadata (exclude content to save space)
            metadata = {k: v for k, v in chunk.items() if k not in ['content']}
            
            self.add_chunk(chunk_id, content, metadata)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks...")
        
        print(f"✅ Successfully added {len(chunks)} chunks to vector store")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        similarity = dot_product / (norm1 * norm2)
        
        return float(similarity)
    
    def search(self, query: str, top_k: int = 5, 
               min_score: float = 0.0) -> List[SearchResult]:
        """Search for similar chunks"""
        
        if not self.vectors:
            print("Vector store is empty!")
            return []
        
        # Create query embedding
        query_vector = self.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for chunk_id, vector in self.vectors.items():
            score = self.cosine_similarity(query_vector, vector)
            
            if score >= min_score:
                similarities.append((chunk_id, score))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top K results
        results = []
        for chunk_id, score in similarities[:top_k]:
            result = SearchResult(
                chunk_id=chunk_id,
                score=score,
                metadata=self.metadata.get(chunk_id, {}),
                content=self.contents.get(chunk_id, "")
            )
            results.append(result)
        
        return results
    
    def search_by_metadata(self, filters: Dict[str, Any], 
                          top_k: int = 10) -> List[SearchResult]:
        """Search chunks by metadata filters"""
        
        filtered_results = []
        
        for chunk_id, metadata in self.metadata.items():
            # Check if chunk matches all filters
            matches = True
            for key, value in filters.items():
                if metadata.get(key) != value:
                    matches = False
                    break
            
            if matches:
                result = SearchResult(
                    chunk_id=chunk_id,
                    score=1.0,  # Perfect match for metadata
                    metadata=metadata,
                    content=self.contents.get(chunk_id, "")
                )
                filtered_results.append(result)
        
        return filtered_results[:top_k]
    
    def hybrid_search(self, query: str, metadata_filters: Dict[str, Any] = None,
                     top_k: int = 5) -> List[SearchResult]:
        """Combine semantic search with metadata filtering"""
        
        if metadata_filters:
            # First filter by metadata
            filtered_chunks = set()
            for chunk_id, metadata in self.metadata.items():
                matches = True
                for key, value in metadata_filters.items():
                    if metadata.get(key) != value:
                        matches = False
                        break
                if matches:
                    filtered_chunks.add(chunk_id)
            
            if not filtered_chunks:
                return []
            
            # Then do semantic search on filtered chunks
            query_vector = self.embed_text(query)
            similarities = []
            
            for chunk_id in filtered_chunks:
                vector = self.vectors[chunk_id]
                score = self.cosine_similarity(query_vector, vector)
                similarities.append((chunk_id, score))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for chunk_id, score in similarities[:top_k]:
                result = SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    metadata=self.metadata.get(chunk_id, {}),
                    content=self.contents.get(chunk_id, "")
                )
                results.append(result)
            
            return results
        
        else:
            # Regular semantic search
            return self.search(query, top_k)
    
    def get_chunk(self, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID"""
        
        if chunk_id not in self.vectors:
            return None
        
        return SearchResult(
            chunk_id=chunk_id,
            score=1.0,
            metadata=self.metadata.get(chunk_id, {}),
            content=self.contents.get(chunk_id, "")
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        
        if not self.vectors:
            return {"total_chunks": 0}
        
        # Calculate statistics
        vector_sizes = [vec.nbytes for vec in self.vectors.values()]
        total_size_mb = sum(vector_sizes) / (1024 * 1024)
        
        # Chunk types distribution
        chunk_types = {}
        for metadata in self.metadata.values():
            chunk_type = metadata.get('type', 'unknown')
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            "total_chunks": len(self.vectors),
            "vector_dimension": len(next(iter(self.vectors.values()))),
            "total_size_mb": round(total_size_mb, 2),
            "chunk_types": chunk_types,
            "embedding_model": self.embedding_model
        }
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        
        print(f"Saving vector store to {filepath}...")
        
        data = {
            'vectors': self.vectors,
            'metadata': self.metadata,
            'contents': self.contents,
            'embedding_model': self.embedding_model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save readable stats
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
            self.embedding_model = data.get('embedding_model', self.embedding_model)
            
            print(f"✅ Loaded {len(self.vectors)} chunks successfully")
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False


def main():
    """Test the vector store with legal chunks"""
    
    # Initialize vector store
    vector_store = SimpleVectorStore()
    
    # Load chunks from previous step
    chunks_file = 'legal_chunks.json'
    if not os.path.exists(chunks_file):
        print(f"Chunks file not found: {chunks_file}")
        print("Please run legal_chunker.py first to generate chunks")
        return
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks from {chunks_file}")
    
    # Add chunks to vector store (this will take some time)
    vector_store.add_chunks_batch(chunks)
    
    # Save vector store
    vector_store.save('legal_vectors.pkl')
    
    # Show statistics
    stats = vector_store.get_stats()
    print("\n" + "="*60)
    print("VECTOR STORE STATISTICS:")
    print("="*60)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test some searches
    print("\n" + "="*60)
    print("TESTING SEARCH FUNCTIONALITY:")
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
            print(f"   Title: {result.metadata.get('title', 'N/A')}")
            print(f"   Preview: {result.content[:100]}...")
            print()


if __name__ == "__main__":
    main()
