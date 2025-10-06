#!/usr/bin/env python3
"""
Simple Advanced Retrieval System
Hệ thống retrieval đơn giản với embedding + rerank
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer
from context_aware_vector_store import ContextAwareVectorStore, SearchResult

@dataclass
class SimpleAdvancedResult:
    """Kết quả tìm kiếm đơn giản"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding_score: float
    rerank_score: float
    combined_score: float

class SimpleAdvancedRetrieval:
    """
    Hệ thống retrieval đơn giản với embedding + rerank
    """
    
    def __init__(self, vector_store: ContextAwareVectorStore):
        self.vector_store = vector_store
        
        # Initialize models
        self._initialize_models()
        
        # Weights
        self.embedding_weight = 0.4
        self.reranker_weight = 0.6
        
        print(f"✅ Simple Advanced Retrieval initialized")
    
    def _initialize_models(self):
        """Khởi tạo các models"""
        try:
            # Initialize embedding model
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding")
            print("✅ Embedding model loaded")
            
            # Initialize rerank model
            print("Loading rerank model...")
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                "hghaan/rerank_model", 
                trust_remote_code=True
            )
            self.rerank_model = AutoModel.from_pretrained(
                "hghaan/rerank_model", 
                trust_remote_code=True, 
                device_map="auto"
            )
            print("✅ Rerank model loaded")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def normalize_embedding_score(self, sim_score: float) -> float:
        """Chuyển cosine similarity (-1,1) thành (0,1)"""
        return (sim_score + 1) / 2
    
    def normalize_reranker_score(self, rerank_score: float) -> float:
        """Chuẩn hóa reranker score thành (0,1)"""
        min_score, max_score = 8.0, 15.0
        clipped_score = np.clip(rerank_score, min_score, max_score)
        return (clipped_score - min_score) / (max_score - min_score)
    
    def get_embedding_scores(self, query: str, documents: List[str]) -> List[float]:
        """Tính embedding scores cho documents"""
        try:
            # Tokenize Vietnamese
            segmented_query = ViTokenizer.tokenize(query)
            segmented_docs = [ViTokenizer.tokenize(doc) for doc in documents]
            
            # Encode
            query_emb = self.embedding_model.encode([segmented_query])
            doc_embs = self.embedding_model.encode(segmented_docs)
            
            # Cosine similarity
            similarities = F.cosine_similarity(
                torch.tensor(query_emb), 
                torch.tensor(doc_embs)
            ).flatten()
            
            # Normalize to (0,1)
            normalized_scores = [self.normalize_embedding_score(sim.item()) for sim in similarities]
            
            return normalized_scores
            
        except Exception as e:
            print(f"❌ Error calculating embedding scores: {e}")
            return [0.0] * len(documents)
    
    def get_reranker_scores(self, query: str, documents: List[str]) -> List[float]:
        """Tính reranker scores cho documents"""
        scores = []
        
        for doc in documents:
            try:
                # Prepare input for rerank model
                inputs = self.rerank_tokenizer(
                    f"query: {query} document: {doc}", 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=512
                ).to(self.rerank_model.device)
                
                with torch.no_grad():
                    outputs = self.rerank_model(**inputs)
                
                # Calculate score using CLS token norm
                if hasattr(outputs, 'last_hidden_state'):
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    score = torch.norm(cls_embedding, dim=-1).item()
                else:
                    score = outputs.last_hidden_state.mean().item()
                
                scores.append(score)
                
            except Exception as e:
                print(f"⚠️ Error calculating rerank score: {e}")
                scores.append(8.0)  # Default minimum score
        
        # Normalize scores
        normalized_scores = [self.normalize_reranker_score(score) for score in scores]
        
        return normalized_scores
    
    def search(self, query: str, top_k: int = 5) -> List[SimpleAdvancedResult]:
        """
        Tìm kiếm với embedding + rerank
        """
        
        print(f"🔍 Advanced Search: '{query}'")
        
        # Step 1: Initial embedding search
        print(f"\n📊 Step 1: Initial Embedding Search")
        initial_results = self.vector_store.search(query, top_k=15)
        
        if not initial_results:
            print("❌ No results found in initial search")
            return []
        
        print(f"   Found {len(initial_results)} initial results")
        
        # Step 2: Prepare documents for reranking
        documents = [result.content for result in initial_results]
        
        # Step 3: Calculate embedding scores
        print(f"\n🧮 Step 2: Calculate Embedding Scores")
        embedding_scores = self.get_embedding_scores(query, documents)
        
        # Step 4: Calculate reranker scores
        print(f"\n🎯 Step 3: Calculate Reranker Scores")
        rerank_scores = self.get_reranker_scores(query, documents)
        
        # Step 5: Calculate combined scores
        print(f"\n⚖️ Step 4: Calculate Combined Scores")
        combined_scores = []
        for emb_score, rerank_score in zip(embedding_scores, rerank_scores):
            combined = self.embedding_weight * emb_score + self.reranker_weight * rerank_score
            combined_scores.append(combined)
        
        # Step 6: Create results
        print(f"\n🔗 Step 5: Create Results")
        results = []
        
        for i, (result, emb_score, rerank_score, combined_score) in enumerate(
            zip(initial_results, embedding_scores, rerank_scores, combined_scores)
        ):
            advanced_result = SimpleAdvancedResult(
                chunk_id=result.chunk_id,
                content=result.content,
                metadata=result.metadata,
                embedding_score=emb_score,
                rerank_score=rerank_score,
                combined_score=combined_score
            )
            results.append(advanced_result)
        
        # Step 7: Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Step 8: Take top results
        final_results = results[:top_k]
        
        # Display results
        print(f"\n✅ FINAL RESULTS (Top {len(final_results)}):")
        for i, result in enumerate(final_results, 1):
            article = result.metadata.get('article', 'N/A')
            clause = result.metadata.get('clause', 'N/A')
            
            print(f"   {i}. {article} - Khoản {clause}")
            print(f"      Combined: {result.combined_score:.3f} | "
                  f"Embedding: {result.embedding_score:.3f} | "
                  f"Rerank: {result.rerank_score:.3f}")
            print(f"      Content: {result.content[:100]}...")
            print()
        
        return final_results

def main():
    """Test Simple Advanced Retrieval System"""
    
    print("🧪 TESTING SIMPLE ADVANCED RETRIEVAL SYSTEM")
    print("=" * 60)
    
    # Load vector store
    vector_store = ContextAwareVectorStore()
    if not vector_store.load('context_aware_vectors.pkl'):
        print("❌ Cannot load vector store. Please run context_aware_vector_store.py first.")
        return
    
    # Initialize simple advanced retrieval system
    retrieval_system = SimpleAdvancedRetrieval(vector_store)
    
    # Test queries
    test_queries = [
        "Trách nhiệm hình sự của người dưới 16 tuổi",
        "Hình phạt tử hình áp dụng như thế nào?",
        "Điều kiện miễn trách nhiệm hình sự"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 TESTING: {query}")
        print(f"{'='*80}")
        
        try:
            results = retrieval_system.search(query, top_k=3)
            
            if results:
                print(f"\n📊 DETAILED ANALYSIS:")
                for i, result in enumerate(results, 1):
                    print(f"\nResult {i}: {result.chunk_id}")
                    print(f"   Article: {result.metadata.get('article', 'N/A')}")
                    print(f"   Clause: {result.metadata.get('clause', 'N/A')}")
                    print(f"   Scores: Combined={result.combined_score:.3f}, "
                          f"Embedding={result.embedding_score:.3f}, "
                          f"Rerank={result.rerank_score:.3f}")
                    print(f"   Content: {result.content[:150]}...")
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue to next query...")

if __name__ == "__main__":
    main()

