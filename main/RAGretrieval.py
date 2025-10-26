#!/usr/bin/env python3
"""
Advanced RAG Retrieval System
Hệ thống retrieval tiên tiến cho legal RAG với embedding + rerank + context
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from pyvi import ViTokenizer
from RAGembedding import ContextAwareVectorStore, SearchResult

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AdvancedSearchResult:
    """Kết quả tìm kiếm với thông tin đầy đủ"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    
    # Scores
    embedding_score: float
    rerank_score: float
    combined_score: float
    confidence_score: float
    
    # Context information
    is_split: bool = False
    parent_chunk_id: Optional[str] = None
    sibling_chunk_ids: List[str] = None
    context_summary: Optional[str] = None

@dataclass
class RetrievalConfig:
    """Cấu hình cho hệ thống retrieval"""
    embedding_model_id: str = "huyydangg/DEk21_hcmute_embedding"
    rerank_model_id: str = "hghaan/rerank_model"
    
    # Weights
    embedding_weight: float = 0.4
    reranker_weight: float = 0.6
    
    # Search parameters
    initial_search_k: int = 20
    final_results_k: int = 5
    
    # Rerank parameters
    rerank_min_score: float = 8.0
    rerank_max_score: float = 15.0
    
    # Context parameters
    include_siblings: bool = True
    include_parent: bool = True
    context_boost: float = 0.1
    

# ============================================================================
# MAIN RETRIEVAL SYSTEM
# ============================================================================

class AdvancedRetrievalSystem:
    """
    Hệ thống retrieval tiên tiến với embedding + rerank + context
    """
    
    def __init__(self, vector_store: ContextAwareVectorStore, 
                 config: RetrievalConfig = None,
                 chunks_file: str = "context_aware_chunks.json"):
        
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        self.chunks_metadata = self._load_chunks_metadata(chunks_file)
        self.chunk_families = self._build_chunk_families()
        
        # Initialize models
        self._initialize_models()
        
        print(f"✅ Advanced Retrieval System initialized")
        print(f"   Embedding model: {self.config.embedding_model_id}")
        print(f"   Rerank model: {self.config.rerank_model_id}")
        print(f"   Weights: Embedding {self.config.embedding_weight:.1f} | Rerank {self.config.reranker_weight:.1f}")
    
    def _initialize_models(self):
        """Khởi tạo các models"""
        try:
            # Initialize embedding model
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.config.embedding_model_id)
            print("✅ Embedding model loaded")
            
            # Initialize rerank model
            print("Loading rerank model...")
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                self.config.rerank_model_id, 
                trust_remote_code=True
            )
            
            # Try to load with 4-bit quantization for maximum efficiency
            if torch.cuda.is_available():
                # Set GPU memory limit to 80% for optimal performance
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
                max_memory = int(total_gpu_memory * 0.80)
                print(f"   GPU Memory: {total_gpu_memory / 1024**3:.1f} GB total")
                print(f"   Setting max memory limit: {max_memory / 1024**3:.1f} GB (80%)")
                
                try:
                    print("   Attempting 4-bit quantization for rerank model...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                    self.rerank_model = AutoModel.from_pretrained(
                        self.config.rerank_model_id, 
                        quantization_config=bnb_config,
                        device_map="auto",
                        max_memory={0: max_memory},
                        trust_remote_code=True
                    )
                    print("✅ Rerank model loaded with 4-bit quantization (~75% RAM reduction)")
                except Exception as quant_error:
                    print(f"   ⚠️ 4-bit quantization failed: {quant_error}")
                    print("   Trying 8-bit quantization...")
                    try:
                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True
                        )
                        self.rerank_model = AutoModel.from_pretrained(
                            self.config.rerank_model_id, 
                            quantization_config=bnb_config,
                            device_map="auto",
                            max_memory={0: max_memory},
                            trust_remote_code=True
                        )
                        print("✅ Rerank model loaded with 8-bit quantization (~50% RAM reduction)")
                    except Exception as quant_error2:
                        print(f"   ⚠️ 8-bit quantization failed: {quant_error2}")
                        print("   Loading with standard precision...")
                        self.rerank_model = AutoModel.from_pretrained(
                            self.config.rerank_model_id,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                            device_map="auto",
                            max_memory={0: max_memory}
                        )
                        print("✅ Rerank model loaded with float16 (standard)")
            else:
                print("   CPU mode - loading without quantization...")
                self.rerank_model = AutoModel.from_pretrained(
                    self.config.rerank_model_id,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                print("✅ Rerank model loaded with float32 (CPU optimized)")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def _load_chunks_metadata(self, chunks_file: str) -> Dict[str, Dict]:
        """Load metadata của chunks"""
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            metadata_map = {}
            for chunk in chunks:
                metadata_map[chunk['chunk_id']] = chunk
            
            print(f"✅ Loaded metadata for {len(metadata_map)} chunks")
            return metadata_map
            
        except FileNotFoundError:
            print(f"⚠️ {chunks_file} not found. Context linking disabled.")
            return {}
    
    def _build_chunk_families(self) -> Dict[str, List[str]]:
        """Xây dựng mapping các chunk families"""
        families = {}
        
        for chunk_id, metadata in self.chunks_metadata.items():
            if metadata.get('is_split', False):
                parent_id = metadata.get('parent_chunk_id')
                if parent_id:
                    if parent_id not in families:
                        families[parent_id] = []
                    families[parent_id].append(chunk_id)
        
        print(f"✅ Built {len(families)} chunk families")
        return families

# ============================================================================
# SCORING METHODS
# ============================================================================

    def normalize_embedding_score(self, sim_score: float) -> float:
        """Chuyển cosine similarity (-1,1) thành (0,1)"""
        return (sim_score + 1) / 2
    
    def normalize_reranker_score(self, rerank_score: float) -> float:
        """Chuẩn hóa reranker score thành (0,1)"""
        clipped_score = np.clip(rerank_score, self.config.rerank_min_score, self.config.rerank_max_score)
        return (clipped_score - self.config.rerank_min_score) / (self.config.rerank_max_score - self.config.rerank_min_score)
    
    def get_embedding_scores(self, query: str, initial_results: List[SearchResult]) -> List[float]:
        """Tính embedding scores sử dụng embeddings có sẵn từ vector store"""
        try:
            # Chỉ tạo embedding cho query
            segmented_query = ViTokenizer.tokenize(query)
            query_emb = self.embedding_model.encode([segmented_query])
            
            # Lấy embeddings có sẵn từ vector store
            doc_embs = []
            for result in initial_results:
                if result.chunk_id in self.vector_store.vectors:
                    doc_emb = self.vector_store.vectors[result.chunk_id]  # Direct access
                    # Convert to numpy array if needed
                    if isinstance(doc_emb, list):
                        doc_emb = np.array(doc_emb)
                    doc_embs.append(doc_emb)
                else:
                    # Fallback: tạo embedding mới nếu không tìm thấy
                    segmented_doc = ViTokenizer.tokenize(result.content)
                    doc_emb = self.embedding_model.encode([segmented_doc])[0]
                    doc_embs.append(doc_emb)
            
            # Cosine similarity
            # Convert to numpy array first to avoid warning
            doc_embs_array = np.array(doc_embs)
            similarities = F.cosine_similarity(
                torch.tensor(query_emb), 
                torch.tensor(doc_embs_array)
            ).flatten()
            
            # Normalize to (0,1)
            normalized_scores = [self.normalize_embedding_score(sim.item()) for sim in similarities]
            
            return normalized_scores
            
        except Exception as e:
            print(f"❌ Error calculating embedding scores: {e}")
            return [0.0] * len(initial_results)
    
    def get_reranker_scores(self, query: str, documents: List[str]) -> List[float]:
        """Tính reranker scores cho documents - Batch processing"""
        try:
            # Prepare batch texts
            batch_texts = []
            for doc in documents:
                batch_texts.append(f"query: {query} document: {doc}")
            
            # Batch tokenization
            inputs = self.rerank_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=384
            ).to(self.rerank_model.device)
            
            # Single forward pass for entire batch
            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
            
            # Calculate scores for entire batch
            if hasattr(outputs, 'last_hidden_state'):
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
                scores = torch.norm(cls_embeddings, dim=-1)  # [batch_size]
            else:
                scores = outputs.last_hidden_state.mean(dim=1)
            
            # Normalize scores
            normalized_scores = [self.normalize_reranker_score(score.item()) for score in scores]
            
            return normalized_scores
            
        except Exception as e:
            print(f"⚠️ Batch processing failed: {e}")
            # Fallback to sequential processing
            scores = []
            for doc in documents:
                try:
                    inputs = self.rerank_tokenizer(
                        f"query: {query} document: {doc}", 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True,
                        max_length=384
                    ).to(self.rerank_model.device)
                    
                    with torch.no_grad():
                        outputs = self.rerank_model(**inputs)
                    
                    if hasattr(outputs, 'last_hidden_state'):
                        cls_embedding = outputs.last_hidden_state[:, 0, :]
                        score = torch.norm(cls_embedding, dim=-1).item()
                    else:
                        score = outputs.last_hidden_state.mean().item()
                    
                    scores.append(score)
                    
                except Exception as e2:
                    print(f"⚠️ Error calculating rerank score for document: {e2}")
                    scores.append(self.config.rerank_min_score)
            
            # Normalize scores
            normalized_scores = [self.normalize_reranker_score(score) for score in scores]
            return normalized_scores
    
    def calculate_combined_scores(self, embedding_scores: List[float], 
                                rerank_scores: List[float]) -> List[float]:
        """Tính combined scores"""
        combined_scores = []
        
        for emb_score, rerank_score in zip(embedding_scores, rerank_scores):
            combined = (self.config.embedding_weight * emb_score + 
                       self.config.reranker_weight * rerank_score)
            combined_scores.append(combined)
        
        return combined_scores

# ============================================================================
# CONTEXT METHODS
# ============================================================================

    def get_related_chunks(self, chunk_id: str) -> List[SearchResult]:
        """Lấy các chunks liên quan"""
        related_chunks = []
        
        if chunk_id not in self.chunks_metadata:
            return related_chunks
        
        metadata = self.chunks_metadata[chunk_id]
        
        # Lấy sibling chunks
        if self.config.include_siblings and metadata.get('is_split', False):
            sibling_ids = metadata.get('sibling_chunk_ids', [])
            for sibling_id in sibling_ids:
                sibling_chunk = self.vector_store.get_chunk(sibling_id)
                if sibling_chunk:
                    related_chunks.append(sibling_chunk)
        
        # Lấy parent chunk
        if self.config.include_parent and metadata.get('is_split', False):
            parent_id = metadata.get('parent_chunk_id')
            if parent_id:
                parent_chunk = self.vector_store.get_chunk(parent_id)
                if parent_chunk:
                    related_chunks.append(parent_chunk)
        
        return related_chunks
    
    def create_advanced_search_result(self, chunk_id: str, content: str, 
                                    metadata: Dict[str, Any],
                                    embedding_score: float, 
                                    rerank_score: float,
                                    combined_score: float) -> AdvancedSearchResult:
        """Tạo AdvancedSearchResult với thông tin đầy đủ"""
        
        # Lấy thông tin context
        chunk_metadata = self.chunks_metadata.get(chunk_id, {})
        
        # Tính confidence score
        confidence_score = self._calculate_confidence_score(
            combined_score, chunk_metadata
        )
        
        return AdvancedSearchResult(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
            embedding_score=embedding_score,
            rerank_score=rerank_score,
            combined_score=combined_score,
            confidence_score=confidence_score,
            is_split=chunk_metadata.get('is_split', False),
            parent_chunk_id=chunk_metadata.get('parent_chunk_id'),
            sibling_chunk_ids=chunk_metadata.get('sibling_chunk_ids', []),
            context_summary=chunk_metadata.get('context_summary')
        )
    
    def _calculate_confidence_score(self, combined_score: float, 
                                  chunk_metadata: Dict[str, Any]) -> float:
        """Tính confidence score dựa trên combined score và context"""
        
        # Base confidence từ combined score
        base_confidence = combined_score
        
        # Context bonus
        context_bonus = 0.0
        if chunk_metadata.get('is_split', False):
            # Bonus cho chunks có context đầy đủ
            context_bonus = self.config.context_boost
        else:
            # Bonus cho chunks không bị split (ngữ cảnh nguyên vẹn)
            context_bonus = self.config.context_boost * 1.5
        
        # Final confidence
        final_confidence = min(1.0, base_confidence + context_bonus)
        
        return final_confidence

# ============================================================================
# MAIN SEARCH METHODS
# ============================================================================

    def search(self, query: str, top_k: Optional[int] = None) -> List[AdvancedSearchResult]:
        """
        Tìm kiếm với embedding + rerank
        """
        
        if top_k is None:
            top_k = self.config.final_results_k
        
        print(f"🔍 Advanced Search: '{query}'")
        print(f"   Initial search: {self.config.initial_search_k} chunks")
        print(f"   Final results: {top_k} chunks")
        
        # Step 1: Initial embedding search
        print(f"\n📊 Step 1: Initial Embedding Search")
        initial_results = self.vector_store.search(
            query, 
            top_k=self.config.initial_search_k
        )
        
        if not initial_results:
            print("❌ No results found in initial search")
            return []
        
        print(f"   Found {len(initial_results)} initial results")
        
        # Step 2: Prepare documents for reranking
        documents = [result.content for result in initial_results]
        
        # Step 3: Calculate embedding scores (sử dụng embeddings có sẵn)
        print(f"\n🧮 Step 2: Calculate Embedding Scores")
        embedding_scores = self.get_embedding_scores(query, initial_results)
        
        # Step 4: Calculate reranker scores
        print(f"\n🎯 Step 3: Calculate Reranker Scores")
        rerank_scores = self.get_reranker_scores(query, documents)
        
        # Step 5: Calculate combined scores
        print(f"\n⚖️ Step 4: Calculate Combined Scores")
        combined_scores = self.calculate_combined_scores(embedding_scores, rerank_scores)
        
        # Step 6: Create advanced search results
        print(f"\n🔗 Step 5: Create Advanced Results")
        advanced_results = []
        
        for i, (result, emb_score, rerank_score, combined_score) in enumerate(
            zip(initial_results, embedding_scores, rerank_scores, combined_scores)
        ):
            advanced_result = self.create_advanced_search_result(
                result.chunk_id,
                result.content,
                result.metadata,
                emb_score,
                rerank_score,
                combined_score
            )
            advanced_results.append(advanced_result)
        
        # Step 7: Sort by combined score
        advanced_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Step 8: Take top results
        final_results = advanced_results[:top_k]
        
        # Display results
        print(f"\n✅ FINAL RESULTS (Top {len(final_results)}):")
        for i, result in enumerate(final_results, 1):
            article = result.metadata.get('article', 'N/A')
            clause = result.metadata.get('clause', 'N/A')
            
            print(f"   {i}. {article} - Khoản {clause}")
            print(f"      Combined: {result.combined_score:.3f} | "
                  f"Embedding: {result.embedding_score:.3f} | "
                  f"Rerank: {result.rerank_score:.3f}")
            print(f"      Confidence: {result.confidence_score:.3f}")
            print(f"      Content: {result.content[:100]}...")
            
            if result.is_split:
                print(f"      Context: Split chunk with {len(result.sibling_chunk_ids)} siblings")
            print()
        
        return final_results
    
    def search_with_context(self, query: str, top_k: Optional[int] = None) -> List[AdvancedSearchResult]:
        """
        Tìm kiếm với context assembly
        """
        
        # Get initial results
        results = self.search(query, top_k)
        
        if not results:
            return results
        
        # Add context information
        print(f"\n🔗 Adding Context Information:")
        
        for result in results:
            related_chunks = self.get_related_chunks(result.chunk_id)
            
            if related_chunks:
                print(f"   {result.chunk_id}: Found {len(related_chunks)} related chunks")
                
                # Boost confidence for chunks with context
                result.confidence_score = min(1.0, result.confidence_score + 0.05)
        
        return results

    def _assemble_full_context(self, chunk_result) -> str:
        """Assemble full context for split chunks by fetching and merging siblings"""
        
        metadata = chunk_result.metadata
        
        # Check if this is a split chunk
        if not metadata.get('is_split', False):
            # Not split, return as-is
            return chunk_result.content
        
        # This is a split chunk - need to reassemble
        parent_chunk_id = metadata.get('parent_chunk_id')
        sibling_ids = metadata.get('sibling_chunk_ids', [])
        current_chunk_id = chunk_result.chunk_id
        
        if not parent_chunk_id or not sibling_ids:
            # Missing info, return as-is
            return chunk_result.content
        
        print(f"      🔗 Assembling split chunk: {current_chunk_id}")
        print(f"         Parent: {parent_chunk_id}")
        print(f"         Siblings: {len(sibling_ids)} parts")
        
        # Collect all parts (current + siblings)
        all_parts = []
        
        # Add current chunk
        all_parts.append({
            'chunk_id': current_chunk_id,
            'content': chunk_result.content,
            'split_index': metadata.get('split_index', 0)
        })
        
        # Fetch sibling chunks from vector store
        for sibling_id in sibling_ids:
            sibling_result = self.vector_store.get_chunk(sibling_id)
            if sibling_result:
                sibling_metadata = sibling_result.metadata
                all_parts.append({
                    'chunk_id': sibling_id,
                    'content': sibling_result.content,
                    'split_index': sibling_metadata.get('split_index', 0)
                })
        
        # Sort by split_index to maintain correct order
        all_parts.sort(key=lambda x: x['split_index'])
        
        # Merge contents
        merged_content = ' '.join(part['content'] for part in all_parts)
        
        print(f"      ✅ Reassembled {len(all_parts)} parts into full context")
        
        return merged_content
    
    def _format_context(self, results: List) -> str:
        """Format retrieved chunks into context with full reassembly for split chunks"""
        
        context_parts = []
        processed_parents = set()  # Track processed parent chunks to avoid duplicates
        
        for result in results:
            article = result.metadata.get('article', 'N/A')
            title = result.metadata.get('article_title', '')
            
            # Check if this is a split chunk
            if result.metadata.get('is_split', False):
                parent_chunk_id = result.metadata.get('parent_chunk_id')
                
                # Skip if we already processed this parent
                if parent_chunk_id in processed_parents:
                    continue
                
                # Assemble full content from all parts
                full_content = self._assemble_full_context(result)
                processed_parents.add(parent_chunk_id)
                
                formatted = f"""=== {article}. {title} ===
[Đã ghép lại từ {result.metadata.get('total_splits', 1)} phần]
{full_content}
---"""
            else:
                # Not split, use as-is
                formatted = f"""=== {article}. {title} ===
{result.content}
---"""
            
            context_parts.append(formatted)
        
        return '\n\n'.join(context_parts)

# ============================================================================
# UTILITY METHODS
# ============================================================================

    def get_system_stats(self) -> Dict[str, Any]:
        """Lấy thống kê hệ thống"""
        return {
            "embedding_model": self.config.embedding_model_id,
            "rerank_model": self.config.rerank_model_id,
            "embedding_weight": self.config.embedding_weight,
            "reranker_weight": self.config.reranker_weight,
            "total_chunks": len(self.vector_store.vectors),
            "chunk_families": len(self.chunk_families),
            "context_enabled": len(self.chunks_metadata) > 0
        }
    
    def format_search_results(self, results: List[AdvancedSearchResult]) -> str:
        """Format kết quả tìm kiếm"""
        if not results:
            return "Không tìm thấy kết quả phù hợp."
        
        response_parts = []
        
        for i, result in enumerate(results, 1):
            article = result.metadata.get('article', 'N/A')
            clause = result.metadata.get('clause', 'N/A')
            
            response_parts.append(f"=== KẾT QUẢ {i} ===")
            response_parts.append(f"📋 {article} - Khoản {clause}")
            response_parts.append(f"🎯 Độ tin cậy: {result.confidence_score:.3f}")
            response_parts.append(f"📊 Combined: {result.combined_score:.3f} | "
                                f"Embedding: {result.embedding_score:.3f} | "
                                f"Rerank: {result.rerank_score:.3f}")
            
            if result.is_split:
                response_parts.append(f"🔗 Context: Split chunk với {len(result.sibling_chunk_ids)} siblings")
            
            response_parts.append(f"📝 Nội dung: {result.content}")
            response_parts.append("\n" + "-" * 60 + "\n")
        
        return "\n".join(response_parts)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Test Advanced Retrieval System"""
    
    print("🧪 TESTING ADVANCED RETRIEVAL SYSTEM")
    print("=" * 60)
    
    # Load vector store
    vector_store = ContextAwareVectorStore()
    if not vector_store.load('context_aware_vectors.pkl'):
        print("❌ Cannot load vector store. Please run RAGembedding.py first.")
        return
    
    # Initialize advanced retrieval system
    config = RetrievalConfig(
        embedding_weight=0.4,
        reranker_weight=0.6,
        initial_search_k=15,
        final_results_k=3
    )
    
    retrieval_system = AdvancedRetrievalSystem(vector_store, config)
    
    # Test queries
    test_queries = [
        "Trách nhiệm hình sự của người dưới 16 tuổi",
        "Hình phạt tử hình áp dụng như thế nào?",
        "Điều kiện miễn trách nhiệm hình sự",
        "Hình phạt cho tội trộm cắp tài sản"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 TESTING: {query}")
        print(f"{'='*80}")
        
        try:
            results = retrieval_system.search_with_context(query, top_k=3)
            
            if results:
                formatted_response = retrieval_system.format_search_results(results)
                print(formatted_response)
            else:
                print("❌ No results found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue to next query...")
    
    # Show system stats
    stats = retrieval_system.get_system_stats()
    print(f"\n📊 SYSTEM STATISTICS:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    main()

