#!/usr/bin/env python3
"""
Context-Aware Retrieval System with HuggingFace Embedding
Hệ thống retrieval thông minh với model huyydangg/DEk21_hcmute_embedding
"""

import json
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from context_aware_vector_store import ContextAwareVectorStore, SearchResult

@dataclass
class ContextAwareSearchResult:
    """Kết quả tìm kiếm với ngữ cảnh đầy đủ"""
    primary_chunk: SearchResult
    related_chunks: List[SearchResult]
    context_summary: str
    total_chunks: int
    confidence_score: float

class ContextAwareRetrievalHF:
    """
    Hệ thống retrieval thông minh với HuggingFace embedding
    """
    
    def __init__(self, vector_store: ContextAwareVectorStore, chunks_file: str = "context_aware_chunks.json"):
        self.vector_store = vector_store
        self.chunks_metadata = self._load_chunks_metadata(chunks_file)
        self.chunk_families = self._build_chunk_families()
    
    def _load_chunks_metadata(self, chunks_file: str) -> Dict[str, Dict]:
        """Load metadata của chunks"""
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Tạo mapping từ chunk_id đến metadata
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
    
    def search_with_context(self, query: str, top_k: int = 5, 
                          include_siblings: bool = True,
                          include_parent: bool = True) -> List[ContextAwareSearchResult]:
        """
        Tìm kiếm với ngữ cảnh đầy đủ
        
        Args:
            query: Câu hỏi tìm kiếm
            top_k: Số lượng kết quả chính
            include_siblings: Có bao gồm chunks anh em không
            include_parent: Có bao gồm chunk cha không
        """
        
        # Tìm kiếm cơ bản
        primary_results = self.vector_store.search(query, top_k=top_k)
        
        context_results = []
        
        for primary_result in primary_results:
            # Lấy chunks liên quan
            related_chunks = self._get_related_chunks(
                primary_result.chunk_id, include_siblings, include_parent
            )
            
            # Tạo context summary
            context_summary = self._create_context_summary(primary_result, related_chunks)
            
            # Tính confidence score
            confidence_score = self._calculate_confidence_score(primary_result, related_chunks)
            
            # Tạo kết quả với ngữ cảnh
            context_result = ContextAwareSearchResult(
                primary_chunk=primary_result,
                related_chunks=related_chunks,
                context_summary=context_summary,
                total_chunks=len(related_chunks) + 1,
                confidence_score=confidence_score
            )
            
            context_results.append(context_result)
        
        return context_results
    
    def _get_related_chunks(self, chunk_id: str, include_siblings: bool, 
                          include_parent: bool) -> List[SearchResult]:
        """Lấy các chunks liên quan"""
        related_chunks = []
        
        if chunk_id not in self.chunks_metadata:
            return related_chunks
        
        metadata = self.chunks_metadata[chunk_id]
        
        # Lấy sibling chunks
        if include_siblings and metadata.get('is_split', False):
            sibling_ids = metadata.get('sibling_chunk_ids', [])
            for sibling_id in sibling_ids:
                sibling_chunk = self.vector_store.get_chunk(sibling_id)
                if sibling_chunk:
                    related_chunks.append(sibling_chunk)
        
        # Lấy parent chunk
        if include_parent and metadata.get('is_split', False):
            parent_id = metadata.get('parent_chunk_id')
            if parent_id:
                parent_chunk = self.vector_store.get_chunk(parent_id)
                if parent_chunk:
                    related_chunks.append(parent_chunk)
        
        return related_chunks
    
    def _create_context_summary(self, primary_chunk: SearchResult, 
                              related_chunks: List[SearchResult]) -> str:
        """Tạo tóm tắt ngữ cảnh"""
        
        # Lấy context summary từ metadata nếu có
        if primary_chunk.chunk_id in self.chunks_metadata:
            metadata = self.chunks_metadata[primary_chunk.chunk_id]
            context_summary = metadata.get('context_summary')
            if context_summary:
                return context_summary
        
        # Nếu không có context summary, tạo từ nội dung
        all_content = [primary_chunk.content]
        for chunk in related_chunks:
            all_content.append(chunk.content)
        
        # Tạo summary từ nội dung
        combined_content = " ".join(all_content)
        
        # Lấy câu đầu và cuối
        sentences = combined_content.split('.')
        if len(sentences) <= 2:
            return combined_content[:300] + "..." if len(combined_content) > 300 else combined_content
        
        first_sentence = sentences[0]
        last_sentence = sentences[-1]
        
        return f"{first_sentence}... {last_sentence}"
    
    def _calculate_confidence_score(self, primary_chunk: SearchResult, 
                                  related_chunks: List[SearchResult]) -> float:
        """Tính điểm tin cậy dựa trên ngữ cảnh"""
        
        # Điểm cơ bản từ primary chunk
        base_score = primary_chunk.score
        
        # Bonus cho việc có ngữ cảnh đầy đủ
        context_bonus = 0.0
        if related_chunks:
            # Càng nhiều chunks liên quan, điểm càng cao
            context_bonus = min(0.2, len(related_chunks) * 0.05)
        
        # Bonus cho chunks không bị split (ngữ cảnh nguyên vẹn)
        integrity_bonus = 0.0
        if primary_chunk.chunk_id in self.chunks_metadata:
            metadata = self.chunks_metadata[primary_chunk.chunk_id]
            if not metadata.get('is_split', False):
                integrity_bonus = 0.1
        
        final_score = min(1.0, base_score + context_bonus + integrity_bonus)
        return final_score
    
    def get_full_context(self, chunk_id: str) -> Dict[str, Any]:
        """Lấy ngữ cảnh đầy đủ của một chunk"""
        
        if chunk_id not in self.chunks_metadata:
            return {"error": "Chunk not found"}
        
        metadata = self.chunks_metadata[chunk_id]
        
        context = {
            "chunk_id": chunk_id,
            "content": metadata.get('content', ''),
            "token_count": metadata.get('token_count', 0),
            "article": metadata.get('article', ''),
            "clause": metadata.get('clause', ''),
            "hierarchy_path": metadata.get('hierarchy_path', ''),
            "is_split": metadata.get('is_split', False),
            "context_summary": metadata.get('context_summary', ''),
            "related_chunks": []
        }
        
        # Thêm thông tin chunks liên quan
        if metadata.get('is_split', False):
            context["parent_chunk_id"] = metadata.get('parent_chunk_id')
            context["sibling_chunk_ids"] = metadata.get('sibling_chunk_ids', [])
            context["split_index"] = metadata.get('split_index', 0)
            context["total_splits"] = metadata.get('total_splits', 1)
            
            # Lấy nội dung của sibling chunks
            for sibling_id in metadata.get('sibling_chunk_ids', []):
                if sibling_id in self.chunks_metadata:
                    sibling_metadata = self.chunks_metadata[sibling_id]
                    context["related_chunks"].append({
                        "chunk_id": sibling_id,
                        "content": sibling_metadata.get('content', ''),
                        "split_index": sibling_metadata.get('split_index', 0)
                    })
        
        return context
    
    def format_context_response(self, context_results: List[ContextAwareSearchResult]) -> str:
        """Format kết quả tìm kiếm với ngữ cảnh"""
        
        if not context_results:
            return "Không tìm thấy kết quả phù hợp."
        
        response_parts = []
        
        for i, result in enumerate(context_results, 1):
            # Thông tin chunk chính
            primary = result.primary_chunk
            article = primary.metadata.get('article', 'N/A') if primary.metadata else 'N/A'
            clause = primary.metadata.get('clause', 'N/A') if primary.metadata else 'N/A'
            
            response_parts.append(f"=== KẾT QUẢ {i} ===")
            response_parts.append(f"📋 {article} - Khoản {clause}")
            response_parts.append(f"🎯 Độ tin cậy: {result.confidence_score:.2f}")
            response_parts.append(f"📊 Tổng chunks: {result.total_chunks}")
            
            # Context summary
            if result.context_summary:
                response_parts.append(f"📖 Ngữ cảnh: {result.context_summary}")
            
            # Nội dung chunk chính
            response_parts.append(f"📝 Nội dung chính:")
            response_parts.append(primary.content)
            
            # Chunks liên quan
            if result.related_chunks:
                response_parts.append(f"\n🔗 Chunks liên quan ({len(result.related_chunks)} chunks):")
                for j, related in enumerate(result.related_chunks, 1):
                    related_article = related.metadata.get('article', 'N/A') if related.metadata else 'N/A'
                    related_clause = related.metadata.get('clause', 'N/A') if related.metadata else 'N/A'
                    response_parts.append(f"   {j}. {related_article} - Khoản {related_clause}")
                    response_parts.append(f"      {related.content[:200]}...")
            
            response_parts.append("\n" + "-" * 60 + "\n")
        
        return "\n".join(response_parts)

def main():
    """Test context-aware retrieval with HuggingFace embedding"""
    
    print("🔍 Testing Context-Aware Retrieval with HuggingFace Embedding")
    print("=" * 70)
    
    # Load vector store
    vector_store = ContextAwareVectorStore()
    if not vector_store.load('context_aware_vectors.pkl'):
        print("❌ Cannot load vector store. Please run context_aware_vector_store.py first.")
        return
    
    # Initialize context-aware retrieval
    retrieval = ContextAwareRetrievalHF(vector_store, "context_aware_chunks.json")
    
    # Test queries
    test_queries = [
        "Người lao động có quyền nghỉ thai sản bao lâu?",
        "Hình phạt tử hình áp dụng như thế nào?",
        "Điều kiện miễn trách nhiệm hình sự",
        "Trách nhiệm hình sự của người dưới 16 tuổi",
        "Hình phạt cho tội trộm cắp tài sản"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        # Search with context
        results = retrieval.search_with_context(query, top_k=2)
        
        # Format and display results
        formatted_response = retrieval.format_context_response(results)
        print(formatted_response)
        
        # Show detailed context information
        if results:
            print("🔍 DETAILED CONTEXT ANALYSIS:")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Primary chunk: {result.primary_chunk.chunk_id}")
                print(f"  Score: {result.primary_chunk.score:.3f}")
                print(f"  Token count: {result.primary_chunk.metadata.get('token_count', 'N/A')}")
                print(f"  Is split: {result.primary_chunk.metadata.get('is_split', False)}")
                if result.primary_chunk.metadata.get('is_split'):
                    print(f"  Parent: {result.primary_chunk.metadata.get('parent_chunk_id', 'N/A')}")
                    print(f"  Siblings: {result.primary_chunk.metadata.get('sibling_chunk_ids', [])}")
                print(f"  Related chunks: {len(result.related_chunks)}")
                print(f"  Confidence: {result.confidence_score:.3f}")
        
        input("\nPress Enter to continue to next query...")

if __name__ == "__main__":
    main()

