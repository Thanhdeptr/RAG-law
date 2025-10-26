#!/usr/bin/env python3
"""
Simple test for Context Assembly without LLM generation
Test chức năng Context Assembly đơn giản không cần LLM
"""

import json
from RAGembedding import ContextAwareVectorStore
from RAGretrieval import AdvancedRetrievalSystem, RetrievalConfig

def test_context_assembly():
    """Test Context Assembly functionality"""
    
    print("=" * 60)
    print("🧪 TESTING CONTEXT ASSEMBLY (Simple)")
    print("=" * 60)
    
    # Load vector store
    print("📥 Loading vector store...")
    vector_store = ContextAwareVectorStore()
    if not vector_store.load('context_aware_vectors.pkl'):
        print("❌ Cannot load vector store")
        return
    
    print(f"✅ Vector store loaded: {len(vector_store.vectors)} chunks")
    
    # Initialize retrieval system
    print("📥 Initializing retrieval system...")
    config = RetrievalConfig(
        embedding_weight=0.4,
        reranker_weight=0.6,
        initial_search_k=10,
        final_results_k=3
    )
    
    retrieval_system = AdvancedRetrievalSystem(vector_store, config)
    
    # Test query that should return split chunks
    query = "Khoan hồng đối với người tự thú"
    
    print(f"\n🔍 Testing query: '{query}'")
    print("Expected: Điều 3 Khoản 1 (split into multiple parts)")
    
    # Search for results
    results = retrieval_system.search(query, top_k=3)
    
    if not results:
        print("❌ No results found")
        return
    
    print(f"\n✅ Found {len(results)} results")
    
    # Test Context Assembly
    print(f"\n🔗 Testing Context Assembly:")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- RESULT {i} ---")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Is Split: {result.is_split}")
        print(f"Parent ID: {result.parent_chunk_id}")
        print(f"Sibling IDs: {result.sibling_chunk_ids}")
        print(f"Content (first 100 chars): {result.content[:100]}...")
        
        if result.is_split:
            print(f"\n🔗 Testing assembly for split chunk...")
            
            # Test the assembly method
            try:
                assembled_content = retrieval_system._assemble_full_context(result)
                print(f"✅ Assembly successful!")
                print(f"Original length: {len(result.content)} chars")
                print(f"Assembled length: {len(assembled_content)} chars")
                print(f"Assembled content (first 200 chars): {assembled_content[:200]}...")
                
                # Check if assembly actually merged content
                if len(assembled_content) > len(result.content):
                    print(f"✅ Content was successfully merged!")
                else:
                    print(f"⚠️ Content length didn't increase - may not have siblings")
                    
            except Exception as e:
                print(f"❌ Assembly failed: {e}")
        else:
            print(f"ℹ️ Not a split chunk - no assembly needed")
    
    # Test formatting with assembly
    print(f"\n📄 Testing context formatting with assembly:")
    try:
        formatted_context = retrieval_system._format_context(results)
        print(f"✅ Context formatting successful!")
        print(f"Formatted context length: {len(formatted_context)} chars")
        print(f"Formatted context preview:")
        print("-" * 40)
        print(formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Context formatting failed: {e}")
    
    print(f"\n✅ Context Assembly test completed!")

if __name__ == "__main__":
    test_context_assembly()

