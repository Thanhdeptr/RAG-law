#!/usr/bin/env python3
"""
Test Context Assembly Feature
Kiểm tra tính năng ghép lại chunks bị split
"""

from RAGassistant import LegalRAGAssistant


def test_context_assembly():
    """Test context assembly với câu hỏi có chunks bị split"""
    
    print("="*60)
    print("🧪 TEST CONTEXT ASSEMBLY FEATURE")
    print("="*60)
    
    # Initialize assistant
    print("\n🚀 Initializing Legal RAG Assistant...")
    assistant = LegalRAGAssistant()
    
    # Test queries that likely return split chunks
    test_queries = [
        "Khoan hồng đối với người tự thú, đầu thú",  # Điều 3 khoản 1 - known split
        "Căn cứ xác định tội phạm",  # Điều 9 khoản 1 - known split
        "Tội mua bán trái phép chất ma túy",  # May have split chunks
    ]
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "="*60)
        print(f"TEST {i}: {query}")
        print("="*60)
        
        try:
            response = assistant.ask(query)
            
            print("\n📊 RESULTS:")
            print(f"Query Type: {response.query_analysis.query_type}")
            print(f"Retrieved Chunks: {len(response.retrieved_chunks)}")
            
            # Check for split chunks
            split_chunks = [
                r for r in response.retrieved_chunks 
                if r.metadata.get('is_split', False)
            ]
            
            if split_chunks:
                print(f"\n🔗 SPLIT CHUNKS DETECTED: {len(split_chunks)}")
                for chunk in split_chunks:
                    print(f"   • {chunk.chunk_id}")
                    print(f"     Parent: {chunk.metadata.get('parent_chunk_id')}")
                    print(f"     Siblings: {len(chunk.metadata.get('sibling_chunk_ids', []))}")
                    print(f"     Split Index: {chunk.metadata.get('split_index')}/{chunk.metadata.get('total_splits')}")
            else:
                print("\n✅ NO SPLIT CHUNKS (all complete)")
            
            print(f"\n⚖️  Answer Preview:")
            print(response.answer[:500] + "..." if len(response.answer) > 500 else response.answer)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "-"*60)
        input("Press Enter to continue to next test...")


def test_specific_split_chunk():
    """Test với một chunk cụ thể đã biết là split"""
    
    print("\n" + "="*60)
    print("🎯 TEST SPECIFIC SPLIT CHUNK")
    print("="*60)
    
    assistant = LegalRAGAssistant()
    
    # Query that should return Điều 3 Khoản 1 (known to be split)
    query = "Khoan hồng đối với người tự thú"
    
    print(f"\nQuery: {query}")
    print("\nExpected: Điều 3 Khoản 1 (split into multiple parts)")
    
    response = assistant.ask(query)
    
    print("\n📊 DETAILED ANALYSIS:")
    for i, chunk in enumerate(response.retrieved_chunks, 1):
        print(f"\n{i}. {chunk.chunk_id}")
        print(f"   Article: {chunk.metadata.get('article')}")
        print(f"   Is Split: {chunk.metadata.get('is_split', False)}")
        
        if chunk.metadata.get('is_split'):
            print(f"   Parent: {chunk.metadata.get('parent_chunk_id')}")
            print(f"   Siblings: {chunk.metadata.get('sibling_chunk_ids', [])}")
            print(f"   Split: {chunk.metadata.get('split_index')}/{chunk.metadata.get('total_splits')}")
            print(f"   Context Summary: {chunk.metadata.get('context_summary', 'N/A')[:100]}...")
        
        print(f"   Content Length: {len(chunk.content)} chars")
        print(f"   Content Preview: {chunk.content[:150]}...")


def main():
    """Main test function"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--specific':
        test_specific_split_chunk()
    else:
        test_context_assembly()


if __name__ == "__main__":
    main()


