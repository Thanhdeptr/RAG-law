#!/usr/bin/env python3
"""Rebuild vector embeddings for new clause-level chunks"""

import json
import numpy as np
import ollama
from simple_vector_store import SimpleVectorStore

def rebuild_vectors():
    print("🔄 Rebuilding vector embeddings for new clause-level chunks...")
    
    # Load new chunks
    with open('legal_clause_chunks.json', 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    print(f"📊 Loaded {len(chunks_data)} clause-level chunks")
    
    # Initialize vector store
    vector_store = SimpleVectorStore()
    
    # Ollama client will be initialized automatically by SimpleVectorStore
    
    print("🤖 Generating embeddings...")
    
    # Process chunks in batches (smaller to avoid server overload)
    batch_size = 10
    for i in range(0, len(chunks_data), batch_size):
        batch = chunks_data[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks_data)-1)//batch_size + 1}")
        
        for chunk_data in batch:
            try:
                # Create content for embedding
                content = chunk_data['content']
                title = chunk_data.get('title', '')
                hierarchy = chunk_data.get('hierarchy_path', '')
                
                # Combine content with context for better embeddings
                embedding_text = f"{hierarchy}\n{title}\n{content}"
                
                # Embedding will be generated automatically by SimpleVectorStore.add_chunk()
                
                # Add to vector store
                vector_store.add_chunk(
                    chunk_id=chunk_data['chunk_id'],
                    content=embedding_text,
                    metadata={
                        'type': chunk_data['type'],
                        'title': title,
                        'hierarchy_path': hierarchy,
                        'part': chunk_data.get('part'),
                        'part_title': chunk_data.get('part_title'),
                        'chapter': chunk_data.get('chapter'),
                        'chapter_title': chunk_data.get('chapter_title'),
                        'article': chunk_data.get('article'),
                        'article_title': chunk_data.get('article_title'),
                        'clause': chunk_data.get('clause'),
                        'cross_references': chunk_data.get('cross_references', []),
                        'footnotes': chunk_data.get('footnotes', [])
                    }
                )
                
            except Exception as e:
                print(f"❌ Error processing chunk {chunk_data['chunk_id']}: {e}")
                continue
    
    # Save vector store
    vector_store.save('legal_vectors.pkl')
    print(f"✅ Saved vector embeddings for {len(chunks_data)} chunks")
    print(f"📁 Vector store saved to: legal_vectors.pkl")

if __name__ == "__main__":
    rebuild_vectors()
