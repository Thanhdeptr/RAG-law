#!/usr/bin/env python3
"""
RAG Chunking System - Context-Aware Legal Document Chunker
Chunking văn bản pháp luật theo điều khoản với context preservation
"""

import re
import json
import unicodedata
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pyvi import ViTokenizer


# ======================== DATA CLASS ========================

@dataclass
class ContextAwareChunk:
    """Chunk với thông tin ngữ cảnh đầy đủ"""
    chunk_id: str
    type: str
    content: str
    token_count: int
    title: Optional[str] = None
    
    # Hierarchical information
    part: Optional[str] = None
    part_title: Optional[str] = None
    chapter: Optional[str] = None
    chapter_title: Optional[str] = None
    article: Optional[str] = None
    article_title: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    
    # Context linking
    hierarchy_path: Optional[str] = None
    cross_references: List[str] = None
    footnotes: List[str] = None
    
    # Context preservation
    is_split: bool = False
    parent_chunk_id: Optional[str] = None  # ID của chunk gốc trước khi split
    sibling_chunk_ids: List[str] = None    # IDs của các chunk anh em
    context_summary: Optional[str] = None  # Tóm tắt ngữ cảnh của chunk gốc
    split_index: int = 0                   # Thứ tự trong các chunk con
    total_splits: int = 1                  # Tổng số chunk con
    
    def __post_init__(self):
        if self.cross_references is None:
            self.cross_references = []
        if self.footnotes is None:
            self.footnotes = []
        if self.sibling_chunk_ids is None:
            self.sibling_chunk_ids = []


# ======================== MAIN CHUNKER CLASS ========================

class ContextAwareChunker:
    """Chunker thông minh: Chunking theo điều khoản + Context preservation"""
    
    def __init__(self, max_tokens: int = 180):
        self.max_tokens = max_tokens
        
        # Regex patterns
        self.patterns = {
            'part': re.compile(r'^Phần thứ (nhất|hai|ba|tư|năm|sáu|bảy|tám|chín|mười)', re.IGNORECASE),
            'chapter': re.compile(r'^Chương ([IVXLCDM]+)$', re.IGNORECASE),
            'article': re.compile(r'^Điều (\d+[a-z]?)\.?\s*(.*)$', re.IGNORECASE),  # Support 217a, 217b
            'clause': re.compile(r'^(\d+)\.(?:\[(\d+)\])?\s*(.*)$'),
            'point': re.compile(r'^([a-z]|đ)\)\s*(.*)$'),
            'footnote': re.compile(r'\[(\d+)\]\s*(.*)$'),
            'cross_ref': re.compile(r'(Điều \d+[a-z]?|điều \d+[a-z]?|khoản \d+|điểm [a-z])', re.IGNORECASE)
        }
        
        # Context tracking
        self.current_part = None
        self.current_part_title = None
        self.current_chapter = None
        self.current_chapter_title = None
        self.current_article = None
        self.current_article_title = None
        self.part_titles = {}
        self.chapter_titles = {}
        
        self.chunk_families = {}  # parent_id -> [child_ids]
    
    # ============ UTILITY METHODS ============
    
    def count_tokens(self, text: str) -> int:
        """Count tokens for Vietnamese text"""
        try:
            segmented = ViTokenizer.tokenize(text)
            tokens = segmented.split()
            return len(tokens)
        except Exception as e:
            print(f"Error tokenizing: {e}")
            return len(text.split())
    
    def create_context_summary(self, content: str) -> str:
        """Tạo tóm tắt ngữ cảnh cho chunk"""
        # Lấy câu đầu và câu cuối để tạo context summary
        sentences = self._split_by_sentences(content)
        
        if len(sentences) <= 2:
            return content[:200] + "..." if len(content) > 200 else content
        
        # Tạo summary từ câu đầu, giữa và cuối
        first_sentence = sentences[0]
        last_sentence = sentences[-1]
        
        # Lấy câu giữa nếu có nhiều câu
        middle_sentence = ""
        if len(sentences) > 4:
            middle_idx = len(sentences) // 2
            middle_sentence = sentences[middle_idx]
        
        summary_parts = [first_sentence]
        if middle_sentence:
            summary_parts.append(middle_sentence)
        summary_parts.append(last_sentence)
        
        return " ... ".join(summary_parts)
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Chia text theo câu"""
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        return sentences
    
    # ============ SPLITTING METHODS ============
    
    def smart_split_with_context(self, content: str, chunk_id: str, metadata: Dict) -> List[ContextAwareChunk]:
        """Chia chunk dài nhưng bảo toàn ngữ cảnh"""
        chunks = []
        
        # Tạo context summary cho chunk gốc
        context_summary = self.create_context_summary(content)
        
        # Chia theo câu
        sentences = self._split_by_sentences(content)
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Nếu câu đơn lẻ quá dài, chia nhỏ hơn
            if sentence_tokens > self.max_tokens:
                # Lưu chunk hiện tại nếu có
                if current_chunk:
                    chunks.append(self._create_context_chunk(
                        f"{chunk_id}_part_{chunk_index}", current_chunk, current_tokens, 
                        metadata, chunk_id, context_summary, chunk_index, len(sentences)
                    ))
                    chunk_index += 1
                    current_chunk = ""
                    current_tokens = 0
                
                # Chia câu dài thành các phần nhỏ
                sub_chunks = self._split_long_sentence_with_context(
                    sentence, chunk_id, metadata, context_summary, chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                continue
            
            # Nếu thêm câu này vượt quá giới hạn
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunks.append(self._create_context_chunk(
                    f"{chunk_id}_part_{chunk_index}", current_chunk, current_tokens, 
                    metadata, chunk_id, context_summary, chunk_index, len(sentences)
                ))
                chunk_index += 1
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_tokens += sentence_tokens
        
        # Thêm chunk cuối
        if current_chunk:
            chunks.append(self._create_context_chunk(
                f"{chunk_id}_part_{chunk_index}", current_chunk, current_tokens, 
                metadata, chunk_id, context_summary, chunk_index, len(sentences)
            ))
        
        # Cập nhật sibling relationships
        self._update_sibling_relationships(chunks, chunk_id)
        
        return chunks
    
    def _split_long_sentence_with_context(self, sentence: str, chunk_id: str, 
                                        metadata: Dict, context_summary: str, 
                                        start_index: int) -> List[ContextAwareChunk]:
        """Chia câu dài với ngữ cảnh"""
        chunks = []
        parts = re.split(r'[,;]\s*', sentence)
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = start_index
        
        for part in parts:
            part_tokens = self.count_tokens(part)
            
            if part_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(self._create_context_chunk(
                        f"{chunk_id}_part_{chunk_index}", current_chunk, current_tokens, 
                        metadata, chunk_id, context_summary, chunk_index, len(parts)
                    ))
                    chunk_index += 1
                    current_chunk = ""
                    current_tokens = 0
                
                word_chunks = self._split_by_words_with_context(
                    part, chunk_id, metadata, context_summary, chunk_index
                )
                chunks.extend(word_chunks)
                chunk_index += len(word_chunks)
                continue
            
            if current_tokens + part_tokens > self.max_tokens and current_chunk:
                chunks.append(self._create_context_chunk(
                    f"{chunk_id}_part_{chunk_index}", current_chunk, current_tokens, 
                    metadata, chunk_id, context_summary, chunk_index, len(parts)
                ))
                chunk_index += 1
                current_chunk = part
                current_tokens = part_tokens
            else:
                current_chunk += ", " + part if current_chunk else part
                current_tokens += part_tokens
        
        if current_chunk:
            chunks.append(self._create_context_chunk(
                f"{chunk_id}_part_{chunk_index}", current_chunk, current_tokens, 
                metadata, chunk_id, context_summary, chunk_index, len(parts)
            ))
        
        return chunks
    
    def _split_by_words_with_context(self, text: str, chunk_id: str, 
                                   metadata: Dict, context_summary: str, 
                                   start_index: int) -> List[ContextAwareChunk]:
        """Chia theo từ với ngữ cảnh (phương án cuối cùng)"""
        chunks = []
        words = text.split()
        
        current_chunk = ""
        current_tokens = 0
        chunk_index = start_index
        
        for word in words:
            word_tokens = self.count_tokens(word)
            
            if current_tokens + word_tokens > self.max_tokens and current_chunk:
                chunks.append(self._create_context_chunk(
                    f"{chunk_id}_part_{chunk_index}", current_chunk, current_tokens, 
                    metadata, chunk_id, context_summary, chunk_index, len(words)
                ))
                chunk_index += 1
                current_chunk = word
                current_tokens = word_tokens
            else:
                current_chunk += " " + word if current_chunk else word
                current_tokens += word_tokens
        
        if current_chunk:
            chunks.append(self._create_context_chunk(
                f"{chunk_id}_part_{chunk_index}", current_chunk, current_tokens, 
                metadata, chunk_id, context_summary, chunk_index, len(words)
            ))
        
        return chunks
    
    # ============ CHUNK CREATION METHODS ============
    
    def _create_context_chunk(self, chunk_id: str, content: str, token_count: int, 
                            metadata: Dict, parent_id: str, context_summary: str, 
                            split_index: int, total_splits: int) -> ContextAwareChunk:
        """Tạo chunk với thông tin ngữ cảnh"""
        return ContextAwareChunk(
            chunk_id=chunk_id,
            type='clause',
            content=content,
            token_count=token_count,
            title=metadata.get('title'),
            part=metadata.get('part'),
            part_title=metadata.get('part_title'),
            chapter=metadata.get('chapter'),
            chapter_title=metadata.get('chapter_title'),
            article=metadata.get('article'),
            article_title=metadata.get('article_title'),
            clause=metadata.get('clause'),
            point=metadata.get('point'),
            hierarchy_path=metadata.get('hierarchy_path'),
            cross_references=metadata.get('cross_references', []),
            footnotes=metadata.get('footnotes', []),
            is_split=True,
            parent_chunk_id=parent_id,
            context_summary=context_summary,
            split_index=split_index,
            total_splits=total_splits
        )
    
    def _update_sibling_relationships(self, chunks: List[ContextAwareChunk], parent_id: str):
        """Cập nhật mối quan hệ anh em giữa các chunks"""
        if len(chunks) <= 1:
            return
        
        # Lấy tất cả chunk IDs
        sibling_ids = [chunk.chunk_id for chunk in chunks]
        
        # Cập nhật sibling_chunk_ids cho mỗi chunk
        for chunk in chunks:
            chunk.sibling_chunk_ids = [cid for cid in sibling_ids if cid != chunk.chunk_id]
        
        self.chunk_families[parent_id] = sibling_ids
    
    # ============ MAIN PARSING METHODS ============
    
    def chunk_document(self, text: str) -> List[ContextAwareChunk]:
        """Main chunking function - Parse legal document and create chunks"""
        lines = text.split('\n')
        chunks = []
        current_clause_content = []
        current_clause_number = None
        current_article_context = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Normalize Unicode to handle combining characters
            line = unicodedata.normalize('NFC', line)
            
            # Check article header
            article_match = self.patterns['article'].match(line)
            if article_match:
                if current_clause_content and current_clause_number:
                    self._process_clause_with_context(current_clause_content, current_clause_number, 
                                                    current_article_context, chunks)
                
                article_num = article_match.group(1)
                article_title = article_match.group(2).strip()
                current_article_context = {
                    'article': f"Điều {article_num}",
                    'article_title': article_title,
                    'part': self.current_part,
                    'part_title': self.part_titles.get(self.current_part),
                    'chapter': self.current_chapter,
                    'chapter_title': self.chapter_titles.get(self.current_chapter),
                    'hierarchy_path': f"{self.current_part} > {self.current_chapter}" if self.current_part and self.current_chapter else None
                }
                current_clause_content = []
                current_clause_number = None
                continue
            
            # Check clause
            clause_match = self.patterns['clause'].match(line)
            if clause_match:
                if current_clause_content and current_clause_number:
                    self._process_clause_with_context(current_clause_content, current_clause_number, 
                                                    current_article_context, chunks)
                
                current_clause_number = clause_match.group(1)
                current_clause_content = [line]
                continue
            
            # Check part/chapter
            part_match = self.patterns['part'].match(line)
            if part_match:
                self.current_part = line
                self.part_titles[line] = None
                continue
            
            chapter_match = self.patterns['chapter'].match(line)
            if chapter_match:
                self.current_chapter = line
                self.chapter_titles[line] = None
                continue
            
            # Check title (UPPERCASE)
            if line.isupper() and len(line) > 5:
                if self.current_part and not self.part_titles.get(self.current_part):
                    self.part_titles[self.current_part] = line
                elif self.current_chapter and not self.chapter_titles.get(self.current_chapter):
                    self.chapter_titles[self.current_chapter] = line
                continue
            
            # Add to current clause
            if current_clause_number:
                current_clause_content.append(line)
        
        # Process last clause
        if current_clause_content and current_clause_number:
            self._process_clause_with_context(current_clause_content, current_clause_number, 
                                            current_article_context, chunks)
        
        return chunks
    
    def _process_clause_with_context(self, clause_content: List[str], clause_number: str, 
                                   article_context: Dict, chunks: List[ContextAwareChunk]):
        """Process clause - Create single chunk or split if over token limit"""
        content = '\n'.join(clause_content)
        token_count = self.count_tokens(content)
        metadata = {'clause': clause_number, **article_context}
        chunk_id = f"{article_context['article'].lower().replace(' ', '_')}_khoan_{clause_number}"
        
        if token_count <= self.max_tokens:
            # Create single chunk
            chunk = ContextAwareChunk(
                chunk_id=chunk_id,
                type='clause',
                content=content,
                token_count=token_count,
                title=metadata.get('title'),
                part=metadata.get('part'),
                part_title=metadata.get('part_title'),
                chapter=metadata.get('chapter'),
                chapter_title=metadata.get('chapter_title'),
                article=metadata.get('article'),
                article_title=metadata.get('article_title'),
                clause=metadata.get('clause'),
                point=metadata.get('point'),
                hierarchy_path=metadata.get('hierarchy_path'),
                cross_references=metadata.get('cross_references', []),
                footnotes=metadata.get('footnotes', [])
            )
            chunks.append(chunk)
        else:
            # Split into multiple chunks with context preservation
            split_chunks = self.smart_split_with_context(content, chunk_id, metadata)
            chunks.extend(split_chunks)
    
    # ============ SAVE/LOAD METHODS ============
    
    def save_chunks(self, chunks: List[ContextAwareChunk], filename: str):
        """Save chunks to JSON file"""
        chunks_data = []
        for chunk in chunks:
            chunk_dict = {
                'chunk_id': chunk.chunk_id,
                'type': chunk.type,
                'content': chunk.content,
                'token_count': chunk.token_count,
                'title': chunk.title,
                'part': chunk.part,
                'part_title': chunk.part_title,
                'chapter': chunk.chapter,
                'chapter_title': chunk.chapter_title,
                'article': chunk.article,
                'article_title': chunk.article_title,
                'clause': chunk.clause,
                'point': chunk.point,
                'hierarchy_path': chunk.hierarchy_path,
                'cross_references': chunk.cross_references,
                'footnotes': chunk.footnotes,
                'is_split': chunk.is_split,
                'parent_chunk_id': chunk.parent_chunk_id,
                'sibling_chunk_ids': chunk.sibling_chunk_ids,
                'context_summary': chunk.context_summary,
                'split_index': chunk.split_index,
                'total_splits': chunk.total_splits
            }
            chunks_data.append(chunk_dict)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(chunks)} context-aware chunks to {filename}")
    
    def get_chunk_families(self) -> Dict[str, List[str]]:
        """Get chunk families mapping"""
        return self.chunk_families.copy()


# ======================== TEST/DEMO FUNCTION ========================

def main():
    """Test and demo context-aware chunker"""
    
    # Read legal document
    try:
        with open('../data/01_VBHN-VPQH_363655.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("❌ Legal document not found. Please ensure the file exists.")
        return
    
    # Initialize chunker with 180 token limit
    chunker = ContextAwareChunker(max_tokens=180)
    print("🔧 Creating context-aware chunks...")
    chunks = chunker.chunk_document(text)
    print(f"📊 Generated {len(chunks)} chunks")
    
    # Analyze results
    token_counts = [chunk.token_count for chunk in chunks]
    max_tokens = max(token_counts)
    avg_tokens = sum(token_counts) / len(token_counts)
    over_limit = sum(1 for tc in token_counts if tc > 180)
    split_chunks = [c for c in chunks if c.is_split]
    
    print(f"📈 Statistics:")
    print(f"   Average tokens: {avg_tokens:.1f}")
    print(f"   Max tokens: {max_tokens}")
    print(f"   Chunks over 180 tokens: {over_limit}")
    print(f"   Split chunks: {len(split_chunks)}")
    print(f"   Chunk families: {len(chunker.get_chunk_families())}")
    
    # Show context linking examples
    if split_chunks:
        print(f"\n🔗 Context Linking Examples:")
        for chunk in split_chunks[:3]:
            print(f"   {chunk.chunk_id}:")
            print(f"     Parent: {chunk.parent_chunk_id}")
            print(f"     Siblings: {chunk.sibling_chunk_ids}")
            print(f"     Context: {chunk.context_summary[:100]}...")
            print()
    
    # Save chunks
    chunker.save_chunks(chunks, "context_aware_chunks.json")
    print(f"\n✅ Context-aware chunking complete!")


if __name__ == "__main__":
    main()
