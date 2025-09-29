#!/usr/bin/env python3
"""
Legal Document Chunker
Chuyên dụng cho việc chunking văn bản pháp luật Việt Nam theo cấu trúc phân cấp
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LegalChunk:
    """Represents a legal document chunk with hierarchical metadata"""
    chunk_id: str
    type: str  # 'part', 'chapter', 'article', 'clause', 'point'
    content: str
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
    
    # Additional metadata
    hierarchy_path: Optional[str] = None
    cross_references: List[str] = None
    footnotes: List[str] = None
    
    def __post_init__(self):
        if self.cross_references is None:
            self.cross_references = []
        if self.footnotes is None:
            self.footnotes = []


class FootnoteLookup:
    """Class để tra cứu footnote từ cuối file"""
    
    def __init__(self, text: str):
        self.footnotes = self._extract_footnotes(text)
    
    def _extract_footnotes(self, text: str) -> Dict[str, str]:
        """Trích xuất tất cả footnote từ cuối file"""
        lines = text.split('\n')
        footnotes = {}
        
        # Tìm từ cuối file lên
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
                
            # Match footnote pattern: [1] content
            match = re.match(r'^\[(\d+)\]\s*(.*)$', line)
            if match:
                footnote_num = match.group(1)
                footnote_content = match.group(2).strip()
                footnotes[footnote_num] = footnote_content
            else:
                # Nếu gặp dòng không phải footnote, dừng lại
                break
                
        return footnotes
    
    def lookup_footnote(self, footnote_num: str) -> Optional[str]:
        """Tra cứu footnote theo số"""
        return self.footnotes.get(footnote_num)
    
    def get_all_footnotes(self) -> Dict[str, str]:
        """Lấy tất cả footnote"""
        return self.footnotes.copy()


class LegalDocumentChunker:
    """
    Chunker chuyên dụng cho văn bản pháp luật Việt Nam
    Hỗ trợ cấu trúc: Phần > Chương > Điều > Khoản > Điểm
    """
    
    def __init__(self):
        # Regex patterns for different legal structures
        self.patterns = {
            'part': re.compile(r'^Phần thứ (nhất|hai|ba|tư|năm|sáu|bảy|tám|chín|mười)', re.IGNORECASE),
            'chapter': re.compile(r'^Chương ([IVXLCDM]+)$', re.IGNORECASE),
            'article': re.compile(r'^Điều (\d+)\.?\s*(.*)$', re.IGNORECASE),  # Chỉ nhận "Điều 180. Title"
            'clause': re.compile(r'^(\d+)\.(?:\[(\d+)\])?\s*(.*)$'),  # Hỗ trợ "1.[225] content" và "1. content"
            'point': re.compile(r'^([a-z]|đ)\)\s*(.*)$'),
            'footnote': re.compile(r'\[(\d+)\]\s*(.*)$'),  # Nhận tất cả chú thích [1], [180], etc.
            'cross_ref': re.compile(r'(Điều \d+|điều \d+|khoản \d+|điểm [a-z])', re.IGNORECASE)
        }
        
        # Current context tracking
        self.current_part = None
        self.current_part_title = None
        self.current_chapter = None
        self.current_chapter_title = None
        self.current_article = None
        self.current_article_title = None
        
        # Store titles for each part/chapter to avoid overwriting
        self.part_titles = {}  # {"Phần thứ nhất": "NHỮNG QUY ĐỊNH CHUNG"}
        self.chapter_titles = {}  # {"Chương I": "ĐIỀU KHOẢN CƠ BẢN"}
        
    def parse_document(self, text: str) -> List[LegalChunk]:
        """
        Parse toàn bộ văn bản pháp luật thành các chunks có cấu trúc
        """
        lines = text.split('\n')
        chunks = []
        
        # Buffer for accumulating content
        current_content = []
        current_type = None
        current_id = None
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check for different legal structures
            chunk_info = self._identify_line_type(line)
            
            if chunk_info:
                # Save previous chunk if exists
                if current_content and current_type and current_id:
                    chunk = self._create_chunk(
                        current_id, current_type, '\n'.join(current_content)
                    )
                    if chunk:
                        chunks.append(chunk)
                
                # Update context and start new chunk
                self._update_context(chunk_info)
                current_type = chunk_info['type']
                current_id = self._generate_chunk_id(chunk_info)
                current_content = [line]
                
            else:
                # Check if this is a title (FULL UPPERCASE) and update context
                if line.isupper() and len(line.strip()) > 3 and not line.isdigit():
                    # Store title in dictionaries to avoid overwriting
                    if self.current_part and self.current_part not in self.part_titles:
                        self.part_titles[self.current_part] = line.strip()
                        self.current_part_title = line.strip()
                    elif self.current_chapter and self.current_chapter not in self.chapter_titles:
                        self.chapter_titles[self.current_chapter] = line.strip()
                        self.current_chapter_title = line.strip()
                # Continue accumulating content
                if current_content:
                    current_content.append(line)
        
        # Add final chunk
        if current_content and current_type and current_id:
            chunk = self._create_chunk(
                current_id, current_type, '\n'.join(current_content)
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _identify_line_type(self, line: str) -> Optional[Dict[str, Any]]:
        """Identify the type of legal structure in a line"""
        
        # Check for Part
        match = self.patterns['part'].match(line)
        if match:
            return {
                'type': 'part',
                'number': match.group(1),
                'title': line.split(':', 1)[1].strip() if ':' in line else None
            }
        
        # Check for Chapter
        match = self.patterns['chapter'].match(line)
        if match:
            return {
                'type': 'chapter', 
                'number': match.group(1),
                'title': None  # Title will be captured from next line
            }
        
        # Check for Title (FULL UPPERCASE text - chapter/part titles)
        # Don't create separate chunks, just update context
        if line.isupper() and len(line.strip()) > 3 and not line.isdigit():
            return None  # Skip creating title chunks, update context in _update_context
        
        # Check for Article (only "Điều 180. Title" format)
        match = self.patterns['article'].match(line)
        if match:
            return {
                'type': 'article',
                'number': match.group(1),
                'title': match.group(2).strip() if match.group(2) else None
            }
        
        # Check for Clause (support both "1. content" and "1.[225] content")
        match = self.patterns['clause'].match(line)
        if match:
            clause_number = match.group(1)
            footnote_ref = match.group(2) if match.group(2) else None
            content = match.group(3).strip() if match.group(3) else ""
            return {
                'type': 'clause',
                'number': clause_number,
                'footnote_ref': footnote_ref,
                'content': content
            }
        
        # Check for Point
        match = self.patterns['point'].match(line)
        if match:
            return {
                'type': 'point',
                'letter': match.group(1),
                'content': match.group(2).strip()
            }
        
        # Check for Footnote (both [1] and [180] style)
        match = self.patterns['footnote'].match(line)
        if match:
            return {
                'type': 'footnote',
                'number': match.group(1),
                'content': match.group(2).strip() if match.group(2) else None
            }
        
        return None
    
    def _update_context(self, chunk_info: Dict[str, Any]):
        """Update current hierarchical context"""
        
        if chunk_info['type'] == 'part':
            self.current_part = f"Phần thứ {chunk_info['number']}"
            self.current_part_title = chunk_info.get('title')  # Will be set by title detection
            # Reset lower levels
            self.current_chapter = None
            self.current_chapter_title = None
            self.current_article = None
            self.current_article_title = None
            
        elif chunk_info['type'] == 'chapter':
            self.current_chapter = f"Chương {chunk_info['number']}"
            self.current_chapter_title = chunk_info.get('title')  # Will be set by title detection
            # Reset lower levels
            self.current_article = None
            self.current_article_title = None
            
        elif chunk_info['type'] == 'article':
            self.current_article = f"Điều {chunk_info['number']}"
            self.current_article_title = chunk_info.get('title')
        elif chunk_info['type'] == 'title':
            # Title updates the current context (part or chapter title)
            if self.current_part and not self.current_part_title:
                self.current_part_title = chunk_info['content']
            elif self.current_chapter and not self.current_chapter_title:
                self.current_chapter_title = chunk_info['content']
        elif chunk_info['type'] == 'footnote':
            # Footnote doesn't change context, just marks content
            pass
    
    def _generate_chunk_id(self, chunk_info: Dict[str, Any]) -> str:
        """Generate unique chunk ID"""
        
        if chunk_info['type'] == 'part':
            return f"phan_{chunk_info['number']}"
        elif chunk_info['type'] == 'chapter':
            return f"chuong_{chunk_info['number']}"
        elif chunk_info['type'] == 'article':
            return f"dieu_{chunk_info['number']}"
        elif chunk_info['type'] == 'clause':
            article_num = self.current_article.split()[-1] if self.current_article else "0"
            return f"dieu_{article_num}_khoan_{chunk_info['number']}"
        elif chunk_info['type'] == 'point':
            article_num = self.current_article.split()[-1] if self.current_article else "0"
            return f"dieu_{article_num}_diem_{chunk_info['letter']}"
        elif chunk_info['type'] == 'footnote':
            return f"footnote_{chunk_info['number']}"
        elif chunk_info['type'] == 'title':
            return f"title_{hash(chunk_info['content'])}"
        
        return f"chunk_{hash(str(chunk_info))}"
    
    def _create_chunk(self, chunk_id: str, chunk_type: str, content: str) -> Optional[LegalChunk]:
        """Create a LegalChunk with full metadata"""
        
        if not content.strip():
            return None
        
        # Extract title from content for articles
        title = None
        if chunk_type == 'article':
            lines = content.split('\n')
            first_line = lines[0] if lines else ""
            # Handle "Điều 180. Title" format
            if '. ' in first_line:
                title = first_line.split('. ', 1)[1].strip()
        elif chunk_type == 'title':
            # Title is the content itself
            title = content.strip()
        elif chunk_type == 'chapter':
            # Chapter title usually comes after chapter declaration
            lines = content.split('\n')
            if len(lines) > 1:
                title = lines[1].strip()
        
        # Extract cross-references
        cross_refs = self.patterns['cross_ref'].findall(content)
        
        # Extract footnotes
        footnotes = self.patterns['footnote'].findall(content)
        
        # Build hierarchy path
        hierarchy_parts = []
        if self.current_part:
            hierarchy_parts.append(self.current_part)
        if self.current_chapter:
            hierarchy_parts.append(self.current_chapter)
        if self.current_article and chunk_type in ['clause', 'point']:
            hierarchy_parts.append(self.current_article)
        
        hierarchy_path = ' > '.join(hierarchy_parts) if hierarchy_parts else None
        
        return LegalChunk(
            chunk_id=chunk_id,
            type=chunk_type,
            content=content.strip(),
            title=title,
            part=self.current_part,
            part_title=self.current_part_title,
            chapter=self.current_chapter,
            chapter_title=self.current_chapter_title,
            article=self.current_article,
            article_title=self.current_article_title,
            hierarchy_path=hierarchy_path,
            cross_references=cross_refs,
            footnotes=footnotes
        )
    
    
    def enrich_with_footnotes(self, chunks: List[LegalChunk], text: str) -> List[LegalChunk]:
        """Bổ sung thông tin footnote vào các chunks"""
        footnote_lookup = FootnoteLookup(text)
        
        for chunk in chunks:
            if chunk.footnotes:
                enriched_footnotes = []
                for footnote_num in chunk.footnotes:
                    footnote_content = footnote_lookup.lookup_footnote(footnote_num)
                    if footnote_content:
                        enriched_footnotes.append(f"[{footnote_num}] {footnote_content}")
                    else:
                        enriched_footnotes.append(f"[{footnote_num}] (Không tìm thấy)")
                
                # Thêm footnote vào cuối content
                if enriched_footnotes:
                    chunk.content += "\n\n📖 CHÚ THÍCH:\n" + "\n".join(enriched_footnotes)
        
        return chunks
    
    def chunk_by_clauses_only(self, text: str) -> List[LegalChunk]:
        """
        Chunking strategy focusing ONLY on Clauses (Khoản) as primary chunks
        Articles become metadata for each clause
        """
        all_chunks = self.parse_document(text)
        
        clause_chunks = []
        current_article_context = None
        
        for chunk in all_chunks:
            if chunk.type == 'article':
                # Update article context (metadata for clauses)
                current_article_context = {
                    'article': chunk.article,
                    'article_title': chunk.article_title,
                    'part': chunk.part,
                    'chapter': chunk.chapter,
                    'hierarchy_path': chunk.hierarchy_path,
                    'article_content': chunk.content  # Store full article as metadata
                }
                
                # Tách khoản từ article content với original text
                clause_chunks.extend(self._split_article_into_clauses(chunk, current_article_context, text))
                
            elif chunk.type in ['clause', 'point'] and current_article_context:
                # Skip individual clause/point chunks from parse_document
                # We only want clause chunks created by _split_article_into_clauses
                continue
                
            elif chunk.type in ['footnote', 'title']:
                # Skip footnotes and titles - they are just references/metadata
                continue
        
        return clause_chunks
    
    def _extract_article_full_content(self, article_num: str, original_text: str) -> str:
        """Extract full article content from original text"""
        lines = original_text.split('\n')
        
        # Find the start of the article
        start_line = None
        for i, line in enumerate(lines):
            if f"Điều {article_num}." in line:
                start_line = i
                break
        
        if start_line is None:
            return ""
        
        # Find the end of the article (next article or end of document)
        end_line = len(lines)
        for i in range(start_line + 1, len(lines)):
            line = lines[i].strip()
            if line.startswith('Điều ') and line != f"Điều {article_num}.":
                end_line = i
                break
        
        # Extract content from start_line to end_line
        article_lines = lines[start_line:end_line]
        return '\n'.join(article_lines)
    
    def _split_article_into_clauses(self, article_chunk: LegalChunk, article_context: Dict, original_text: str) -> List[LegalChunk]:
        """Tách article thành các clause chunks riêng biệt"""
        clause_chunks = []
        
        # Get full article content from original text instead of truncated article chunk
        # Find article number from chunk ID (e.g., "dieu_3" -> "3")
        article_num = article_chunk.chunk_id.split('_')[1]
        
        # Extract full article content from original text
        full_content = self._extract_article_full_content(article_num, original_text)
        if not full_content:
            return clause_chunks
            
        lines = full_content.split('\n')
        
        current_clause = None
        current_clause_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Kiểm tra xem có phải khoản mới không
            clause_match = self.patterns['clause'].match(line)
            if clause_match:
                # Lưu khoản trước đó nếu có
                if current_clause and current_clause_content:
                    clause_chunk = self._create_clause_chunk(
                        article_chunk, article_context, current_clause, current_clause_content
                    )
                    if clause_chunk:
                        clause_chunks.append(clause_chunk)
                
                # Bắt đầu khoản mới
                current_clause = clause_match.group(1)
                footnote_ref = clause_match.group(2) if clause_match.group(2) else None
                content = clause_match.group(3).strip() if clause_match.group(3) else ""
                current_clause_content = [content] if content else []
                
            else:
                # Thêm vào khoản hiện tại (bao gồm cả điểm a), b), c)...)
                if current_clause_content is not None:
                    current_clause_content.append(line)
        
        # Lưu khoản cuối cùng
        if current_clause and current_clause_content:
            clause_chunk = self._create_clause_chunk(
                article_chunk, article_context, current_clause, current_clause_content
            )
            if clause_chunk:
                clause_chunks.append(clause_chunk)
        
        return clause_chunks
    
    def _create_clause_chunk_with_article_metadata(self, clause_chunk: LegalChunk, article_context: Dict) -> LegalChunk:
        """Tạo clause chunk với article làm metadata"""
        return LegalChunk(
            chunk_id=clause_chunk.chunk_id,
            type='clause',
            content=clause_chunk.content,
            title=clause_chunk.content.split('. ', 1)[1] if '. ' in clause_chunk.content else None,
            
            # Article metadata
            part=article_context['part'],
            part_title=self.part_titles.get(article_context['part']),  # ✅ GET FROM DICT
            chapter=article_context['chapter'],
            chapter_title=self.chapter_titles.get(article_context['chapter']),  # ✅ GET FROM DICT
            article=article_context['article'],
            article_title=article_context['article_title'],
            hierarchy_path=article_context['hierarchy_path'],
            
            # Clause-specific info
            clause=clause_chunk.content.split('.', 1)[0] if '.' in clause_chunk.content else None,
            cross_references=clause_chunk.cross_references,
            footnotes=clause_chunk.footnotes
        )
    
    def _create_point_chunk_with_article_metadata(self, point_chunk: LegalChunk, article_context: Dict) -> LegalChunk:
        """Tạo point chunk với article làm metadata"""
        return LegalChunk(
            chunk_id=point_chunk.chunk_id,
            type='point',
            content=point_chunk.content,
            title=point_chunk.content.split(') ', 1)[1] if ') ' in point_chunk.content else None,
            
            # Article metadata
            part=article_context['part'],
            part_title=self.part_titles.get(article_context['part']),  # ✅ GET FROM DICT
            chapter=article_context['chapter'],
            chapter_title=self.chapter_titles.get(article_context['chapter']),  # ✅ GET FROM DICT
            article=article_context['article'],
            article_title=article_context['article_title'],
            hierarchy_path=article_context['hierarchy_path'],
            
            # Point-specific info
            point=point_chunk.content.split(')', 1)[0] + ')' if ')' in point_chunk.content else None,
            cross_references=point_chunk.cross_references,
            footnotes=point_chunk.footnotes
        )
    
    def _create_clause_chunk(self, article_chunk: LegalChunk, article_context: Dict, 
                           clause_number: str, clause_content: List[str]) -> LegalChunk:
        """Tạo clause chunk từ thông tin khoản"""
        if not clause_content:
            return None
            
        content = '\n'.join(clause_content)
        chunk_id = f"{article_chunk.chunk_id}_khoan_{clause_number}"
        
        return LegalChunk(
            chunk_id=chunk_id,
            type='clause',
            content=content,
            title=content.split('. ', 1)[1] if '. ' in content else None,
            
            # Inherit article context
            part=article_context['part'],
            part_title=self.part_titles.get(article_context['part']),  # ✅ GET FROM DICT
            chapter=article_context['chapter'],
            chapter_title=self.chapter_titles.get(article_context['chapter']),  # ✅ GET FROM DICT
            article=article_context['article'],
            article_title=article_context['article_title'],
            hierarchy_path=article_context['hierarchy_path'],
            
            # Clause-specific info
            clause=clause_number,
            cross_references=article_chunk.cross_references,
            footnotes=article_chunk.footnotes
        )


def main():
    """Test the legal chunker"""
    
    # Read the legal document
    try:
        with open('../data/01_VBHN-VPQH_363655.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("Legal document not found. Please ensure the file exists.")
        return
    
    # Initialize chunker
    chunker = LegalDocumentChunker()
    
    # Chunk by clauses only (articles become metadata)
    print("Chunking legal document by clauses...")
    clause_chunks = chunker.chunk_by_clauses_only(text)
    clause_chunks = chunker.enrich_with_footnotes(clause_chunks, text)
    print(f"Generated {len(clause_chunks)} clause-level chunks")
    
    # Use clause chunks for analysis
    chunks_to_analyze = clause_chunks
    
    # Show some examples
    print("\n" + "="*60)
    print("SAMPLE CHUNKS:")
    print("="*60)
    
    for i, chunk in enumerate(chunks_to_analyze[:5]):  # Show first 5 chunks
        print(f"\nChunk {i+1}:")
        print(f"ID: {chunk.chunk_id}")
        print(f"Type: {chunk.type}")
        print(f"Title: {chunk.title}")
        print(f"Hierarchy: {chunk.hierarchy_path}")
        print(f"Content preview: {chunk.content[:200]}...")
        if chunk.cross_references:
            print(f"Cross-references: {chunk.cross_references}")
        print("-" * 40)
    
    # Save chunks for next step
    import json
    chunks_data = []
    for chunk in chunks_to_analyze:
        chunks_data.append({
            'chunk_id': chunk.chunk_id,
            'type': chunk.type,
            'content': chunk.content,
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
            'footnotes': chunk.footnotes
        })
    
    # Save clause chunks
    with open('legal_clause_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved {len(chunks_to_analyze)} clause-level chunks to 'legal_clause_chunks.json'")


if __name__ == "__main__":
    main()
