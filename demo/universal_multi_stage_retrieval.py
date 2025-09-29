#!/usr/bin/env python3
"""
Universal Multi-stage Retrieval for Legal RAG
Tổng quát hóa cho tất cả chủ đề pháp luật, không hardcode specific domains
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from simple_vector_store import SimpleVectorStore, SearchResult


@dataclass
class RetrievalStageResult:
    """Kết quả từ một stage của retrieval"""
    stage_name: str
    results: List[SearchResult]
    query_variants: List[str]
    execution_time: float


class UniversalMultiStageRetrieval:
    """
    Universal Multi-stage Retrieval System
    Không hardcode chủ đề cụ thể, tự động detect và adapt
    """
    
    def __init__(self, vector_store: SimpleVectorStore, enable_query_enhancement: bool = True):
        self.vector_store = vector_store
        self.stage_results = []
        self.enable_query_enhancement = enable_query_enhancement
        
        # Universal legal patterns (không specific về chủ đề)
        self.legal_patterns = self._build_universal_patterns()
        self.topic_keywords = self._build_topic_keywords()
        
        # Stage weights for consensus scoring
        self.stage_weights = {
            'stage1_semantic': 1.0,      # Semantic similarity - baseline
            'stage3_keyword': 0.8,       # Keyword matching - slightly lower
            'stage5_crossref': 0.9,      # Cross-reference - important
            'stage6_exact': 2.0          # Exact metadata - highest priority
        }
        
        # Initialize LLM Query Enhancer if enabled
        self.query_enhancer = None
        if enable_query_enhancement:
            try:
                from llm_query_enhancer import LLMQueryEnhancer
                self.query_enhancer = LLMQueryEnhancer()
                print("✅ LLM Query Enhancer initialized")
            except Exception as e:
                print(f"⚠️ LLM Query Enhancer initialization failed: {e}")
                self.enable_query_enhancement = False
        
    def _build_universal_patterns(self) -> Dict[str, List[str]]:
        """Xây dựng patterns tổng quát cho pháp luật - tối ưu cho clause-level chunks"""
        
        return {
            "legal_prefixes": [
                "Bộ luật Hình sự quy định",
                "pháp luật về",
                "điều khoản",
                "quy định pháp lý",
                "theo luật định",
                "khoản",
                "điều",
                "điểm"
            ],
            
            "question_types": {
                "definition": ["là gì", "khái niệm", "định nghĩa", "hiểu như thế nào"],
                "procedure": ["thủ tục", "quy trình", "cách thức", "làm sao"],
                "penalty": ["hình phạt", "xử lý", "phạt", "tù", "tiền"],
                "clause_specific": ["khoản", "điều", "điểm", "mục"],
                "condition": ["điều kiện", "yêu cầu", "khi nào", "trường hợp"],
                "comparison": ["khác nhau", "giống", "so sánh", "phân biệt"],
                "exception": ["ngoại lệ", "trừ", "không áp dụng", "miễn"]
            },
            
            "legal_verbs": [
                "vi phạm", "xâm phạm", "thực hiện", "gây ra", "có hành vi",
                "phạm tội", "làm trái", "không tuân thủ", "vi phạm",
                "xử lý", "xét xử", "truy cứu", "kết án", "tuyên phạt",
                "miễn", "giảm", "tăng", "áp dụng", "ban hành"
            ],
            
            "legal_nouns": [
                "tội phạm", "vi phạm", "hành vi", "hậu quả", "thiệt hại",
                "chứng cứ", "bằng chứng", "căn cứ", "cơ sở", "lý do",
                "quyền", "nghĩa vụ", "trách nhiệm", "quyền lợi", "lợi ích"
            ]
        }
    
    def _build_topic_keywords(self) -> Dict[str, List[str]]:
        """Xây dựng từ khóa cho các chủ đề pháp luật (tự động detect)"""
        
        return {
            "general_responsibility": ["trách nhiệm", "xử lý", "tuổi", "năng lực"],
            
            "crimes_against_person": [
                "giết người", "cố ý gây thương tích", "hiếp dâm", "cưỡng dâm",
                "bắt cóc", "giam giữ", "đe dọa", "xúc phạm", "danh dự"
            ],
            
            "crimes_against_property": [
                "trộm cắp", "cướp giật", "cướp tài sản", "lừa đảo",
                "chiếm đoạt", "tài sản", "của cải", "tiền bạc"
            ],
            
            "drug_crimes": [
                "ma túy", "chất kích thích", "tàng trữ", "mua bán",
                "vận chuyển", "sử dụng trái phép", "chất gây nghiện"
            ],
            
            "corruption": [
                "tham nhũng", "nhận hối lộ", "đưa hối lộ", "lạm dụng chức vụ",
                "tham ô", "chiếm đoạt", "chức vụ", "quyền hạn"
            ],
            
            "environmental_crimes": [
                "môi trường", "ô nhiễm", "phá rừng", "khai thác",
                "chất thải", "tài nguyên", "sinh thái", "bảo vệ"
            ],
            
            "cybercrime": [
                "máy tính", "mạng", "thông tin", "dữ liệu", "hack",
                "truy cập trái phép", "công nghệ", "internet", "website"
            ],
            
            "economic_crimes": [
                "kinh tế", "thuế", "tài chính", "ngân hàng", "đầu tư",
                "chứng khoán", "tiền tệ", "giao dịch", "thương mại"
            ],
            
            "traffic_crimes": [
                "giao thông", "lái xe", "tai nạn", "rượu bia", "tốc độ",
                "vi phạm luật", "phương tiện", "đường bộ"
            ],
            
            "administrative_crimes": [
                "hành chính", "công vụ", "cán bộ", "viên chức", "công chức",
                "thủ tục", "giấy tờ", "chứng thực", "công quyền"
            ]
        }
    
    
    def extract_universal_keywords(self, query: str) -> Dict[str, List[str]]:
        """Extract keywords tổng quát không bias chủ đề"""
        
        keywords = {
            "legal_verbs": [],
            "legal_nouns": [],
            "numbers": [],
            "entities": [],
            "core_terms": []
        }
        
        query_lower = query.lower()
        
        # Extract legal verbs
        for verb in self.legal_patterns["legal_verbs"]:
            if verb in query_lower:
                keywords["legal_verbs"].append(verb)
        
        # Extract legal nouns  
        for noun in self.legal_patterns["legal_nouns"]:
            if noun in query_lower:
                keywords["legal_nouns"].append(noun)
        
        # Extract numbers
        numbers = re.findall(r'\d+', query)
        keywords["numbers"] = numbers
        
        # Extract potential entities (proper nouns, specific terms)
        # Simple approach: words that start with capital letters
        entities = re.findall(r'\b[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+\b', query)
        keywords["entities"] = entities
        
        # Extract core terms (split query into words, remove stop words)
        stop_words = {"của", "và", "với", "về", "trong", "từ", "đến", "có", "là", "được", "cho", "khi", "nào", "như", "thế", "sao", "gì", "bao", "nhiều"}
        words = query_lower.split()
        core_terms = [word for word in words if len(word) > 2 and word not in stop_words]
        keywords["core_terms"] = core_terms
        
        return keywords
    
    def stage1_universal_semantic(self, query: str) -> RetrievalStageResult:
        """Stage 1: Universal semantic search - không bias chủ đề"""
        
        import time
        start_time = time.time()
        
        print(f"🔍 Stage 1: Universal Semantic Search")
        print(f"   Query: '{query}'")
        
        # Direct semantic search
        results = self.vector_store.search(query, top_k=7)
        
        print(f"   → Found {len(results)} results")
        for i, result in enumerate(results[:3], 1):
            article = result.metadata.get('article', 'N/A') if result.metadata else 'N/A'
            title = result.metadata.get('title', 'N/A') if result.metadata else 'N/A'
            # Safe string slicing - handle None values
            title_safe = str(title) if title is not None else 'N/A'
            title_short = title_safe[:40] if len(title_safe) > 40 else title_safe
            print(f"   {i}. {article} - {title_short}... (score: {result.score:.3f})")
        
        execution_time = time.time() - start_time
        
        return RetrievalStageResult(
            stage_name="Universal Semantic",
            results=results,
            query_variants=[query],
            execution_time=execution_time
        )
    
    
    def stage3_keyword_extraction(self, query: str) -> RetrievalStageResult:
        """Stage 3: Universal keyword extraction và search"""
        
        import time
        start_time = time.time()
        
        print(f"🔑 Stage 3: Universal Keyword Extraction")
        
        all_results = []
        query_variants = []
        
        # Extract keywords
        keywords_dict = self.extract_universal_keywords(query)
        
        print(f"   Extracted keywords:")
        for k_type, k_list in keywords_dict.items():
            if k_list:
                print(f"   • {k_type}: {k_list}")
        
        # Search với từng loại keyword
        for keyword_type, keyword_list in keywords_dict.items():
            for keyword in keyword_list:
                if len(keyword) > 2:  # Skip very short keywords
                    results = self.vector_store.search(keyword, top_k=4)
                    all_results.extend(results)
                    query_variants.append(keyword)
                    
                    if results and results[0].score > 0.5:  # Only log good results
                        print(f"   • '{keyword}' → {len(results)} results (best: {results[0].score:.3f})")
        
        execution_time = time.time() - start_time
        
        return RetrievalStageResult(
            stage_name="Universal Keywords",
            results=all_results,
            query_variants=query_variants,
            execution_time=execution_time
        )
    
    
    def stage5_cross_reference_search(self, query: str) -> RetrievalStageResult:
        """Stage 5: Cross-reference và hierarchical search"""
        
        import time
        start_time = time.time()
        
        print(f"🔗 Stage 5: Cross-reference Search")
        
        all_results = []
        query_variants = []
        
        # A. Tìm cross-references trong query
        cross_refs = self.find_cross_references(query)
        
        if cross_refs:
            print(f"   Found cross-references: {cross_refs}")
            
            for ref in cross_refs:
                results = self.vector_store.search(ref, top_k=4)
                all_results.extend(results)
                query_variants.append(f"cross_ref:{ref}")
        
        # B. Hierarchical search (search trong cùng chapter/part)
        # Nếu đã có results từ previous stages, tìm trong cùng hierarchy
        if hasattr(self, 'previous_results') and self.previous_results:
            hierarchies = self.extract_hierarchies_from_results(self.previous_results)
            
            for hierarchy in hierarchies:
                # Search trong cùng chapter/part
                hierarchy_results = self.vector_store.search_by_metadata(hierarchy, top_k=5)
                
                # Filter by semantic relevance
                for result in hierarchy_results:
                    if self.is_semantically_relevant(query, result.content):
                        all_results.append(result)
                        query_variants.append(f"hierarchy:{hierarchy}")
        
        execution_time = time.time() - start_time
        
        return RetrievalStageResult(
            stage_name="Cross-reference",
            results=all_results,
            query_variants=query_variants,
            execution_time=execution_time
        )
    
    def stage6_exact_metadata_search(self, query: str) -> RetrievalStageResult:
        """Stage 6: Exact metadata matching - match với metadata của chunks"""
        
        import time
        start_time = time.time()
        
        print(f"\n📋 Stage 6: Exact Metadata Search")
        
        query_variants = []
        all_results = []
        
        # 1. Extract structure references from query
        structure_refs = self.extract_structure_references(query)
        print(f"   Extracted structure refs: {structure_refs}")
        
        if structure_refs:
            # Build metadata filter for exact matching
            metadata_filters = {}
            
            for ref_type, ref_value in structure_refs:
                if ref_type == "điều":
                    metadata_filters['article'] = f"Điều {ref_value}"
                    print(f"   Filtering by article: Điều {ref_value}")
                    
                elif ref_type == "khoản":
                    metadata_filters['clause'] = ref_value
                    print(f"   Filtering by clause: {ref_value}")
                    
                elif ref_type == "điểm":
                    metadata_filters['point'] = ref_value
                    print(f"   Filtering by point: {ref_value}")
                    
                elif ref_type == "chương":
                    metadata_filters['chapter'] = f"Chương {ref_value.upper()}"
                    print(f"   Filtering by chapter: Chương {ref_value.upper()}")
                    
                elif ref_type == "phần":
                    metadata_filters['part'] = f"Phần {ref_value}"
                    print(f"   Filtering by part: Phần {ref_value}")
            
            # Find chunks that match ALL metadata filters
            if metadata_filters:
                matching_chunks = self.find_chunks_by_metadata(metadata_filters)
                print(f"   Found {len(matching_chunks)} exact matches")
                
                for chunk_id in matching_chunks:
                    if chunk_id in self.vector_store.vectors:
                        # Get the chunk data
                        metadata = self.vector_store.metadata.get(chunk_id, {})
                        content = metadata.get('content', '')
                        
                        # Create SearchResult with very high score for exact matches
                        from simple_vector_store import SearchResult
                        result = SearchResult(
                            chunk_id=chunk_id,
                            content=content,
                            score=1.0,  # Maximum score for exact metadata match
                            metadata=metadata
                        )
                        all_results.append(result)
                        query_variants.append(f"exact_match:{chunk_id}")
        
        # 2. If no exact matches, fall back to semantic search
        if not all_results:
            print(f"   No exact matches found, falling back to semantic search")
            
            # Extract hierarchy terms for semantic search
            hierarchy_terms = self.extract_hierarchy_terms(query)
            print(f"   Extracted hierarchy terms: {hierarchy_terms}")
            
            for term in hierarchy_terms:
                hierarchy_query = f"{term} {query}"
                query_variants.append(hierarchy_query)
                results = self.vector_store.search(query=hierarchy_query, top_k=6, min_score=0.3)
                all_results.extend(results)
        
        execution_time = time.time() - start_time
        
        print(f"   Query variants: {len(query_variants)}")
        print(f"   Results found: {len(all_results)}")
        print(f"   Execution time: {execution_time:.3f}s")
        
        return RetrievalStageResult(
            stage_name="Exact Metadata Search",
            results=all_results,
            query_variants=query_variants,
            execution_time=execution_time
        )
    
    def find_chunks_by_metadata(self, metadata_filters: Dict[str, str]) -> List[str]:
        """Find chunks that match ALL metadata filters exactly (case-insensitive for chapter)"""
        matching_chunks = []
        
        for chunk_id, metadata in self.vector_store.metadata.items():
            # Check if this chunk matches ALL filters
            matches_all = True
            
            for filter_key, filter_value in metadata_filters.items():
                chunk_value = metadata.get(filter_key, '')
                
                # Case-insensitive matching for chapter
                if filter_key == 'chapter':
                    if chunk_value.upper() != filter_value.upper():
                        matches_all = False
                        break
                else:
                    if chunk_value != filter_value:
                        matches_all = False
                        break
            
            if matches_all:
                matching_chunks.append(chunk_id)
        
        return matching_chunks
    
    def calculate_consensus_score(self, all_results: List[SearchResult]) -> Dict[str, Dict]:
        """
        Calculate consensus-based scoring dựa trên:
        1. Stage weights
        2. Number of stages that found the result
        3. Combined confidence score
        """
        
        # Track results by chunk_id
        result_tracker = {}
        stage_names = ['stage1_semantic', 'stage3_keyword', 'stage5_crossref', 'stage6_exact']
        
        # Initialize tracking for each result
        for result in all_results:
            chunk_id = result.chunk_id
            
            if chunk_id not in result_tracker:
                result_tracker[chunk_id] = {
                    'chunk_id': chunk_id,
                    'content': result.content,
                    'metadata': result.metadata,
                    'stages_found': [],  # Which stages found this result
                    'stage_scores': {},  # Score from each stage
                    'max_stage_score': 0.0,
                    'consensus_count': 0,
                    'weighted_sum': 0.0
                }
        
        # Map stage results to stage names using index
        stage_names_list = []
        for stage_result in self.stage_results:
            if 'Universal Semantic' in stage_result.stage_name:
                stage_names_list.append('stage1_semantic')
            elif 'Universal Keywords' in stage_result.stage_name:
                stage_names_list.append('stage3_keyword')
            elif 'Cross-reference' in stage_result.stage_name:
                stage_names_list.append('stage5_crossref')
            elif 'Exact Metadata' in stage_result.stage_name:
                stage_names_list.append('stage6_exact')
            else:
                stage_names_list.append('unknown')
        
        # Count appearances in each stage
        for i, stage_result in enumerate(self.stage_results):
            stage_name = stage_names_list[i] if i < len(stage_names_list) else 'unknown'
            
            for result in stage_result.results:
                chunk_id = result.chunk_id
                
                if chunk_id in result_tracker:
                    # Add to stages found
                    if stage_name not in result_tracker[chunk_id]['stages_found']:
                        result_tracker[chunk_id]['stages_found'].append(stage_name)
                        result_tracker[chunk_id]['consensus_count'] += 1
                    
                    # Store stage score
                    result_tracker[chunk_id]['stage_scores'][stage_name] = result.score
                    result_tracker[chunk_id]['max_stage_score'] = max(
                        result_tracker[chunk_id]['max_stage_score'], 
                        result.score
                    )
                    
                    # Calculate weighted sum
                    stage_weight = self.stage_weights.get(stage_name, 1.0)
                    result_tracker[chunk_id]['weighted_sum'] += result.score * stage_weight
        
        return result_tracker
    
    def consensus_rerank(self, all_results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank results based on consensus scoring:
        1. Number of stages that found the result (consensus)
        2. Stage weights
        3. Combined confidence score
        """
        
        # Calculate consensus scores
        consensus_data = self.calculate_consensus_score(all_results)
        
        # Create new SearchResult objects with consensus scores
        consensus_results = []
        
        for chunk_id, data in consensus_data.items():
            # Calculate consensus score
            consensus_count = data['consensus_count']
            weighted_sum = data['weighted_sum']
            max_stage_score = data['max_stage_score']
            
            # Consensus factor: more stages = higher confidence
            consensus_factor = consensus_count / len(self.stage_results)  # 0.25, 0.5, 0.75, 1.0
            
            # Weight factor: sum of weighted scores
            weight_factor = weighted_sum / sum(self.stage_weights.values())  # Normalize
            
            # Final consensus score
            consensus_score = (consensus_factor * 0.4) + (weight_factor * 0.4) + (max_stage_score * 0.2)
            
            # Special bonus for exact matches (Stage 6)
            if 'stage6_exact' in data['stages_found']:
                consensus_score = max(consensus_score, 0.95)  # Minimum score for exact matches
            
            # Create new SearchResult with consensus score
            from simple_vector_store import SearchResult
            consensus_result = SearchResult(
                chunk_id=chunk_id,
                content=data['content'],
                score=consensus_score,
                metadata=data['metadata']
            )
            
            consensus_results.append(consensus_result)
        
        # Sort by consensus score
        consensus_results.sort(key=lambda x: x.score, reverse=True)
        
        return consensus_results
    
    def enhance_query(self, query: str) -> Tuple[str, Dict]:
        """
        Enhance query using LLM Query Enhancer
        Returns enhanced query and enhancement metadata
        """
        if not self.enable_query_enhancement or not self.query_enhancer:
            return query, {}
        
        try:
            enhancement = self.query_enhancer.enhance_query(query)
            
            # Return enhanced query and metadata
            enhancement_metadata = {
                'original_query': enhancement.original_query,
                'enhanced_query': enhancement.enhanced_query,
                'extracted_keywords': enhancement.extracted_keywords,
                'legal_terms': enhancement.legal_terms,
                'confidence_score': enhancement.confidence_score,
                'enhancement_type': enhancement.enhancement_type.value,
                'reasoning': enhancement.reasoning
            }
            
            return enhancement.enhanced_query, enhancement_metadata
            
        except Exception as e:
            print(f"⚠️ Query enhancement failed: {e}")
            return query, {}
    
    def extract_structure_references(self, query: str) -> List[Tuple[str, str]]:
        """Extract structure references from query (universal patterns)"""
        structure_refs = []
        
        # Universal patterns for legal documents (improved)
        patterns = [
            (r'[đd]iều\s+(\d+)', 'điều'),        # "Điều 180" hoặc "iều 180" → ("điều", "180")
            (r'khoản\s+(\d+)', 'khoản'),         # "Khoản 2" → ("khoản", "2")
            (r'điểm\s+([a-z])', 'điểm'),         # "Điểm a" → ("điểm", "a")
            (r'(\d+)\.\s*[a-z]', 'khoản'),       # "1. a)" → ("khoản", "1")
            (r'chương\s+([ivx]+)', 'chương'),    # "Chương I" → ("chương", "I")
            (r'phần\s+([ivx]+)', 'phần'),        # "Phần I" → ("phần", "I")
            (r'mục\s+(\d+)', 'mục'),             # "Mục 1" → ("mục", "1")
        ]
        
        for pattern, ref_type in patterns:
            matches = re.findall(pattern, query.lower())
            for match in matches:
                structure_refs.append((ref_type, match))
        
        return structure_refs
    
    def detect_content_patterns(self, query: str) -> List[str]:
        """Detect content patterns in query (universal approach)"""
        content_patterns = []
        
        # Universal content indicators
        content_indicators = [
            "quy định", "điều kiện", "trường hợp", "ngoại lệ", "miễn",
            "thủ tục", "quy trình", "cách thức", "xử lý", "áp dụng",
            "hiệu lực", "thi hành", "thực hiện", "tuân thủ", "phạt tiền",
            "phạt tù", "hình phạt"
        ]
        
        for indicator in content_indicators:
            if indicator in query.lower():
                content_patterns.append(indicator)
        
        return content_patterns
    
    def extract_hierarchy_terms(self, query: str) -> List[str]:
        """Extract hierarchy terms from query (universal)"""
        hierarchy_terms = []
        
        # Universal hierarchy terms
        hierarchy_indicators = ["phần", "chương", "điều", "khoản", "điểm", "mục"]
        
        for term in hierarchy_indicators:
            if term in query.lower():
                hierarchy_terms.append(term)
        
        return hierarchy_terms
    
    def find_cross_references(self, query: str) -> List[str]:
        """Tìm cross-references một cách dynamic"""
        
        cross_refs = []
        
        # Extract article/chapter references from query
        article_matches = re.findall(r'điều\s*(\d+)', query.lower())
        chapter_matches = re.findall(r'chương\s*([ivx]+)', query.lower())
        
        for article_num in article_matches:
            cross_refs.append(f"điều {article_num}")
        
        for chapter_num in chapter_matches:
            cross_refs.append(f"chương {chapter_num}")
        
        return list(set(cross_refs))  # Remove duplicates
    
    def extract_hierarchies_from_results(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        """Extract hierarchies từ previous results"""
        
        hierarchies = []
        
        for result in results[:3]:  # Top 3 results
            metadata = result.metadata
            
            if metadata.get('chapter'):
                hierarchies.append({'chapter': metadata['chapter']})
            
            if metadata.get('part'):
                hierarchies.append({'part': metadata['part']})
        
        # Remove duplicates
        unique_hierarchies = []
        for h in hierarchies:
            if h not in unique_hierarchies:
                unique_hierarchies.append(h)
        
        return unique_hierarchies
    
    def is_semantically_relevant(self, query: str, content: str, threshold: float = 0.5) -> bool:
        """Check semantic relevance giữa query và content"""
        
        # Simple approach: check keyword overlap
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        overlap = len(query_words.intersection(content_words))
        relevance_score = overlap / len(query_words) if query_words else 0
        
        return relevance_score >= threshold
    
    def deduplicate_results(self, all_results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results"""
        
        seen_chunks: Set[str] = set()
        unique_results = []
        
        for result in all_results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def multi_factor_rerank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Multi-factor reranking with exact match priority"""
        
        def calculate_comprehensive_score(result: SearchResult) -> float:
            # Priority 1: Exact metadata matches (score = 1.0)
            if result.score == 1.0:
                return 1.0
            
            score = 0.0
            
            # Factor 1: Original similarity (30%)
            score += result.score * 0.3
            
            # Factor 2: Content quality (25%)
            content_length = len(result.content)
            content_bonus = min(content_length / 500, 1.0) * 0.25
            score += content_bonus
            
            # Factor 3: Legal hierarchy importance (25%)
            hierarchy_bonus = 0.0
            if result.metadata.get('type') == 'article':
                hierarchy_bonus = 0.25
            elif result.metadata.get('chapter'):
                hierarchy_bonus = 0.15
            score += hierarchy_bonus
            
            # Factor 4: Keyword density (20%)
            query_words = set(query.lower().split())
            content_words = set(result.content.lower().split())
            
            if query_words:
                keyword_density = len(query_words.intersection(content_words)) / len(query_words)
                score += keyword_density * 0.2
            
            return min(score, 0.99)  # Cap at 0.99 for non-exact matches
        
        # Sort by comprehensive score
        reranked = sorted(results, key=calculate_comprehensive_score, reverse=True)
        
        return reranked
    
    def execute_multi_stage_retrieval(self, query: str, max_results: int = 5) -> Tuple[List[SearchResult], List[RetrievalStageResult]]:
        """Execute complete multi-stage retrieval pipeline - 4 stages only"""
        
        print(f"\n{'='*60}")
        print(f"🚀 UNIVERSAL MULTI-STAGE RETRIEVAL (4 Stages)")
        print(f"Original Query: '{query}'")
        print(f"{'='*60}")
        
        # Enhance query using LLM if enabled
        enhanced_query = query
        enhancement_metadata = {}
        if self.enable_query_enhancement:
            print(f"\n🔧 QUERY ENHANCEMENT:")
            enhanced_query, enhancement_metadata = self.enhance_query(query)
            
            if enhancement_metadata:
                print(f"   Original: {enhancement_metadata.get('original_query', query)}")
                print(f"   Enhanced: {enhancement_metadata.get('enhanced_query', query)}")
                print(f"   Keywords: {enhancement_metadata.get('extracted_keywords', [])}")
                print(f"   Legal Terms: {enhancement_metadata.get('legal_terms', [])}")
                print(f"   Confidence: {enhancement_metadata.get('confidence_score', 0):.2f}")
                print(f"   Type: {enhancement_metadata.get('enhancement_type', 'none')}")
        
        # Use enhanced query for retrieval
        retrieval_query = enhanced_query
        
        self.stage_results = []
        all_results = []
        
        # Execute 4 stages only (removed hardcoded Stage 2 and Stage 4)
        stage1_result = self.stage1_universal_semantic(retrieval_query)
        self.stage_results.append(stage1_result)
        all_results.extend(stage1_result.results)
        
        stage3_result = self.stage3_keyword_extraction(retrieval_query)
        self.stage_results.append(stage3_result)
        all_results.extend(stage3_result.results)
        
        # Store results for cross-reference stage
        self.previous_results = all_results
        
        stage5_result = self.stage5_cross_reference_search(retrieval_query)
        self.stage_results.append(stage5_result)
        all_results.extend(stage5_result.results)
        
        # Exact metadata search (new Stage 6)
        stage6_result = self.stage6_exact_metadata_search(retrieval_query)
        self.stage_results.append(stage6_result)
        all_results.extend(stage6_result.results)
        
        # Post-processing
        print(f"\n🔄 Post-processing...")
        print(f"   Total raw results: {len(all_results)}")
        
        # Deduplicate
        unique_results = self.deduplicate_results(all_results)
        print(f"   After deduplication: {len(unique_results)}")
        
        # Consensus-based rerank (NEW)
        final_results = self.consensus_rerank(unique_results)
        print(f"   After consensus reranking: {len(final_results)}")
        
        # Take top results
        top_results = final_results[:max_results]
        
        print(f"\n✅ FINAL RESULTS (Top {len(top_results)}):")
        consensus_data = self.calculate_consensus_score(unique_results)
        
        for i, result in enumerate(top_results, 1):
            article = result.metadata.get('article', 'N/A') if result.metadata else 'N/A'
            title = result.metadata.get('title', 'N/A') if result.metadata else 'N/A'
            # Safe string slicing - handle None values
            title_safe = str(title) if title is not None else 'N/A'
            title_short = title_safe[:50] if len(title_safe) > 50 else title_safe
            
            # Show consensus information
            chunk_data = consensus_data.get(result.chunk_id, {})
            consensus_count = chunk_data.get('consensus_count', 0)
            stages_found = chunk_data.get('stages_found', [])
            
            print(f"   {i}. {article} - {title_short}... (score: {result.score:.3f})")
            print(f"      🎯 Consensus: {consensus_count}/4 stages | Stages: {', '.join(stages_found)}")
        
        # Execution summary
        total_time = sum(stage.execution_time for stage in self.stage_results)
        print(f"\n📊 Execution Summary:")
        print(f"   Total execution time: {total_time:.3f}s")
        for stage in self.stage_results:
            print(f"   • {stage.stage_name}: {stage.execution_time:.3f}s ({len(stage.results)} results)")
        
        return top_results, self.stage_results


def main():
    """Test Universal Multi-stage Retrieval với các chủ đề khác nhau"""
    
    # Load vector store
    from simple_vector_store import SimpleVectorStore
    import os
    
    vector_store = SimpleVectorStore()
    if not vector_store.load('legal_vectors.pkl'):
        print("❌ Không thể load vector store")
        return
    
    # Initialize Universal Multi-stage Retrieval
    umsr = UniversalMultiStageRetrieval(vector_store)
    
    # Test với các queries phù hợp với clause-level chunks
    test_queries = [
        "Điều 180 phạt tiền bao nhiêu?",  # Article-specific
        "Khoản 2 Điều 225 quy định gì?",  # Clause-specific
        "Điều 3 khoản 1 về trách nhiệm hình sự",  # Article + clause
        "phạt tiền từ 50 triệu đến 300 triệu",  # Content-specific
        "người dưới 16 tuổi có bị truy cứu trách nhiệm hình sự không"  # General legal question
    ]
    
    for query in test_queries:
        print(f"\n" + "="*80)
        print(f"🧪 TESTING QUERY: '{query}'")
        print(f"="*80)
        
        try:
            results, stage_results = umsr.execute_multi_stage_retrieval(query, max_results=3)
            
            print(f"\n📋 DETAILED RESULTS:")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Article: {result.metadata.get('article', 'N/A')}")
                print(f"  Title: {result.metadata.get('title', 'N/A')}")
                print(f"  Score: {result.score:.3f}")
                print(f"  Content preview: {result.content[:150]}...")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        input("\nPress Enter to continue to next query...")


if __name__ == "__main__":
    main()
