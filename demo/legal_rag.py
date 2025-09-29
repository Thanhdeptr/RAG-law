#!/usr/bin/env python3
"""
Legal RAG System
Hệ thống RAG chuyên dụng cho văn bản pháp luật Việt Nam
Dựa trên ConvoRAG nhưng được tối ưu cho legal domain
"""

import heapq
import numpy as np
import ollama
from typing import List, Dict, Tuple, Any, Optional
import json
import os
from simple_vector_store import SimpleVectorStore, SearchResult
from universal_multi_stage_retrieval import UniversalMultiStageRetrieval
from legal_chunker import FootnoteLookup


class LegalRAG:
    """
    Legal RAG System với khả năng:
    - Hiểu context pháp luật
    - Trích dẫn chính xác điều khoản
    - Xử lý câu hỏi phức tạp về luật
    - Conversation history cho legal consultation
    """
    
    def __init__(self, 
                 vector_store_path: str = "legal_vectors.pkl",
                 llm_model: str = "llama3.2:3b",
                 ollama_host: str = "http://192.168.10.32:11434",
                 debug_mode: bool = False):
        
        self.llm_model = llm_model
        self.ollama_client = ollama.Client(host=ollama_host)
        self.debug_mode = debug_mode
        
        # Load vector store and initialize Universal Multi-stage Retrieval
        self.vector_store = SimpleVectorStore(ollama_host=ollama_host)
        if os.path.exists(vector_store_path):
            self.vector_store.load(vector_store_path)
            print(f"✅ Loaded legal vector store with {len(self.vector_store.vectors)} chunks")
            
            # Initialize Universal Multi-stage Retrieval as the ONLY retrieval system
            self.retrieval_system = UniversalMultiStageRetrieval(self.vector_store)
            print(f"✅ Retrieval System: Universal Multi-stage Retrieval")
        else:
            raise FileNotFoundError(f"Vector store not found: {vector_store_path}. Please run simple_vector_store.py first.")
            
        # Conversation history for legal consultation
        self.conversation_history = []
        
        # Initialize footnote lookup
        try:
            with open('../data/01_VBHN-VPQH_363655.txt', 'r', encoding='utf-8') as f:
                self.legal_text = f.read()
            self.footnote_lookup = FootnoteLookup(self.legal_text)
            print(f"✅ Loaded footnote lookup with {len(self.footnote_lookup.get_all_footnotes())} footnotes")
        except FileNotFoundError:
            print("⚠️ Warning: Legal document not found for footnote lookup")
            self.footnote_lookup = None
    
    def debug_print(self, message: str):
        """Print debug message if debug mode is enabled"""
        if self.debug_mode:
            print(message)
    
    def preprocess_legal_query(self, query: str) -> str:
        """Preprocess user query to standardize legal terminology and fix common issues"""
        
        self.debug_print(f"   🔧 Debug - Original query: '{query}'")
        
        try:
            system_prompt = """
            Bạn là chuyên gia xử lý ngôn ngữ pháp luật Việt Nam. Nhiệm vụ của bạn là chuẩn hóa câu hỏi của người dùng để tìm kiếm chính xác trong cơ sở dữ liệu pháp luật.

            NHIỆM VỤ:
            1. **Sửa lỗi chính tả** (ví dụ: "ggiay" → "giấy", "lam gia" → "làm giả")
            2. **Chuẩn hóa thuật ngữ pháp luật**:
               - "làm giả giấy tờ" → "làm giả tài liệu"
               - "chiếm đoạt" → "cưỡng đoạt tài sản" 
               - "giết người" → "tội giết người"
               - "trộm cắp" → "tội trộm cắp tài sản"
            3. **Chuyển đổi số tiền sang định dạng chuẩn**:
               - "700 triệu" → "700.000.000 đồng"
               - "1 tỷ" → "1.000.000.000 đồng"
               - "50 ngàn" → "50.000 đồng"
               - "2 trăm triệu" → "200.000.000 đồng"
            4. **Chuẩn hóa cấu trúc câu hỏi**:
               - Thêm "tội" trước tên tội phạm nếu cần
               - Sử dụng thuật ngữ chính thống trong Bộ luật Hình sự

            QUY TẮC:
            - GIỮ NGUYÊN ý nghĩa gốc của câu hỏi
            - CHỈ chuẩn hóa thuật ngữ và sửa lỗi
            - SỬ DỤNG thuật ngữ chính xác trong Bộ luật Hình sự Việt Nam
            - TRẢ LỜI chỉ câu hỏi đã được chuẩn hóa, không giải thích

            VÍ DỤ:
            Input: "toi lam gia ggiay to tuy than thi hinh phat la gi"
            Output: "tôi làm giả tài liệu tùy thân thì hình phạt là gì"

            Input: "chiem doat 700 trieu co bi tu hinh khong"  
            Output: "cưỡng đoạt tài sản trị giá 700.000.000 đồng có bị tử hình không"
            """
            
            user_prompt = f'Chuẩn hóa câu hỏi: "{query}"'
            
            # Use low temperature for consistent preprocessing
            preprocessed = self.generate_answer(system_prompt, user_prompt, temperature=0.1).strip()
            
            # Clean up response - remove quotes and extra text
            if preprocessed.startswith('"') and preprocessed.endswith('"'):
                preprocessed = preprocessed[1:-1]
            
            # Remove common prefixes from LLM response
            prefixes_to_remove = [
                "Câu hỏi đã chuẩn hóa:",
                "Output:",
                "Kết quả:",
                "Chuẩn hóa:"
            ]
            
            for prefix in prefixes_to_remove:
                if preprocessed.startswith(prefix):
                    preprocessed = preprocessed[len(prefix):].strip()
            
            # Fallback to original if preprocessing seems wrong
            if (len(preprocessed) > len(query) * 2 or 
                len(preprocessed) < len(query) * 0.3 or
                "lỗi" in preprocessed.lower()):
                self.debug_print(f"   ⚠️ Preprocessing failed, using original query")
                return query
            
            self.debug_print(f"   ✅ Preprocessed query: '{preprocessed}'")
            return preprocessed
            
        except Exception as e:
            self.debug_print(f"   ❌ Preprocessing error: {str(e)}")
            return query  # Fallback to original query
        
    def enrich_results_with_footnotes(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """Bổ sung thông tin footnote vào search results"""
        if not self.footnote_lookup or not search_results:
            return search_results
            
        enriched_results = []
        for result in search_results:
            # Tạo copy của result để không modify original
            enriched_result = SearchResult(
                content=result.content,
                score=result.score,
                metadata=result.metadata.copy() if result.metadata else {}
            )
            
            # Tìm footnote trong content
            import re
            footnote_matches = re.findall(r'\[(\d+)\]', result.content)
            
            if footnote_matches:
                footnote_info = []
                for footnote_num in set(footnote_matches):  # Remove duplicates
                    footnote_content = self.footnote_lookup.lookup_footnote(footnote_num)
                    if footnote_content:
                        footnote_info.append(f"[{footnote_num}] {footnote_content}")
                
                if footnote_info:
                    # Thêm footnote info vào metadata
                    enriched_result.metadata['footnotes'] = footnote_info
                    
                    # Thêm footnote vào cuối content
                    enriched_result.content += "\n\n📖 CHÚ THÍCH:\n" + "\n".join(footnote_info)
            
            enriched_results.append(enriched_result)
        
        return enriched_results

    def search_legal_context(self, query: str, top_k: int = 10)-> Tuple[str, List[SearchResult]]:
        """Search for relevant legal context using Universal Multi-stage Retrieval"""
        
        # Execute Universal Multi-stage Retrieval
        results, stage_results = self.retrieval_system.execute_multi_stage_retrieval(query, max_results=top_k)
        
        # Enrich results with footnotes
        results = self.enrich_results_with_footnotes(results)
        
        # Format context for LLM with comprehensive error handling
        context_parts = []
        for i, result in enumerate(results):
            try:
                # Debug result structure
                self.debug_print(f"   🔍 Debug - Result {i}: type={type(result)}")
                if hasattr(result, '__dict__'):
                    self.debug_print(f"   🔍 Debug - Result {i} attrs: {list(result.__dict__.keys())}")
                
                # Safe result access
                if not hasattr(result, 'metadata'):
                    self.debug_print(f"   ⚠️ Warning - Result {i} has no metadata attribute")
                    article, title, hierarchy = 'N/A', 'Không có tiêu đề', ''
                else:
                    metadata = result.metadata
                    if metadata is None:
                        self.debug_print(f"   ⚠️ Warning - Result {i} metadata is None")
                        article, title, hierarchy = 'N/A', 'Không có tiêu đề', ''
                    elif not isinstance(metadata, dict):
                        self.debug_print(f"   ⚠️ Warning - Result {i} metadata not dict: {type(metadata)}")
                        article, title, hierarchy = 'N/A', 'Không có tiêu đề', ''
                    else:
                        # Safe metadata extraction with None checks
                        article = metadata.get('article', 'N/A') if metadata else 'N/A'
                        title = metadata.get('title', '') if metadata else ''
                        hierarchy = metadata.get('hierarchy_path', '') if metadata else ''
                
                # Safe content access
                if not hasattr(result, 'content'):
                    self.debug_print(f"   ⚠️ Warning - Result {i} has no content attribute")
                    content = 'No content available'
                else:
                    content = result.content if result.content else 'No content available'
                
                # Handle None title
                title_display = title if title else 'Không có tiêu đề'
                
            except Exception as e:
                self.debug_print(f"   ❌ Error processing result {i}: {str(e)}")
                article, title_display, hierarchy, content = 'N/A', 'Lỗi xử lý', '', 'Error processing result'
            
            formatted_context = f"""
=== {article}. {title_display} ===
[{hierarchy}]

{content}

---
"""
            context_parts.append(formatted_context.strip())
        
        combined_context = '\n\n'.join(context_parts)
        return combined_context, results
    
    
    def detect_query_type(self, query: str) -> str:
        """Classify legal query type using LLM with rule-based fallback"""
        
        # Try LLM-based classification first
        try:
            system_prompt = """
            Bạn là chuyên gia phân loại câu hỏi pháp luật. Hãy phân loại câu hỏi vào một trong các loại:
            
            1. "legal-specific" - Câu hỏi cụ thể về điều luật, hình phạt, thủ tục pháp lý
               Ví dụ: "Điều 40 quy định gì?", "Hình phạt tử hình áp dụng như thế nào?"
            
            2. "legal-general" - Câu hỏi chung về khái niệm pháp luật
               Ví dụ: "Trách nhiệm hình sự là gì?", "Phân loại tội phạm như thế nào?"
            
            3. "legal-consultation" - Câu hỏi tư vấn, áp dụng luật vào tình huống cụ thể  
               Ví dụ: "Người 15 tuổi có bị xử lý hình sự không?", "Trường hợp nào được miễn tù?"
            
            4. "non-legal" - Câu hỏi không liên quan đến pháp luật
               Ví dụ: "Thời tiết hôm nay thế nào?", "Cách nấu phở"
            
            Chỉ trả lời tên loại, không giải thích.
            """
            
            user_prompt = f'Phân loại câu hỏi: "{query}"'
            
            # Try LLM classification with short timeout
            response = self.generate_answer(system_prompt, user_prompt, temperature=0.1).lower().strip()
            
            # Check if response contains error message
            if "lỗi" in response.lower():
                raise Exception("LLM classification failed")
            
            if "legal-specific" in response:
                return "legal-specific"
            elif "legal-general" in response:
                return "legal-general" 
            elif "legal-consultation" in response:
                return "legal-consultation"
            elif "non-legal" in response:
                return "non-legal"
            else:
                # If LLM response is unclear, fallback to rule-based
                raise Exception("Unclear LLM response")
                
        except Exception as e:
            self.debug_print(f"   ⚠️ LLM classification failed ({str(e)[:50]}...), using rule-based fallback")
            return self._rule_based_classification(query)
    
    def _rule_based_classification(self, query: str) -> str:
        """Rule-based fallback for query classification"""
        
        query_lower = query.lower()
        
        # Legal-specific patterns
        specific_patterns = [
            r'điều\s+\d+', r'khoản\s+\d+', r'chương\s+[ivx]+',
            'quy định gì', 'nội dung', 'điều khoản', 'luật định'
        ]
        
        # Legal-general patterns  
        general_patterns = [
            'là gì', 'khái niệm', 'định nghĩa', 'hiểu như thế nào',
            'phân loại', 'các loại', 'nguyên tắc'
        ]
        
        # Legal-consultation patterns
        consultation_patterns = [
            'tôi', 'mình', 'có bị', 'có phải', 'trường hợp', 'nếu như',
            'có thể', 'được không', 'bị xử lý', 'phạm tội', 'vi phạm'
        ]
        
        # Non-legal patterns
        non_legal_patterns = [
            'thời tiết', 'ăn uống', 'du lịch', 'mua sắm', 'giải trí',
            'thể thao', 'âm nhạc', 'phim ảnh'
        ]
        
        # Check patterns
        import re
        
        # Check non-legal first
        if any(pattern in query_lower for pattern in non_legal_patterns):
            return "non-legal"
        
        # Check specific legal
        if any(re.search(pattern, query_lower) for pattern in specific_patterns):
            return "legal-specific"
        
        # Check general legal
        if any(pattern in query_lower for pattern in general_patterns):
            return "legal-general"
        
        # Check consultation
        if any(pattern in query_lower for pattern in consultation_patterns):
            return "legal-consultation"
        
        # Default for legal queries
        return "legal-consultation"
    
    def generate_answer(self, system_prompt: str, user_prompt: str, 
                       temperature: float = 0.1) -> str:
        """Generate response using LLM with controlled creativity"""
        try:
            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={
                    'temperature': temperature,  # Low for legal accuracy
                    'top_p': 0.3,               # Conservative token selection
                    'top_k': 20,                # Limited vocabulary choices
                    'repeat_penalty': 1.1       # Avoid repetition
                }
            )
            
            # Safe response extraction with comprehensive error checking
            if response is None:
                return "Lỗi: Không nhận được response từ LLM"
            
            # Debug: Print response structure
            self.debug_print(f"   🔍 Debug - Response type: {type(response)}")
            
            # Handle Ollama ChatResponse object
            if hasattr(response, 'message'):
                # Ollama ChatResponse object
                message = response.message
                if message is None:
                    return "Lỗi: Response message là None"
                
                if hasattr(message, 'content'):
                    content = message.content
                    if content is None:
                        return "Lỗi: Message content là None"
                    return str(content)
                else:
                    return f"Lỗi: Message không có content attribute. Available attrs: {dir(message)}"
                    
            elif isinstance(response, dict):
                # Dict response format
                self.debug_print(f"   🔍 Debug - Response keys: {list(response.keys())}")
                
                if 'message' not in response:
                    return f"Lỗi: Response thiếu 'message' field. Available keys: {list(response.keys())}"
                
                message = response['message']
                if message is None:
                    return "Lỗi: Message field là None"
                
                if not isinstance(message, dict):
                    return f"Lỗi: Message không phải dict: {type(message)} - {str(message)[:200]}"
                
                if 'content' not in message:
                    return f"Lỗi: Message thiếu 'content' field. Available keys: {list(message.keys())}"
                
                content = message['content']
                if content is None:
                    return "Lỗi: Content là None"
                
                if not isinstance(content, str):
                    return f"Lỗi: Content không phải string: {type(content)} - {str(content)[:200]}"
                    
                return content
            else:
                return f"Lỗi: Response format không được hỗ trợ: {type(response)} - {str(response)[:200]}"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"Lỗi khi tạo câu trả lời: {str(e)}\nChi tiết: {error_details}"
    
    def contextualize_query(self, current_query: str) -> str:
        """Enhance query with conversation history"""
        
        if not self.conversation_history:
            return current_query
        
        # If query is very detailed already, don't modify
        if len(current_query.split()) > 15:
            return current_query
        
        # Build conversation context
        history_context = "Lịch sử trao đổi về pháp luật:\n\n"
        for idx, (q, a) in enumerate(self.conversation_history[-3:]):  # Last 3 exchanges
            history_context += f"Câu hỏi {idx+1}: {q}\nTrả lời: {a[:200]}...\n\n"
        
        system_prompt = """
        Bạn là chuyên gia pháp luật. Nhiệm vụ của bạn là phân tích câu hỏi hiện tại trong bối cảnh 
        cuộc trao đổi trước đó và quyết định:
        
        1. Câu hỏi hiện tại có cần thêm context từ lịch sử không?
        2. Nếu có, hãy viết lại câu hỏi để nó độc lập và đầy đủ
        3. Nếu không, giữ nguyên câu hỏi
        
        QUAN TRỌNG:
        - Chỉ sử dụng thông tin có trong lịch sử trao đổi
        - Không thêm thông tin không có
        - Giữ câu hỏi ngắn gọn (dưới 25 từ)
        - Chỉ trả lời câu hỏi đã được viết lại, không giải thích
        """
        
        user_prompt = f"""
        Lịch sử trao đổi:
        {history_context}
        
        Câu hỏi hiện tại: "{current_query}"
        
        Câu hỏi được viết lại:
        """
        
        reformulated_query = self.generate_answer(system_prompt, user_prompt).strip()
        
        # Clean up response
        if ":" in reformulated_query:
            reformulated_query = reformulated_query.split(":", 1)[1].strip()
        
        # Fallback to original if reformulation seems wrong
        if (len(reformulated_query.split()) > 30 or 
            "dựa trên" in reformulated_query.lower() or
            reformulated_query.lower() == current_query.lower()):
            return current_query
        
        return reformulated_query
    
    def handle_non_legal_query(self, query: str) -> str:
        """Handle non-legal queries"""
        
        system_prompt = """
        Bạn là trợ lý AI chuyên về pháp luật Việt Nam. Người dùng vừa hỏi một câu hỏi không liên quan đến pháp luật.
        
        Hãy trả lời lịch sự và hướng dẫn người dùng quay lại chủ đề pháp luật.
        Giữ câu trả lời ngắn gọn (2-3 câu).
        """
        
        user_prompt = f'Câu hỏi của người dùng: "{query}"'
        
        return self.generate_answer(system_prompt, user_prompt)
    
    def legal_rag(self, query: str) -> str:
        """Main RAG function for legal queries with comprehensive error handling"""
        
        try:
            self.debug_print(f"   🔍 Debug - Starting legal_rag with query: '{query[:50]}...'")
            
            # Detect query type
            self.debug_print(f"   🔍 Debug - Detecting query type...")
            query_type = self.detect_query_type(query)
            self.debug_print(f"   🔍 Debug - Query type: {query_type}")
            
            if query_type == "non-legal":
                response = self.handle_non_legal_query(query)
                self.conversation_history.append((query, response))
                return response
            
            # Contextualize query with conversation history
            self.debug_print(f"   🔍 Debug - Contextualizing query...")
            contextualized_query = self.contextualize_query(query)
            self.debug_print(f"   🔍 Debug - Contextualized query: '{contextualized_query[:50]}...'")
            
            # Preprocess query to standardize legal terminology
            self.debug_print(f"   🔍 Debug - Preprocessing legal query...")
            preprocessed_query = self.preprocess_legal_query(contextualized_query)
            
            # Search for relevant legal context
            self.debug_print(f"   🔍 Debug - Searching for legal context...")
            context, search_results = self.search_legal_context(preprocessed_query)
            self.debug_print(f"   🔍 Debug - Search completed. Results: {len(search_results) if search_results else 0}")
            
            if not search_results:
                response = "Tôi không tìm thấy thông tin liên quan trong Bộ luật Hình sự. Vui lòng hỏi câu hỏi khác hoặc cung cấp thêm chi tiết."
                self.conversation_history.append((query, response))
                return response
            
            # Generate answer based on query type with appropriate temperature
            self.debug_print(f"   🔍 Debug - Setting up prompts and temperature...")
            if query_type == "legal-specific":
                system_prompt = self._get_specific_legal_prompt()
                temperature = 0.1  # Very conservative for specific legal facts
            elif query_type == "legal-general":
                system_prompt = self._get_general_legal_prompt()
                temperature = 0.2  # Slightly more flexible for explanations
            else:  # legal-consultation
                system_prompt = self._get_consultation_legal_prompt()
                temperature = 0.2  # More flexibility for consultation advice
            
            self.debug_print(f"   🔍 Debug - Temperature: {temperature}")
            
            user_prompt = f"""
            Dựa trên các điều khoản pháp luật sau đây, hãy trả lời câu hỏi:

            ĐIỀU KHOẢN LIÊN QUAN:
            {context}

            CÂU HỎI GỐC: {query}
            CÂU HỎI ĐÃ CHUẨN HÓA: {preprocessed_query}

            TRẢ LỜI:
            """
            
            self.debug_print(f"   🔍 Debug - Generating answer...")
            answer = self.generate_answer(system_prompt, user_prompt, temperature)
            self.debug_print(f"   🔍 Debug - Answer generated. Length: {len(answer) if answer else 0}")
            
            if not answer or "Lỗi" in answer:
                return f"❌ Lỗi khi tạo câu trả lời: {answer[:200]}..." if answer else "❌ Không nhận được câu trả lời từ hệ thống"
            
            # Add citation information
            self.debug_print(f"   🔍 Debug - Formatting citations...")
            citations = self._format_citations(search_results)
            if citations:
                answer += f"\n\n📖 Căn cứ pháp lý:\n{citations}"
            
            # Add confidence and relevance metrics
            confidence_info = self._format_confidence_metrics(search_results, preprocessed_query, query)
            answer += f"\n\n{confidence_info}"
            
            # Update conversation history
            self.conversation_history.append((query, answer))
            
            self.debug_print(f"   ✅ Debug - Legal RAG completed successfully")
            return answer
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.debug_print(f"   ❌ Critical error in legal_rag: {str(e)}")
            self.debug_print(f"   ❌ Full traceback: {error_details}")
            return f"❌ Lỗi hệ thống: {str(e)}\n\nVui lòng thử lại hoặc liên hệ hỗ trợ kỹ thuật."
    
    def _get_specific_legal_prompt(self) -> str:
        """System prompt for specific legal queries"""
        return """
        Bạn là chuyên gia pháp luật hình sự Việt Nam. Nhiệm vụ của bạn là trả lời chính xác 
        các câu hỏi cụ thể về điều luật.

        NGUYÊN TẮC:
        - Trích dẫn chính xác điều, khoản, điểm
        - Giải thích rõ ràng, dễ hiểu
        - Nêu đầy đủ nội dung quy định
        - Không bịa đặt thông tin không có trong điều khoản

        Trả lời bằng tiếng Việt, ngắn gọn nhưng đầy đủ thông tin.
        """
    
    def _get_general_legal_prompt(self) -> str:
        """System prompt for general legal queries"""
        return """
        Bạn là chuyên gia pháp luật hình sự Việt Nam. Nhiệm vụ của bạn là giải thích 
        các khái niệm và nguyên tắc pháp luật một cách dễ hiểu.

        NGUYÊN TẮC:
        - Giải thích khái niệm một cách rõ ràng
        - Đưa ra ví dụ minh họa khi cần thiết  
        - Liên kết các điều khoản liên quan
        - Sử dụng ngôn ngữ dễ hiểu cho người dân

        Trả lời bằng tiếng Việt, có cấu trúc rõ ràng.
        """
    
    def _get_consultation_legal_prompt(self) -> str:
        """System prompt for legal consultation queries"""
        return """
        Bạn là chuyên gia tư vấn pháp luật hình sự Việt Nam. Nhiệm vụ của bạn là 
        phân tích tình huống và đưa ra lời khuyên dựa trên pháp luật.

        NGUYÊN TẮC:
        - Phân tích tình huống cụ thể
        - Áp dụng đúng điều khoản pháp luật
        - Đưa ra lời khuyên thực tiễn
        - Cảnh báo về hậu quả pháp lý nếu có
        - Khuyến nghị tìm tư vấn chuyên sâu khi cần

        LƯU Ý: Đây chỉ là tư vấn sơ bộ, không thay thế tư vấn pháp lý chuyên nghiệp.
        
        Trả lời bằng tiếng Việt, có cấu trúc: Phân tích - Kết luận - Khuyến nghị.
        """
    
    def _format_citations(self, search_results: List[SearchResult]) -> str:
        """Format legal citations with comprehensive error handling"""
        
        citations = []
        for i, result in enumerate(search_results[:3]):  # Top 3 results
            try:
                # Debug result structure
                self.debug_print(f"   🔍 Debug - Citation {i}: type={type(result)}")
                
                # Safe metadata extraction with comprehensive checks
                if not hasattr(result, 'metadata'):
                    self.debug_print(f"   ⚠️ Warning - Citation {i} has no metadata attribute")
                    article, title, hierarchy = 'N/A', '', ''
                else:
                    metadata = result.metadata
                    if metadata is None:
                        self.debug_print(f"   ⚠️ Warning - Citation {i} metadata is None")
                        article, title, hierarchy = 'N/A', '', ''
                    elif not isinstance(metadata, dict):
                        self.debug_print(f"   ⚠️ Warning - Citation {i} metadata not dict: {type(metadata)}")
                        article, title, hierarchy = 'N/A', '', ''
                    else:
                        article = metadata.get('article', 'N/A') if 'article' in metadata else 'N/A'
                        title = metadata.get('title', '') if 'title' in metadata else ''
                        hierarchy = metadata.get('hierarchy_path', '') if 'hierarchy_path' in metadata else ''
                
                # Safe string formatting
                article_str = str(article) if article is not None else 'N/A'
                title_str = str(title) if title is not None else ''
                hierarchy_str = str(hierarchy) if hierarchy is not None else ''
                
                if title_str and title_str.strip():
                    citation = f"• {article_str}. {title_str} ({hierarchy_str})"
                else:
                    citation = f"• {article_str} ({hierarchy_str})"
                    
                citations.append(citation)
                
            except Exception as e:
                self.debug_print(f"   ❌ Error formatting citation {i}: {str(e)}")
                citations.append(f"• Lỗi xử lý trích dẫn {i}")
        
        return '\n'.join(citations) if citations else "• Không có trích dẫn"
    
    def _format_confidence_metrics(self, search_results: List[SearchResult], 
                                 preprocessed_query: str, original_query: str) -> str:
        """Format confidence and relevance metrics for answer reliability assessment"""
        
        if not search_results:
            return "📊 **Độ tin cậy: 0% - Không tìm thấy tài liệu liên quan**"
        
        try:
            # Calculate metrics
            scores = [result.score for result in search_results[:3]]  # Top 3 results
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            # Confidence level based on average score
            if avg_score >= 0.8:
                confidence_level = "Rất cao"
                confidence_color = "🟢"
                confidence_percent = int(avg_score * 100)
            elif avg_score >= 0.7:
                confidence_level = "Cao" 
                confidence_color = "🟡"
                confidence_percent = int(avg_score * 100)
            elif avg_score >= 0.6:
                confidence_level = "Trung bình"
                confidence_color = "🟠"
                confidence_percent = int(avg_score * 100)
            else:
                confidence_level = "Thấp"
                confidence_color = "🔴"
                confidence_percent = int(avg_score * 100)
            
            # Query preprocessing effectiveness
            preprocessing_effective = preprocessed_query != original_query
            preprocessing_status = "✅ Đã chuẩn hóa" if preprocessing_effective else "➖ Không cần"
            
            # Results consistency (how close are the top scores)
            if len(scores) > 1:
                score_range = max_score - min_score
                if score_range <= 0.1:
                    consistency = "Cao (kết quả nhất quán)"
                elif score_range <= 0.2:
                    consistency = "Trung bình"
                else:
                    consistency = "Thấp (kết quả phân tán)"
            else:
                consistency = "N/A"
            
            # Format confidence info
            confidence_info = f"""📊 **Đánh giá độ tin cậy câu trả lời:**
{confidence_color} **Độ tin cậy tổng thể:** {confidence_level} ({confidence_percent}%)
📈 **Điểm số tìm kiếm:** Cao nhất: {max_score:.3f} | Trung bình: {avg_score:.3f} | Thấp nhất: {min_score:.3f}
🔧 **Xử lý câu hỏi:** {preprocessing_status}
🎯 **Tính nhất quán:** {consistency}
📋 **Số tài liệu tham khảo:** {len(search_results)} điều khoản

💡 **Khuyến nghị:** {"Câu trả lời có độ tin cậy cao, có thể tham khảo." if avg_score >= 0.7 else "Nên tham khảo thêm ý kiến chuyên gia hoặc tra cứu thêm tài liệu khác."}"""

            return confidence_info
            
        except Exception as e:
            self.debug_print(f"   ❌ Error formatting confidence metrics: {str(e)}")
            return "📊 **Đánh giá độ tin cậy:** Không thể tính toán (lỗi hệ thống)"
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        
        return {
            "total_exchanges": len(self.conversation_history),
            "vector_store_chunks": len(self.vector_store.vectors),
            "llm_model": self.llm_model,
            "embedding_model": self.vector_store.embedding_model,
            "retrieval_system": "Universal Multi-stage Retrieval"
        }


def main():
    """Interactive legal consultation system"""
    
    import sys
    
    # Check for debug mode argument
    debug_mode = '--debug' in sys.argv or '-d' in sys.argv
    
    # Initialize Legal RAG
    print("🏛️  Khởi tạo Hệ thống Tư vấn Pháp luật Hình sự...")
    if debug_mode:
        print("🔧 Debug mode: ENABLED")
    try:
        legal_rag = LegalRAG(debug_mode=debug_mode)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {e}")
        return
    
    print("\n" + "="*60)
    print("🏛️  HỆ THỐNG TƯ VẤN PHÁP LUẬT HÌNH SỰ VIỆT NAM")
    print("="*60)
    print("📚 Dựa trên: Bộ luật Hình sự số 100/2015/QH13")
    print("🤖 LLM: Llama3.2:3b | Embedding: nomic-embed-text | Retrieval: Universal Multi-stage")
    print("💡 Gõ 'exit' để thoát, 'stats' để xem thống kê")
    if debug_mode:
        print("🔧 Debug logs: ON | Sử dụng 'python legal_rag.py' để tắt debug")
    else:
        print("🔧 Debug logs: OFF | Sử dụng 'python legal_rag.py --debug' để bật debug")
    print("="*60)
    
    while True:
        print("\n" + "-"*50)
        user_query = input("❓ Câu hỏi pháp luật của bạn: ").strip()
        
        if not user_query:
            continue
            
        if user_query.lower() in ['exit', 'quit', 'thoát']:
            print("\n👋 Cảm ơn bạn đã sử dụng hệ thống tư vấn pháp luật!")
            break
            
        if user_query.lower() in ['stats', 'thống kê']:
            stats = legal_rag.get_conversation_stats()
            print("\n📊 THỐNG KÊ HỆ THỐNG:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
            continue
        
        
        # Process legal query
        print("\n🔍 Đang tìm kiếm thông tin pháp luật...")
        try:
            answer = legal_rag.legal_rag(user_query)
            print(f"\n⚖️  Trả lời:\n{answer}")
        except Exception as e:
            print(f"\n❌ Lỗi: {str(e)}")
        
        print("-"*50)
        continue_query = input("\n❓ Bạn có câu hỏi khác không? (y/n): ").strip().lower()
        if continue_query not in ['y', 'yes', 'có']:
            print("\n👋 Cảm ơn bạn đã sử dụng hệ thống tư vấn pháp luật!")
            break


if __name__ == "__main__":
    main()
