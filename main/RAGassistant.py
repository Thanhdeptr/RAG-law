#!/usr/bin/env python3
"""
RAG Assistant - Complete Legal RAG System
Orchestrator for Legal Question Answering with LLM + Retrieval
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json

from RAGembedding import ContextAwareVectorStore, SearchResult
from RAGretrieval import AdvancedRetrievalSystem, RetrievalConfig


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QueryAnalysis:
    """Query analysis result"""
    original_query: str
    normalized_query: str
    query_type: str  # "legal" or "non-legal"
    confidence: float


@dataclass
class RAGResponse:
    """Complete RAG response"""
    answer: str
    query_analysis: QueryAnalysis
    retrieved_chunks: List[SearchResult]
    citations: List[str]
    confidence_score: float


# ============================================================================
# QUERY NORMALIZER
# ============================================================================

class QueryNormalizer:
    """Normalize user queries using LLM"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def normalize(self, query: str) -> str:
        """Normalize user query"""
        system_prompt = """Bạn là chuyên gia chuẩn hóa câu hỏi pháp luật. 
Nhiệm vụ: Chuẩn hóa câu hỏi của người dùng thành câu hỏi rõ ràng, chính xác về mặt pháp lý.

Quy tắc:
1. Sửa lỗi chính tả
2. Chuẩn hóa thuật ngữ pháp luật
3. Làm rõ ý nghĩa câu hỏi
4. Giữ nguyên ý định của người dùng

Chỉ trả lời câu hỏi đã được chuẩn hóa, không giải thích thêm."""

        user_prompt = f"Câu hỏi cần chuẩn hóa: {query}"
        
        try:
            normalized = self._generate(system_prompt, user_prompt, temperature=0.1, max_tokens=128)
            return normalized
        except Exception as e:
            print(f"⚠️ LLM normalization failed: {e}")
            return query  # Fallback to original
    
    def _generate(self, system_prompt: str, user_prompt: str, 
                  temperature: float = 0.1, max_tokens: int = 256) -> str:
        """Generate response using LLM"""
        
        input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,
                num_beams=1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()


# ============================================================================
# QUERY CLASSIFIER
# ============================================================================

class QueryClassifier:
    """Classify queries as legal or non-legal"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def classify(self, query: str) -> Tuple[str, float]:
        """Classify query as legal or non-legal"""
        
        system_prompt = """Bạn là chuyên gia phân loại câu hỏi pháp luật.
Nhiệm vụ: Phân loại câu hỏi thành 2 loại:
- "legal": Câu hỏi về pháp luật, luật, quy định, tòa án, luật sư, tội phạm, hình phạt, v.v.
- "non-legal": Câu hỏi về các chủ đề khác (khoa học, lịch sử, thể thao, giải trí, v.v.)

Chỉ trả lời: legal hoặc non-legal"""

        user_prompt = f"Câu hỏi cần phân loại: {query}"
        
        try:
            response = self._generate(system_prompt, user_prompt, temperature=0.1, max_tokens=32)
            
            # Parse response
            response_lower = response.lower().strip()
            if "legal" in response_lower:
                return "legal", 0.9
            elif "non-legal" in response_lower:
                return "non-legal", 0.9
            else:
                # Fallback to rule-based classification
                return self._rule_based_classify(query)
                
        except Exception as e:
            print(f"⚠️ LLM classification failed: {e}")
            return self._rule_based_classify(query)
    
    def _rule_based_classify(self, query: str) -> Tuple[str, float]:
        """Rule-based fallback classification"""
        legal_keywords = [
            'luật', 'pháp luật', 'tòa án', 'luật sư', 'tội phạm', 'hình phạt',
            'điều', 'khoản', 'điều luật', 'bộ luật', 'nghị định', 'thông tư',
            'xử lý', 'xử phạt', 'truy tố', 'kết án', 'tù', 'phạt tiền',
            'trách nhiệm', 'nghĩa vụ', 'quyền', 'nghĩa vụ', 'vi phạm'
        ]
        
        query_lower = query.lower()
        legal_score = sum(1 for keyword in legal_keywords if keyword in query_lower)
        
        if legal_score > 0:
            return "legal", min(0.7 + legal_score * 0.1, 0.9)
        else:
            return "non-legal", 0.6
    
    def _generate(self, system_prompt: str, user_prompt: str, 
                  temperature: float = 0.1, max_tokens: int = 512) -> str:
        """Generate response using LLM"""
        
        input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,
                num_beams=1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()


# ============================================================================
# ANSWER GENERATOR
# ============================================================================

class AnswerGenerator:
    """Generate answers using LLM + retrieved context"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_legal_answer(self, query: str, retrieved_chunks: List) -> Tuple[str, List[str]]:
        """Generate legal answer using retrieved context"""
        
        # Use simple context formatting (assembly already handled in retrieval)
        context = self._format_simple_context(retrieved_chunks)
        
        system_prompt = f"""Bạn là luật sư chuyên nghiệp với kiến thức sâu về pháp luật Việt Nam.
Nhiệm vụ: Trả lời câu hỏi pháp luật dựa trên ngữ cảnh được cung cấp.

Quy tắc:
1. Chỉ sử dụng thông tin từ ngữ cảnh được cung cấp
2. Trích dẫn chính xác điều luật, khoản luật
3. Giải thích rõ ràng, dễ hiểu
4. Nếu không có thông tin phù hợp, hãy nói rõ

NGỮ CẢNH:
{context}

CÂU HỎI: {query}

TRẢ LỜI:
"""
        
        answer = self._generate(system_prompt, "", temperature=0.2, max_tokens=256)
        
        # Extract citations
        citations = self._extract_citations(retrieved_chunks)
        
        return answer, citations
    
    def generate_non_legal_answer(self, query: str) -> str:
        """Generate non-legal answer"""
        
        system_prompt = """Bạn là trợ lý AI thông minh.
Nhiệm vụ: Trả lời câu hỏi một cách hữu ích và chính xác.

Quy tắc:
1. Trả lời dựa trên kiến thức chung
2. Nếu không biết, hãy nói rõ
3. Giữ câu trả lời ngắn gọn (2-3 câu).
"""
        
        user_prompt = f'Câu hỏi của người dùng: "{query}"'
        
        answer = self._generate(system_prompt, user_prompt, temperature=0.3, max_tokens=128)
        
        return answer
    
    def _generate(self, system_prompt: str, user_prompt: str, 
                  temperature: float = 0.2, max_tokens: int = 1024) -> str:
        """Generate response using LLM"""
        
        if user_prompt:
            input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            input_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                early_stopping=True,
                num_beams=1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _format_simple_context(self, retrieved_chunks: List) -> str:
        """Format retrieved chunks into simple context"""
        
        context_parts = []
        
        for result in retrieved_chunks:
            article = result.metadata.get('article', 'N/A')
            title = result.metadata.get('article_title', '')
            
            formatted = f"""=== {article}. {title} ===
{result.content}
---"""
            context_parts.append(formatted)
        
        return '\n\n'.join(context_parts)
    
    def _extract_citations(self, retrieved_chunks: List) -> List[str]:
        """Extract citations from retrieved chunks"""
        citations = []
        
        for chunk in retrieved_chunks:
            article = chunk.metadata.get('article', 'N/A')
            clause = chunk.metadata.get('clause', 'N/A')
            citations.append(f"{article} - Khoản {clause}")
        
        return citations


# ============================================================================
# MAIN RAG ASSISTANT
# ============================================================================

class LegalRAGAssistant:
    """Complete Legal RAG Assistant"""
    
    def __init__(self, 
                 llm_model_id: str = "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2",
                 vector_store_path: str = "context_aware_vectors.pkl",
                 chunks_file: str = "context_aware_chunks.json"):
        
        print("🚀 Initializing Legal RAG Assistant...")
        
        # Load LLM with 4-bit quantization for maximum speed and memory efficiency
        print(f"📥 Loading LLM: {llm_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
        
        # Check if CUDA is available for quantization
        if torch.cuda.is_available():
            # Set GPU memory limit to 80% for optimal performance
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = int(total_gpu_memory * 0.80)
            print(f"   GPU Memory: {total_gpu_memory / 1024**3:.1f} GB total")
            print(f"   Setting max memory limit: {max_memory / 1024**3:.1f} GB (80%)")
            
            try:
                print("   Attempting 4-bit quantization for LLM (optimized)...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    llm_model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    max_memory={0: max_memory},
                    trust_remote_code=True
                )
                print("✅ LLM loaded with 4-bit quantization (~75% RAM reduction, 2-3x faster)")
            except Exception as quant_error:
                print(f"   ⚠️ 4-bit quantization failed: {quant_error}")
                print("   Trying 8-bit quantization...")
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        llm_model_id,
                        quantization_config=bnb_config,
                        device_map="auto",
                        max_memory={0: max_memory},
                        trust_remote_code=True
                    )
                    print("✅ LLM loaded with 8-bit quantization (~50% RAM reduction)")
                except Exception as quant_error2:
                    print(f"   ⚠️ 8-bit quantization failed: {quant_error2}")
                    print("   Loading with standard precision...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        llm_model_id,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_memory={0: max_memory},
                        trust_remote_code=True
                    )
                    print("✅ LLM loaded with bfloat16 (standard)")
        else:
            # CPU mode - no quantization, use float32 for better compatibility
            print("   CPU mode detected - loading without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print("✅ LLM loaded with float32 (CPU optimized)")
        
        self.device = self.model.device
        print(f"✅ LLM loaded on device: {self.device}")
        
        # Load vector store
        print(f"📥 Loading vector store: {vector_store_path}")
        self.vector_store = ContextAwareVectorStore()
        self.vector_store.load(vector_store_path)
        print(f"✅ Vector store loaded: {len(self.vector_store.vectors)} chunks")
        
        # Initialize retrieval system
        print("📥 Initializing retrieval system...")
        config = RetrievalConfig(
            embedding_weight=0.4,
            reranker_weight=0.6,
            initial_search_k=11,
            final_results_k=5
        )
        self.retrieval_system = AdvancedRetrievalSystem(
            self.vector_store, 
            config,
            chunks_file
        )
        print("✅ Retrieval system initialized")
        
        # Initialize components
        self.normalizer = QueryNormalizer(self.model, self.tokenizer, self.device)
        self.classifier = QueryClassifier(self.model, self.tokenizer, self.device)
        self.generator = AnswerGenerator(self.model, self.tokenizer, self.device)
        
        print("✅ Legal RAG Assistant ready!")
    
    def ask(self, query: str) -> RAGResponse:
        """Main method to process user query"""
        
        print(f"\n🔍 Processing query: '{query}'")
        
        # Step 1: Normalize query
        print(f"   📝 Step 1: Normalizing query...")
        normalized_query = self.normalizer.normalize(query)
        print(f"   ✅ Normalized: '{normalized_query}'")
        
        # Step 2: Classify query
        print(f"   🏷️  Step 2: Classifying query...")
        query_type, confidence = self.classifier.classify(normalized_query)
        print(f"   ✅ Type: {query_type} (confidence: {confidence:.2f})")
        
        # Create query analysis
        query_analysis = QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            query_type=query_type,
            confidence=confidence
        )
        
        if query_type == "legal":
            # Step 3: Retrieve relevant legal context
            print(f"   🔎 Step 3: Retrieving relevant legal context...")
            retrieved_chunks = self.retrieval_system.search_with_context(normalized_query, top_k=5)
            print(f"   ✅ Retrieved {len(retrieved_chunks)} chunks")
            
            # Step 4: Generate legal answer (context assembly handled in retrieval)
            print(f"   🤖 Step 4: Generating legal answer...")
            answer, citations = self.generator.generate_legal_answer(normalized_query, retrieved_chunks)
            
            # Calculate overall confidence
            overall_confidence = confidence * 0.3 + (sum(chunk.confidence_score for chunk in retrieved_chunks) / len(retrieved_chunks)) * 0.7
            
        else:
            # Non-legal query - direct generation
            print(f"   🤖 Step 3: Generating non-legal answer...")
            answer = self.generator.generate_non_legal_answer(normalized_query)
            retrieved_chunks = []
            citations = []
            overall_confidence = confidence
        
        print(f"   ✅ Answer generated (confidence: {overall_confidence:.2f})")
        
        return RAGResponse(
            answer=answer,
            query_analysis=query_analysis,
            retrieved_chunks=retrieved_chunks,
            citations=citations,
            confidence_score=overall_confidence
        )


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Test Legal RAG Assistant"""
    
    print("🧪 TESTING LEGAL RAG ASSISTANT")
    print("=" * 60)
    
    # Initialize assistant
    assistant = LegalRAGAssistant()
    
    # Test queries
    test_queries = [
        "Khoan hồng đối với người tự thú",
        "Hình phạt tử hình áp dụng như thế nào?",
        "Điều kiện miễn trách nhiệm hình sự",
        "Thời tiết hôm nay như thế nào?",  # Non-legal
        "Hình phạt cho tội trộm cắp tài sản"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 TESTING: {query}")
        print(f"{'='*80}")
        
        try:
            response = assistant.ask(query)
            
            print(f"\n📋 QUERY ANALYSIS:")
            print(f"   Original: {response.query_analysis.original_query}")
            print(f"   Normalized: {response.query_analysis.normalized_query}")
            print(f"   Type: {response.query_analysis.query_type}")
            print(f"   Confidence: {response.query_analysis.confidence:.2f}")
            
            print(f"\n💬 ANSWER:")
            print(f"   {response.answer}")
            
            if response.citations:
                print(f"\n📚 CITATIONS:")
                for citation in response.citations:
                    print(f"   - {citation}")
            
            print(f"\n🎯 OVERALL CONFIDENCE: {response.confidence_score:.2f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue to next query...")

if __name__ == "__main__":
    main()
