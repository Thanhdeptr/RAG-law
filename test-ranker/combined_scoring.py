import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer
import numpy as np

class CombinedScorer:
    def __init__(self, embedding_model_id="huyydangg/DEk21_hcmute_embedding", 
                 rerank_model_id="hghaan/rerank_model"):
        """Khởi tạo scorer với embedding và reranker models"""
        self.embedding_model = SentenceTransformer(embedding_model_id)
        
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_id, trust_remote_code=True)
        self.rerank_model = AutoModel.from_pretrained(rerank_model_id, trust_remote_code=True, device_map="auto")
        
        # Trọng số mặc định (có thể điều chỉnh)
        self.embedding_weight = 0.4  # 40% cho embedding
        self.reranker_weight = 0.6   # 60% cho reranker
        
    def normalize_embedding_score(self, sim_score):
        """Chuyển cosine similarity (-1,1) thành (0,1)"""
        return (sim_score + 1) / 2
    
    def normalize_reranker_score(self, rerank_score, min_score=8, max_score=15):
        """Chuẩn hóa reranker score thành (0,1) dựa trên min/max observed"""
        # Clip score trong khoảng min/max
        clipped_score = np.clip(rerank_score, min_score, max_score)
        # Normalize về (0,1)
        return (clipped_score - min_score) / (max_score - min_score)
    
    def get_embedding_score(self, query, documents):
        """Tính embedding score cho documents"""
        # Tokenize Vietnamese
        segmented_query = ViTokenizer.tokenize(query)
        segmented_docs = [ViTokenizer.tokenize(doc) for doc in documents]
        
        # Encode
        query_emb = self.embedding_model.encode([segmented_query])
        doc_embs = self.embedding_model.encode(segmented_docs)
        
        # Cosine similarity
        similarities = F.cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_embs)).flatten()
        
        # Normalize to (0,1)
        normalized_scores = [self.normalize_embedding_score(sim.item()) for sim in similarities]
        
        return normalized_scores
    
    def get_reranker_score(self, query, documents):
        """Tính reranker score cho documents"""
        scores = []
        
        for doc in documents:
            inputs = self.rerank_tokenizer(
                f"query: {query} document: {doc}", 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(self.rerank_model.device)
            
            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
            
            # Sử dụng CLS norm như trước
            if hasattr(outputs, 'last_hidden_state'):
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                score = torch.norm(cls_embedding, dim=-1).item()
            else:
                score = outputs.last_hidden_state.mean().item()
            
            scores.append(score)
        
        # Normalize scores
        normalized_scores = [self.normalize_reranker_score(score) for score in scores]
        
        return normalized_scores
    
    def get_combined_score(self, query, documents, embedding_weight=None, reranker_weight=None):
        """Tính điểm tổng hợp với trọng số"""
        if embedding_weight is None:
            embedding_weight = self.embedding_weight
        if reranker_weight is None:
            reranker_weight = self.reranker_weight
        
        # Tính các score riêng lẻ
        emb_scores = self.get_embedding_score(query, documents)
        rerank_scores = self.get_reranker_score(query, documents)
        
        # Tính điểm tổng hợp
        combined_scores = []
        for emb, rerank in zip(emb_scores, rerank_scores):
            combined = embedding_weight * emb + reranker_weight * rerank
            combined_scores.append(combined)
        
        return combined_scores, emb_scores, rerank_scores
    
    def get_confidence_percentage(self, combined_score):
        """Chuyển điểm tổng hợp thành phần trăm tin cậy"""
        # Sử dụng sigmoid để chuyển (0,1) thành phần trăm
        confidence = 1 / (1 + np.exp(-10 * (combined_score - 0.5))) * 100
        return min(100, max(0, confidence))  # Clip trong [0, 100]

def main():
    # Khởi tạo scorer
    scorer = CombinedScorer()
    
    # Test data
    query = "Người lao động có quyền nghỉ thai sản bao lâu?"
    documents = [
        "Theo điều 139 Bộ luật Lao động 2019, lao động nữ được nghỉ trước và sau khi sinh con là 6 tháng.",
        "Người lao động có quyền nghỉ phép hằng năm theo thỏa thuận với người sử dụng lao động.",
        "Thời tiết hôm nay rất đẹp, nhiều người đi chơi.",
        "Trong lĩnh vực AI, machine learning là một nhánh quan trọng.",
        "Công nghệ blockchain ngày càng phát triển mạnh mẽ."
    ]
    
    print("=" * 80)
    print("HỆ THỐNG TÍNH ĐIỂM TỔNG HỢP VÀ ĐỘ TIN CẬY")
    print("=" * 80)
    
    # Test với trọng số mặc định
    print(f"\n📊 Trọng số: Embedding {scorer.embedding_weight*100:.0f}% | Reranker {scorer.reranker_weight*100:.0f}%")
    combined, emb, rerank = scorer.get_combined_score(query, documents)
    
    # Sắp xếp theo điểm tổng hợp
    results = list(zip(documents, combined, emb, rerank))
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 Query: {query}")
    print("\n📈 KẾT QUẢ XẾP HẠNG:")
    print("-" * 80)
    
    for i, (doc, comb, emb_score, rerank_score) in enumerate(results, 1):
        confidence = scorer.get_confidence_percentage(comb)
        print(f"{i}. Độ tin cậy: {confidence:.1f}%")
        print(f"   📝 Document: {doc[:80]}...")
        print(f"   🔢 Chi tiết: Combined={comb:.3f} | Embedding={emb_score:.3f} | Reranker={rerank_score:.3f}")
        print()
    
    # Test với trọng số khác
    print("\n" + "=" * 80)
    print("🔄 THỬ NGHIỆM VỚI TRỌNG SỐ KHÁC")
    print("=" * 80)
    
    weight_configs = [
        (0.5, 0.5, "Cân bằng"),
        (0.7, 0.3, "Ưu tiên Embedding"),
        (0.2, 0.8, "Ưu tiên Reranker")
    ]
    
    for emb_w, rerank_w, desc in weight_configs:
        print(f"\n⚖️ {desc} (Embedding: {emb_w*100:.0f}%, Reranker: {rerank_w*100:.0f}%)")
        combined, _, _ = scorer.get_combined_score(query, documents, emb_w, rerank_w)
        
        # Top 2 results
        results = list(zip(documents, combined))
        results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (doc, comb) in enumerate(results[:2], 1):
            confidence = scorer.get_confidence_percentage(comb)
            print(f"  {i}. {confidence:.1f}% - {doc[:60]}...")

if __name__ == "__main__":
    main()
