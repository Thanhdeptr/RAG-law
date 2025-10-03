import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from pyvi import ViTokenizer

# ---------------------------
# Bước 1: Embedding model (DEk21_hcmute_embedding)
# ---------------------------
embedding_model = SentenceTransformer("huyydangg/DEk21_hcmute_embedding")

query = "Người lao động có quyền nghỉ thai sản bao lâu?"
documents = [
    "Theo điều 139 Bộ luật Lao động 2019, lao động nữ được nghỉ trước và sau khi sinh con là 6 tháng.",
    "Người lao động có quyền nghỉ phép hằng năm theo thỏa thuận với người sử dụng lao động.",
    "Thời tiết hôm nay rất đẹp, nhiều người đi chơi.",
    "Trong lĩnh vực AI, machine learning là một nhánh quan trọng.",
    "Công nghệ blockchain ngày càng phát triển mạnh mẽ."
]

# Tokenize Vietnamese text trước khi encode
segmented_query = ViTokenizer.tokenize(query)
segmented_docs = [ViTokenizer.tokenize(doc) for doc in documents]

print(f"Query sau tokenization: {segmented_query}")
print(f"Documents sau tokenization: {segmented_docs[:2]}...")  # chỉ in 2 docs đầu

# Encode query + documents
query_emb = embedding_model.encode([segmented_query])
doc_embs = embedding_model.encode(segmented_docs)

# Tính cosine similarity
similarities = F.cosine_similarity(torch.tensor(query_emb), torch.tensor(doc_embs)).flatten()

# Lấy top-k document (ví dụ top 3)
top_k = 3
topk_indices = torch.topk(similarities, top_k).indices.tolist()
topk_docs = [documents[i] for i in topk_indices]

print("Top-k documents từ embedding:")
for i, doc in enumerate(topk_docs, 1):
    print(f"{i}. {doc} - sim={similarities[topk_indices[i-1]]:.4f}")

# ---------------------------
# Bước 2: Reranker (hghaan/rerank_model)
# ---------------------------
rerank_model_id = "hghaan/rerank_model"
tokenizer = AutoTokenizer.from_pretrained(rerank_model_id, trust_remote_code=True)
rerank_model = AutoModel.from_pretrained(rerank_model_id, trust_remote_code=True, device_map="auto")

def get_rerank_score(query, doc):
    inputs = tokenizer(f"query: {query} document: {doc}", return_tensors="pt", truncation=True, padding=True).to(rerank_model.device)
    with torch.no_grad():
        outputs = rerank_model(**inputs)
    
    # Debug: in ra các attributes có sẵn
    print(f"Model outputs attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
    
    # Model rerank thường trả về score trực tiếp hoặc qua pooler_output
    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
        # Sử dụng pooler_output nếu có và không None
        score = outputs.pooler_output.squeeze().item()
        print(f"Using pooler_output: {score}")
    elif hasattr(outputs, 'last_hidden_state'):
        # Sử dụng CLS token (token đầu tiên) với cách tính đơn giản
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        # Lấy norm của CLS embedding làm score
        score = torch.norm(cls_embedding, dim=-1).item()
        print(f"Using CLS norm: {score}")
    else:
        # Fallback: sử dụng toàn bộ hidden states
        score = outputs.last_hidden_state.mean().item()
        print(f"Using mean of all hidden states: {score}")
    
    return score

# Tính score rerank cho top-k
rerank_scores = [(doc, get_rerank_score(query, doc)) for doc in topk_docs]
# Sắp xếp lại theo score giảm dần
reranked_docs = sorted(rerank_scores, key=lambda x: x[1], reverse=True)

print("\nKết quả sau rerank:")
for i, (doc, score) in enumerate(reranked_docs, 1):
    print(f"{i}. {doc} - score={score:.4f}")
