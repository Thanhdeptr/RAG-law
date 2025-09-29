# Legal RAG App cho Văn bản Pháp luật Việt Nam

RAG (Retrieval-Augmented Generation) app để hỏi đáp văn bản pháp luật, sử dụng:
- Ollama local model **gpt-oss:20b** (generation) 
- **nomic-embed-text** (embedding)
- Chunking theo cấu trúc: Chương → Mục → Điều → Khoản → Điểm
- Vector store: ChromaDB

## Cài đặt và chạy

### Bước 1: Cài Python dependencies
```cmd
cd B:\RAG-app
pip install -r requirements.txt
```

### Bước 2: Cấu hình môi trường
1. Copy file cấu hình:
```cmd
copy env.example .env
```

2. Chỉnh sửa `.env` nếu cần (mặc định đã đúng endpoint của bạn):
```
VITE_OLLAMA_BASE=http://192.168.10.32:11434/v1
VITE_MODEL_NAME=gpt-oss:20b
EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./storage/chroma
```

### Bước 3: Tạo thư mục dữ liệu
```cmd
mkdir data
mkdir storage
```

### Bước 4: Đảm bảo Ollama có model embedding
Trên máy Ollama (192.168.10.32), chạy:
```bash
ollama pull nomic-embed-text
```

### Bước 5: Chạy server
```cmd
python run_server.py
```

Server sẽ chạy tại: **http://localhost:8000**

## Sử dụng

### 1. Ingest văn bản (qua API)

**Upload file:**
```bash
curl -X POST "http://localhost:8000/ingest/file" \
  -F "file=@data/luat-mau.pdf" \
  -F "doc_title=Luật Mẫu 2023"
```

**Hoặc gửi text trực tiếp:**
```bash
curl -X POST "http://localhost:8000/ingest/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Chương I\nQuy định chung\nĐiều 1. Phạm vi điều chỉnh\nLuật này quy định...",
    "doc_metadata": {"doc_title": "Luật Test", "year": "2023"}
  }'
```

### 2. Hỏi đáp (qua API)
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Phạm vi điều chỉnh của luật này là gì?",
    "top_k": 5
  }'
```

### 3. Kiểm tra health
```bash
curl http://localhost:8000/health
```

## Cấu trúc project

```
RAG-app/
├── backend/app/
│   ├── main.py           # FastAPI endpoints
│   ├── config.py         # Cấu hình từ env vars
│   ├── embeddings.py     # Client Ollama embedding/chat
│   ├── legal_chunker.py  # Chunking theo cấu trúc pháp lý
│   ├── vectorstore.py    # ChromaDB wrapper
│   ├── schemas.py        # Pydantic models
│   └── text_extractor.py # Extract từ PDF/DOCX
├── data/                 # Thư mục chứa file văn bản
├── storage/chroma/       # ChromaDB persistence
├── run_server.py         # Script chạy server
├── requirements.txt      # Python dependencies
└── .env                  # Cấu hình (copy từ env.example)
```

## API Endpoints

- `POST /ingest/text` - Ingest raw text
- `POST /ingest/file` - Upload file PDF/DOCX/TXT
- `POST /query` - Hỏi đáp với RAG
- `GET /health` - Kiểm tra kết nối Ollama

## Lưu ý

1. **Model embedding**: Đảm bảo `nomic-embed-text` đã được pull trên Ollama server
2. **File format**: Hỗ trợ PDF, DOCX, TXT
3. **Chunking**: Tự động nhận diện Chương/Mục/Điều/Khoản/Điểm
4. **Trích dẫn**: Kết quả trả về có metadata để trích dẫn chính xác
5. **Performance**: Lần đầu ingest sẽ chậm do tạo embeddings, query sau đó rất nhanh
