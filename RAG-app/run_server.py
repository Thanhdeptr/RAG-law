#!/usr/bin/env python3
"""
Run the Legal RAG FastAPI server
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
