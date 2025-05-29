# Wikipedia Chatbot Backend

Welcome to **wikipedia-chatbot-backend** – a Python-powered backend designed to serve as the brain of a Wikipedia-based chatbot application!

## 🚀 Overview

This repository contains the server-side logic for a Wikipedia chatbot system, handling natural language queries, Wikipedia content retrieval, and intelligent response generation. Built with FastAPI and modern NLP libraries, it's structured for easy integration, scalable deployments, and accurate information delivery.

## ✨ Features

- Natural Language Processing for query understanding
- Wikipedia content retrieval and analysis
- Intelligent response generation with confidence scoring
- RESTful API endpoints for easy frontend integration
- CORS support for both development and production environments
- Comprehensive error handling and logging
- Conversation history support for context-aware responses

## 🛠️ Tech Stack

- **Framework:** FastAPI
- **Server:** Uvicorn
- **NLP Libraries:** 
  - spaCy
  - NLTK
  - Sentence Transformers
  - Transformers (Hugging Face)
- **Machine Learning:** 
  - scikit-learn
  - PyTorch
- **Wikipedia Integration:** wikipedia-api
- **Data Processing:** pandas, numpy
- **Environment Management:** python-dotenv

## ⚡ Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wikibot.git
cd wikibot/backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Development mode
python run.py

# Or directly with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

## 📚 API Usage

### Query Endpoint

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "conversation_history": []
  }'
```

Response format:
```json
{
  "response": "string",
  "confidence": float,
  "sources": ["string"]
}
```

### Health Check

```bash
curl http://localhost:8000/
```

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements or bug fixes.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


