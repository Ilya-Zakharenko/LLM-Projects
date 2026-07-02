# <center> **PROJECT: AI Tutor Bot**  
*RAG-based Technical Disciplines Tutor for Cloud.ru*

Intelligent Telegram bot that acts as a personal AI tutor. It answers questions based on user-uploaded educational materials (PDF, DOCX, TXT) using Retrieval-Augmented Generation (RAG).

---

### **Project Goal**

Create an MVP of an AI tutor capable of explaining technical subjects, answering student questions, generating self-assessment quizzes, and always citing sources from the provided materials.

---

### **Key Features**

- **Personal Knowledge Base**: Users can upload their own lecture notes, textbooks, or summaries
- **Semantic Search**: Uses `SentenceTransformer` + FAISS for meaningful retrieval
- **Grounded Generation**: LLM answers based only on uploaded documents with source references
- **Interactive Learning**: `/quiz` mode with hidden answers (Telegram spoilers)
- **Multiple Subjects Support**: Personal and shared corpora
- **Logging & Transparency**: Full request logging and source attribution

---

### **Technologies Used**

- **LLM**: OpenAI (gpt-4o-mini or Cloud.ru models)
- **Embeddings**: `SentenceTransformer`
- **Vector Store**: `FAISS`
- **Document Parsing**: `pypdf`, `python-docx`
- **Bot Framework**: `aiogram` (async)
- **Other**: `dotenv`, logging, chunking with overlap

---

### **Project Architecture**

- `prepare_corpus.py` — processing and indexing documents
- `rag_pipeline.py` — core RAG logic (retrieval + generation)
- `llm_client.py` — LLM interaction layer
- `bot.py` — Telegram bot handlers and conversation logic
- `config.py` & `utils.py` — configuration and utilities

---

### **Conclusion**

This project demonstrates a complete **RAG-based educational AI system**. The bot combines modern semantic search with LLM generation, providing accurate, traceable, and personalized learning assistance. Developed as an MVP for the Cloud.ru hackathon.

---

### **How to run**

```bash
git clone <repository-url>
cd ai_tutor_bot

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

# Create .env file with your tokens
cp .env.example .env

# Run the bot
python -m app.bot
