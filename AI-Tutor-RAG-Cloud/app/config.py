from pathlib import Path
import os
from dotenv import load_dotenv

# .env из корня проекта
load_dotenv()

# директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# путь к данным
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_INDEX_DIR = DATA_DIR / "index"

# папка для пользовательских загрузок
USER_DATA_DIR = DATA_DIR / "user_uploads"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# логи
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOGS_DIR / "app.log"

# параметры чанков
CHUNK_SIZE = 1000        
CHUNK_OVERLAP = 200      

# эмбеддинг
EMBEDDINGS_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# параметр поиска
TOP_K = 5

# пути к индексу
FAISS_INDEX_PATH = DATA_INDEX_DIR / "index.faiss"
METADATA_PATH = DATA_INDEX_DIR / "chunks_metadata.json"

# настройки LLM / API
LLM_API_KEY = os.getenv("CLOUD_API_KEY") or os.getenv("OPENAI_API_KEY", "")
LLM_MODEL_NAME = os.getenv("CLOUD_MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
