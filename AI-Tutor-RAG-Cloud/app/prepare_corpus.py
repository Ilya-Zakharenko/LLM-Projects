import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
import docx
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_INDEX_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDINGS_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    LOG_PATH,
)
from .utils import clean_text, split_text


# логирование
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def iter_source_files(root: Path) -> List[Path]:
    exts = {".pdf", ".docx", ".txt"}
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return files


def read_pdf(path: Path) -> List[Dict[str, Any]]:
    """
    возвращает список для каждой страницы PDF
    """
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages


def read_docx(path: Path) -> str:
    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)


def read_txt(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_chunks() -> List[Dict[str, Any]]:
    """
    возвращает список чанков с метаданными.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    files = iter_source_files(DATA_RAW_DIR)
    logger.info("Найдено %d файлов в raw-корпусе", len(files))

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for file_path in files:
        rel_path = file_path.relative_to(DATA_RAW_DIR)
        subject = rel_path.parts[0] if len(rel_path.parts) > 1 else "default"

        logger.info("Обработка файла: %s", file_path)

        if file_path.suffix.lower() == ".pdf":
            pages = read_pdf(file_path)
            # сохраняем сырое содержимое PDF постранично
            raw_text_full = "\n\n".join([p["text"] or "" for p in pages])
            processed_path = DATA_PROCESSED_DIR / (file_path.stem + "_processed.txt")
            with open(processed_path, "w", encoding="utf-8") as f:
                f.write(raw_text_full)

            for page_info in pages:
                page_text = clean_text(page_info["text"])
                if not page_text:
                    continue
                text_chunks = split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
                for ch in text_chunks:
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "text": ch,
                            "source": str(rel_path),
                            "subject": subject,
                            "page": page_info["page"],
                        }
                    )
                    chunk_id += 1

        elif file_path.suffix.lower() == ".docx":
            raw_text = read_docx(file_path)
            raw_text_clean = clean_text(raw_text)
            processed_path = DATA_PROCESSED_DIR / (file_path.stem + "_processed.txt")
            with open(processed_path, "w", encoding="utf-8") as f:
                f.write(raw_text_clean)

            text_chunks = split_text(raw_text_clean, CHUNK_SIZE, CHUNK_OVERLAP)
            for ch in text_chunks:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": ch,
                        "source": str(rel_path),
                        "subject": subject,
                        "page": None,
                    }
                )
                chunk_id += 1

        elif file_path.suffix.lower() == ".txt":
            raw_text = read_txt(file_path)
            raw_text_clean = clean_text(raw_text)
            processed_path = DATA_PROCESSED_DIR / (file_path.stem + "_processed.txt")
            with open(processed_path, "w", encoding="utf-8") as f:
                f.write(raw_text_clean)

            text_chunks = split_text(raw_text_clean, CHUNK_SIZE, CHUNK_OVERLAP)
            for ch in text_chunks:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": ch,
                        "source": str(rel_path),
                        "subject": subject,
                        "page": None,
                    }
                )
                chunk_id += 1

    logger.info("Всего чанков: %d", len(chunks))
    return chunks


def build_index(chunks: List[Dict[str, Any]]) -> None:
    """
    строит индекс по текстам чанков и сохраняет его вместе с метаданными
    """
    if not chunks:
        logger.warning("Список чанков пуст, индекс не будет создан.")
        return

    texts = [c["text"] for c in chunks]

    logger.info("Загрузка модели эмбеддингов: %s", EMBEDDINGS_MODEL_NAME)
    model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)

    logger.info("Подсчёт эмбеддингов для %d чанков", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")
    dim = embeddings.shape[1]

    logger.info("Создание FAISS индекса (dim=%d)", dim)
    index = faiss.IndexFlatIP(dim) 
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info("FAISS индекс сохранён в %s", FAISS_INDEX_PATH)

    # в метаданные id вектора
    for i, c in enumerate(chunks):
        c["vector_id"] = i

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "chunks": chunks,
                "embeddings_model": EMBEDDINGS_MODEL_NAME,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Метаданные чанков сохранены в %s", METADATA_PATH)


def main():
    logger.info("=== Запуск подготовки корпуса ===")
    chunks = build_chunks()
    build_index(chunks)
    logger.info("=== Подготовка корпуса завершена ===")


if __name__ == "__main__":
    main()