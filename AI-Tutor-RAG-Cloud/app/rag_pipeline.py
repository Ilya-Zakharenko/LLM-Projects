import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx

from .config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    EMBEDDINGS_MODEL_NAME,
    TOP_K,
    LOG_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from .utils import clean_text, split_text
from . import llm_client

# логирование
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# глоб. состояние индексов 
_embeddings_model: Optional[SentenceTransformer] = None
_faiss_index: Optional[faiss.Index] = None
_chunks: List[Dict] = []
_subjects: List[str] = []
_loaded: bool = False

SYSTEM_PROMPT = """
Ты — AI-репетитор по техническим дисциплинам.
Отвечай ТОЛЬКО на основе переданного контекста (фрагменты учебников, конспектов, лекций).
Если из контекста нельзя ответить на вопрос, честно скажи об этом и предложи, как можно уточнить вопрос.
Пиши по-русски, понятным языком. Объясняй шаг за шагом, можно с примерами.
Не выдумывай факты и не ссылайся на материалы, которых нет в контексте.
"""


# функции для чтения текста 

def _read_pdf_full(path: Path) -> str:
    reader = PdfReader(str(path))
    texts = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(texts)


def _read_docx_full(path: Path) -> str:
    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join(paragraphs)


def _read_txt_full(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# Загрузка инициализация индекса 

def load_index() -> None:
    """
    Загружает модель эмбеддингов, FAISS и метаданные чанков
    Если индекс отсутствует создаёт пустой индекс
    """
    global _embeddings_model, _faiss_index, _chunks, _subjects, _loaded

    if _loaded:
        return

    logger.info("Загрузка эмбеддинговой модели: %s", EMBEDDINGS_MODEL_NAME)
    _embeddings_model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)

    dim = _embeddings_model.get_sentence_embedding_dimension()

    if FAISS_INDEX_PATH.exists():
        logger.info("Загрузка FAISS индекса из %s", FAISS_INDEX_PATH)
        _faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        logger.info("FAISS индекс не найден, создаю новый пустой индекс (dim=%d)", dim)
        _faiss_index = faiss.IndexFlatIP(dim)

    if METADATA_PATH.exists():
        logger.info("Загрузка метаданных чанков из %s", METADATA_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _chunks = data.get("chunks", [])
    else:
        logger.info("Файл метаданных не найден, начинаю с пустого списка чанков.")
        _chunks = []

    # собираем список предметов
    subjects_set = set()
    for c in _chunks:
        subj = c.get("subject") or "default"
        subjects_set.add(subj)
    _subjects = sorted(subjects_set)

    _loaded = True
    logger.info(
        "Индекс готов. Всего векторов: %d, чанков: %d, предметов: %d",
        _faiss_index.ntotal,
        len(_chunks),
        len(_subjects),
    )


def list_subjects() -> List[str]:
    """
    Возвращает список доступных предметов
    """
    global _loaded, _subjects
    if not _loaded:
        load_index()
    return _subjects

def get_user_stats(user_id: int) -> Dict[str, object]:
    """
    Возвращает статистику по материалам пользователя
    subject 
    num_chunks
    files
    """
    global _loaded, _chunks
    if not _loaded:
        load_index()

    subject = f"user_{user_id}"
    num_chunks = 0
    files_set = set()

    for c in _chunks:
        if c.get("subject") == subject:
            num_chunks += 1
            src = c.get("source")
            if src:
                files_set.add(src)

    return {
        "subject": subject,
        "num_chunks": num_chunks,
        "files": sorted(files_set),
    }

# добавление пользовательских документов

def prepare_document_chunks(file_path: Path) -> List[str]:
    """
    Читает файл чистит текст и разбивает на чанки
    Не трогает FAISSтолько готовит тексты фрагментов
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        raw_text = _read_pdf_full(file_path)
    elif suffix == ".docx":
        raw_text = _read_docx_full(file_path)
    elif suffix == ".txt":
        raw_text = _read_txt_full(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {suffix}")

    text_clean = clean_text(raw_text)
    if not text_clean:
        logger.warning("Файл %s пустой или не удалось извлечь текст", file_path)
        return []

    chunk_texts = split_text(text_clean, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunk_texts:
        logger.warning(
            "После разбиения на чанки ничего не получилось для файла %s",
            file_path,
        )
        return []

    logger.info(
        "Файл %s подготовлен: %d чанков",
        file_path.name,
        len(chunk_texts),
    )
    return chunk_texts


def add_chunks_to_index(
    chunk_texts: List[str],
    subject: str,
    source: str,
) -> int:
    """
    Добавляет уже подготовленные чанки в FAISS.
    Возвращает количество добавленных чанков
    """
    global _embeddings_model, _faiss_index, _chunks, _subjects, _loaded

    if not _loaded:
        load_index()

    assert _embeddings_model is not None and _faiss_index is not None

    if not chunk_texts:
        return 0

    logger.info(
        "Добавляю в индекс %d чанков (subject=%s, source=%s)",
        len(chunk_texts),
        subject,
        source,
    )

    embeddings = _embeddings_model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    start_vec_id = _faiss_index.ntotal
    _faiss_index.add(embeddings)

    for i, ch_text in enumerate(chunk_texts):
        vec_id = start_vec_id + i
        _chunks.append(
            {
                "chunk_id": len(_chunks),
                "vector_id": vec_id,
                "text": ch_text,
                "source": source,
                "subject": subject,
                "page": None,
            }
        )

    # обновляем список
    _subjects = sorted(set(_subjects) | {subject})

    # сохраняем индекс и метаданные
    faiss.write_index(_faiss_index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"chunks": _chunks, "embeddings_model": EMBEDDINGS_MODEL_NAME},
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Добавление завершено. Всего векторов в индексе: %d",
        _faiss_index.ntotal,
    )
    return len(chunk_texts)


def add_user_document(user_id: int, file_path: Path) -> int:
    chunk_texts = prepare_document_chunks(file_path)
    if not chunk_texts:
        return 0
    subject = f"user_{user_id}"
    source = file_path.name
    return add_chunks_to_index(chunk_texts, subject, source)

# поиск контекста

def retrieve(
    query: str,
    subject: Optional[str] = None,
    k: int = TOP_K,
) -> List[Dict]:
    """
    По запросу пользователя возвращает список релевантных чанков с метаданными
    subject необязательная фильтрация по предмету 
    """
    global _embeddings_model, _faiss_index, _chunks, _loaded

    if not _loaded:
        load_index()

    if _embeddings_model is None or _faiss_index is None or _faiss_index.ntotal == 0:
        logger.info("Индекс пуст, возвращаю пустой список контекстов.")
        return []

    query_clean = clean_text(query)
    logger.info("Поисковый запрос: %s (subject=%s)", query_clean, subject)

    query_vec = _embeddings_model.encode(
        [query_clean],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # если задан subject сперва берём побольше кандидатов и потом фильтруем
    top_n = max(k * 3, k) if subject else k

    scores, ids = _faiss_index.search(query_vec, top_n)
    ids = ids[0]
    scores = scores[0]

    results: List[Dict] = []
    for idx, score in zip(ids, scores):
        if idx < 0:
            continue
        meta = _chunks[idx]
        if subject and meta.get("subject") != subject:
            continue

        results.append(
            {
                "text": meta["text"],
                "source": meta.get("source"),
                "page": meta.get("page"),
                "subject": meta.get("subject"),
                "score": float(score),
            }
        )
        if len(results) >= k:
            break

    logger.info("Найдено %d релевантных чанков", len(results))
    return results


# построение промпта 

def build_prompt(
    query: str,
    contexts: List[Dict],
    answer_style: str = "detailed",
) -> List[Dict[str, str]]:
    """
    собирает список сообщений для LLM 
    contexts список чанков text, source, page, subject
    answer_style detailed, short
    """
    context_blocks = []
    for i, c in enumerate(contexts, start=1):
        label = f"[Фрагмент {i} | источник: {c.get('source')}"
        if c.get("page"):
            label += f", стр. {c.get('page')}"
        label += "]"
        context_blocks.append(f"{label}\n{c['text']}")

    context_text = "\n\n".join(context_blocks)

    if answer_style == "short":
        style_instruction = (
            "Сформулируй КРАТКИЙ ответ (3–5 предложений), по делу, без лишней воды."
        )
    else:
        style_instruction = (
            "Дай развёрнутый, но понятный ответ с пояснениями шаг за шагом, "
            "при необходимости с небольшими примерами."
        )

    user_content = (
        f"Вопрос студента:\n{query}\n\n"
        f"Ниже фрагменты учебных материалов:\n{context_text}\n\n"
        f"{style_instruction} Если информации не хватает — прямо скажи об этом."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]
    return messages


# функция ответа

def answer(
    query: str,
    subject: Optional[str] = None,
    answer_style: str = "detailed",
) -> Tuple[str, List[Dict]]:
    """
      текст ответа модели,
      список использованных источников 
    """
    contexts = retrieve(query, subject=subject, k=TOP_K)

    if not contexts:
        logger.info("Подходящего контекста не найдено.")
        no_ctx_answer = (
            "Я не нашёл подходящих фрагментов в имеющихся учебных материалах. "
            "Возможно, вопрос вне текущего корпуса или сформулирован слишком общо. "
            "Сначала загрузите материалы командой /upload или уточните тему."
        )
        return no_ctx_answer, []

    contexts_for_prompt = contexts[:3]
    messages = build_prompt(query, contexts_for_prompt, answer_style=answer_style)

    try:
        reply = llm_client.generate(
            messages,
            max_tokens=700 if answer_style == "detailed" else 350,
            temperature=0.2,
        )
    except Exception as e:
        logger.exception("Ошибка при обращении к LLM: %s", e)
        reply = (
            "Возникла ошибка при обращении к языковой модели. "
            "Но вот список источников, которые ближе всего к вопросу:\n\n"
        )
        for c in contexts_for_prompt:
            line = f"- {c.get('source')}"
            if c.get("page"):
                line += f", стр. {c.get('page')}"
            reply += line + "\n"

    return reply, contexts


# вопросы для самопроверки

def generate_questions(
    topic_or_query: str,
    n: int = 5,
    subject: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Генерирует вопросы для самопроверки по теме.
    """
    contexts = retrieve(topic_or_query, subject=subject, k=max(TOP_K, n))

    if not contexts:
        logger.info("Контекстов для генерации вопросов не найдено.")
        return []

    contexts_for_prompt = contexts[:5]

    context_blocks = []
    for i, c in enumerate(contexts_for_prompt, start=1):
        label = f"[Фрагмент {i} | источник: {c.get('source')}]"
        context_blocks.append(f"{label}\n{c['text']}")
    context_text = "\n\n".join(context_blocks)

    user_content = (
        f"Тема (запрос студента): {topic_or_query}\n\n"
        f"Фрагменты материалов:\n{context_text}\n\n"
        f"Сгенерируй {n} коротких вопросов для самопроверки по этим материалам. "
        f"К каждому вопросу добавь краткий правильный ответ. "
        "Ответ верни СТРОГО в формате JSON-списка вида:\n"
        '[{"question": "...", "answer": "..."}, ...]\n'
        "Никакого текста вне JSON, без комментариев."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]

    raw = llm_client.generate(
        messages,
        max_tokens=1000,
        temperature=0.3,
    )

    try:
        data = json.loads(raw)
        result: List[Dict[str, str]] = []
        for item in data:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if q:
                result.append({"question": q, "answer": a})
        return result
    except Exception:
        logger.warning("Не удалось распарсить JSON от LLM. Возвращаю сырой текст.")
        return [{"question": "Не удалось распарсить JSON от модели.", "answer": raw}]