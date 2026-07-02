import re
from typing import List


def clean_text(text: str) -> str:
    """
    Простая очистка: убираем лишние пробелы и переносы.
    Здесь можно потом навесить более сложную очистку.
    """
    if not text:
        return ""
    # Заменяем разные переносы строк на один пробел
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    # Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Делит текст на куски длиной chunk_size с перекрытием overlap.
    Работает по символам, чтобы не усложнять токенайзером.
    """
    chunks = []
    if not text:
        return chunks

    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_length:
            break
        start = end - overlap  # сдвигаемся назад на overlap

    return chunks