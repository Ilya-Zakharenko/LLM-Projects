from typing import List, Dict, Optional
import logging

from openai import OpenAI

from .config import LLM_API_KEY, LLM_MODEL_NAME, LOG_PATH

# настройка логов
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not LLM_API_KEY:
            raise RuntimeError(
                "Не задан API-ключ для LLM. "
                "Укажи CLOUD_API_KEY или OPENAI_API_KEY в файле .env"
            )
        _client = OpenAI(api_key=LLM_API_KEY)
    return _client


def generate(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    client = get_client()
    model_name = model or LLM_MODEL_NAME

    logger.info(
        "Запрос к LLM model=%s, max_tokens=%s, temperature=%s",
        model_name,
        max_tokens,
        temperature,
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    text = completion.choices[0].message.content.strip()
    return text
