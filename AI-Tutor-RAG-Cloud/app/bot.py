import logging
import asyncio
from html import escape
from pathlib import Path
from typing import Dict, List

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
    filters,
)

from .config import TELEGRAM_BOT_TOKEN, LOG_PATH, USER_DATA_DIR
from . import rag_pipeline

# логи
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

QUIZ_TOPIC = 1
UPLOAD_FILES = 10
# лимиты
MAX_FILE_SIZE = 20 * 1024 * 1024      
MAX_CHUNKS_PER_FILE = 4000           


# хранилище квизов
user_quizzes: Dict[int, List[dict]] = {}


# вспомог. на всякий случ. 

def get_user_subject(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Текущий выбранный предмет пользователя (или None, если 'все')."""
    return context.user_data.get("subject")


def get_user_answer_style(context: ContextTypes.DEFAULT_TYPE) -> str:
    """Текущий режим ответа пользователя: 'detailed' или 'short'."""
    return context.user_data.get("answer_style", "detailed")


# команды:

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    subject = get_user_subject(context)
    style = get_user_answer_style(context)

    subject_text = (
        "Предмет пока не выбран — ищу по всему корпусу."
        if subject is None
        else f"Текущий предмет: {subject}"
    )
    style_text = "Режим ответов: подробные." if style == "detailed" else "Режим ответов: краткие."

    text = (
        "Привет! Я AI-репетитор.\n\n"
        "Я помогаю тебе по ТВОИМ любым материалам:\n"
        "1) Используй /upload и пришли PDF/DOCX/TXT.\n"
        "2) Нажми /done, когда закончишь.\n"
        "3) Выбери предмет /subject (обычно это «Мои материалы»).\n\n"
        "Дальше можно задавать вопросы или запускать /quiz.\n\n"
        f"{subject_text}\n{style_text}\n\n"
        "Попробуй, например, загрузить конспект лекции и спросить:\n"
        "  «Объясни это...»"
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Я работаю на основе локальных учебных материалов (RAG-подход).\n\n"
        "Команды:\n"
        "• /start — кратко рассказать, что я умею\n"
        "• /help — помощь\n"
        "• /upload — загрузить свои материалы (PDF/DOCX/TXT)\n"
        "• /done — закончить загрузку материалов\n"
        "• /quiz — вопросы для самопроверки по теме\n"
        "• /subject — выбрать предмет (в т.ч. «Мои материалы»)\n"
        "• /mode — стиль ответа (краткий/подробный)\n\n"
        "Любой обычный текст — это вопрос к репетитору."
        "• /status — посмотреть, сколько материалов уже загружено\n"
    )
    await update.message.reply_text(text)



async def subject_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показать список предметов для выбора."""
    subjects = rag_pipeline.list_subjects()
    user = update.effective_user
    user_subject_key = f"user_{user.id}"

    buttons: List[List[InlineKeyboardButton]] = [
        [InlineKeyboardButton("Все предметы", callback_data="subject:__all__")]
    ]

    for subj in subjects:
        if subj.startswith("user_"):
            continue
        label = subj if subj != "default" else "Общий корпус (default)"
        buttons.append(
            [InlineKeyboardButton(label, callback_data=f"subject:{subj}")]
        )

    # материалы если есть
    if user_subject_key in subjects:
        buttons.append(
            [InlineKeyboardButton("Мои материалы", callback_data=f"subject:{user_subject_key}")]
        )

    if len(buttons) == 1:
        await update.message.reply_text(
            "Пока нет ни одного предмета. "
            "Загрузи материалы командой /upload."
        )
        return

    markup = InlineKeyboardMarkup(buttons)
    await update.message.reply_text(
        "Выбери предмет, с которым будем работать по умолчанию:",
        reply_markup=markup,
    )


async def subject_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data.startswith("subject:"):
        return

    value = data.split(":", 1)[1]

    if value == "__all__":
        context.user_data.pop("subject", None)
        text = "Окей, теперь я буду искать ответы по всему корпусу, без ограничения по предмету."
    else:
        context.user_data["subject"] = value
        label = value
        if value == "default":
            label = "Общий корпус (default)"
        if value.startswith("user_"):
            label = "Мои материалы"
        text = f"Предмет по умолчанию установлен: {label}"

    await query.edit_message_text(text=text)



async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Выбор стиля ответов: кратко / подробно."""
    style = get_user_answer_style(context)
    current = (
        "Сейчас режим: подробные ответы."
        if style == "detailed"
        else "Сейчас режим: краткие ответы."
    )

    buttons = [
        [InlineKeyboardButton("Подробные ответы", callback_data="mode:detailed")],
        [InlineKeyboardButton("Краткие ответы", callback_data="mode:short")],
    ]
    markup = InlineKeyboardMarkup(buttons)

    await update.message.reply_text(
        current + "\n\nВыбери, как отвечать дальше:",
        reply_markup=markup,
    )


async def mode_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data.startswith("mode:"):
        return

    value = data.split(":", 1)[1]
    if value not in ("detailed", "short"):
        return

    context.user_data["answer_style"] = value

    if value == "detailed":
        text = "Готов. Теперь буду давать более развёрнутые ответы."
    else:
        text = "Окей. Теперь буду отвечать короче и по делу (3–5 предложений)."

    await query.edit_message_text(text=text)


# ---------- /UPLOAD ----------
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Показывает статистику по загруженным материалам пользователя."""
    user = update.effective_user

    stats = await asyncio.to_thread(
        rag_pipeline.get_user_stats,
        user.id,
    )

    num_chunks = stats["num_chunks"]
    files = stats["files"]

    if num_chunks == 0:
        await update.message.reply_text(
            "Пока у тебя нет загруженных материалов.\n"
            "Используй /upload, чтобы добавить PDF/DOCX/TXT, "
            "а затем выбери «Мои материалы» через /subject."
        )
        return

    files_str = "\n".join(f"- {name}" for name in files) if files else "—"

    await update.message.reply_text(
        "Твои материалы:\n"
        f"• фрагментов в базе: {num_chunks}\n"
        f"• файлов: {len(files)}\n\n"
        f"{files_str}"
    )

async def upload_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Старт загрузки материалов."""
    context.user_data["uploaded_files"] = []
    await update.message.reply_text(
        "Отправь мне файлы с материалами (PDF, DOCX или TXT).\n"
        "Можно несколько — по одному файлу в сообщении.\n\n"
        "Когда закончишь — напиши /done.\n"
        "Чтобы отменить — /cancel."
    )
    return UPLOAD_FILES


async def upload_file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Принимаем файл, сохраняем и добавляем в индекс с отображением прогресса."""
    user = update.effective_user
    doc = update.message.document

    if not doc:
        await update.message.reply_text(
            "Пришли, пожалуйста, файл как документ (PDF, DOCX или TXT)."
        )
        return UPLOAD_FILES

    filename = doc.file_name or "file"
    lower = filename.lower()
    if not (lower.endswith(".pdf") or lower.endswith(".docx") or lower.endswith(".txt")):
        await update.message.reply_text(
            "Пока я умею работать только с PDF, DOCX и TXT. "
            "Попробуй переслать в одном из этих форматов."
        )
        return UPLOAD_FILES

    # проверка размера файла
    if doc.file_size and doc.file_size > MAX_FILE_SIZE:
        mb = doc.file_size / (1024 * 1024)
        await update.message.reply_text(
            f"Файл слишком большой ({mb:.1f} МБ). "
            "Сейчас я могу обрабатывать файлы до 20 МБ.\n"
            "Попробуй разделить его на несколько частей (например, по главам)."
        )
        return UPLOAD_FILES

    # сохр. файл на диск 
    user_dir = USER_DATA_DIR / str(user.id)
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / filename

    file = await doc.get_file()
    await file.download_to_drive(str(file_path))

    # 0%
    await update.message.reply_text(
        f"Файл «{filename}» получен, начинаю обработку… (0%)"
    )
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    # шаг 1. извлечение текста и разбиение на чанки
    chunk_texts = await asyncio.to_thread(
        rag_pipeline.prepare_document_chunks,
        file_path,
    )

    if not chunk_texts:
        await update.message.reply_text(
            "Не удалось извлечь текст из этого файла или он пустой :("
        )
        return UPLOAD_FILES

    # если слишком много чанков
    if len(chunk_texts) > MAX_CHUNKS_PER_FILE:
        await update.message.reply_text(
            "Файл очень большой по содержанию — получилось больше "
            f"{MAX_CHUNKS_PER_FILE} фрагментов.\n"
            "Попробуй разделить его на несколько файлов (например, по главам), "
            "чтобы ответы были точнее и индекс не разрастался слишком сильно."
        )
        return UPLOAD_FILES

    # 60–70%
    await update.message.reply_text(
        f"Текст извлечён и разбит на фрагменты (≈70%). "
        f"Фрагментов: {len(chunk_texts)}. "
        "Добавляю их в базу знаний… (≈90%)"
    )
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    # Шаг 2. считаем эмбеддинги и добавляем в индекс
    num_chunks = await asyncio.to_thread(
        rag_pipeline.add_chunks_to_index,
        chunk_texts,
        f"user_{user.id}",
        filename,
    )

    if num_chunks == 0:
        await update.message.reply_text(
            "Не получилось добавить фрагменты в индекс — что-то пошло не так."
        )
        return UPLOAD_FILES

    uploaded = context.user_data.get("uploaded_files", [])
    uploaded.append(filename)
    context.user_data["uploaded_files"] = uploaded

    # 100%
    await update.message.reply_text(
        f"Готово! Я добавил {num_chunks} фрагментов из файла «{filename}» (100%).\n"
        "Можешь отправить ещё файл или закончить командой /done."
    )

    return UPLOAD_FILES


async def upload_wait_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Если в режиме /upload прислали не файл."""
    await update.message.reply_text(
        "Сейчас я жду файлы с материалами.\n"
        "Отправь PDF/DOCX/TXT или напиши /done, чтобы закончить."
    )
    return UPLOAD_FILES


async def upload_done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Завершаем загрузку материалов."""
    user = update.effective_user
    uploaded = context.user_data.get("uploaded_files", [])

    if not uploaded:
        await update.message.reply_text(
            "Ты пока не загрузил ни одного файла. "
            "Если захочешь — просто запусти /upload ещё раз."
        )
        return ConversationHandler.END

    # преключаем предмет на мои материалы. 
    context.user_data["subject"] = f"user_{user.id}"

    await update.message.reply_text(
        "Отлично, материалы сохранены! "
        "Теперь я буду по умолчанию использовать твои файлы "
        "как предмет «Мои материалы».\n\n"
        "Можешь задавать вопросы или запускать /quiz."
    )
    return ConversationHandler.END


async def upload_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Окей, отменяем загрузку материалов.")
    return ConversationHandler.END


# квиз

async def quiz_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Старт /quiz — спрашиваем тему."""
    await update.message.reply_text(
        "По какой теме сгенерировать вопросы для самопроверки?\n"
        "Например: «производные», «матрицы», «регрессия».\n"
        "Я сгенерирую 5 вопросов, а ответы спрячу под спойлер."
    )
    return QUIZ_TOPIC


async def quiz_generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Получаем тему и генерируем вопросы."""
    user = update.effective_user
    topic = update.message.text.strip()
    subject = get_user_subject(context)

    await update.message.reply_text("Думаю над вопросами, пару секунд…")
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    questions = await asyncio.to_thread(
        rag_pipeline.generate_questions,
        topic,
        5,        # количество вопросов
        subject,  # предмет, если выбран
    )

    if not questions:
        await update.message.reply_text(
            "Не смог найти подходящих материалов для этой темы. "
            "Сначала загрузите файлы через /upload или уточните формулировку."
        )
        return ConversationHandler.END

    user_quizzes[user.id] = questions

    lines = ["Вот вопросы для самопроверки:\n"]
    for i, qa in enumerate(questions, start=1):
        q = escape(qa["question"])
        a = escape(qa["answer"])
        lines.append(
            f"{i}. {q}\n"
            f"   Ответ: <tg-spoiler>{a}</tg-spoiler>"
        )

    await update.message.reply_text(
        "\n\n".join(lines),
        parse_mode=ParseMode.HTML,
    )
    return ConversationHandler.END


async def quiz_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Окей, отменяем квиз.")
    return ConversationHandler.END


# обраотка текста:

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Любой текст без команды — это вопрос к репетитору."""
    if not update.message or not update.message.text:
        return

    question = update.message.text.strip()
    user = update.effective_user

    subject = get_user_subject(context)
    answer_style = get_user_answer_style(context)

    logger.info(
        "Вопрос от %s (%s): %s (subject=%s, style=%s)",
        user.id,
        user.username,
        question,
        subject,
        answer_style,
    )

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    answer_text, contexts = await asyncio.to_thread(
        rag_pipeline.answer,
        question,
        subject,
        answer_style,
    )

    # инфа по источникам
    if contexts:
        sources_lines = []
        used = set()
        for c in contexts:
            key = (c.get("source"), c.get("page"))
            if key in used:
                continue
            used.add(key)
            line = f"- {c.get('source')}"
            if c.get("page"):
                line += f", стр. {c.get('page')}"
            sources_lines.append(line)

        sources_block = "\n\nИсточники:\n" + "\n".join(sources_lines)
    else:
        sources_block = "\n\n(Подходящих источников не найдено в корпусе. "
        sources_block += "Загрузи материалы через /upload.)"

    full_reply = answer_text + sources_block
    await update.message.reply_text(full_reply)


# настройка команд. меню:

async def post_init(application: Application) -> None:
    """Вызывается один раз после запуска приложения, чтобы выставить команды бота."""
    commands = [
        BotCommand("start", "Запустить бота"),
        BotCommand("help", "Помощь"),
        BotCommand("upload", "Загрузить свои материалы"),
        BotCommand("done", "Закончить загрузку материалов"),
        BotCommand("quiz", "Вопросы для самопроверки"),
        BotCommand("subject", "Выбрать предмет"),
        BotCommand("mode", "Стиль ответа"),
        BotCommand("status", "Статистика по моим материалам"),
    ]
    await application.bot.set_my_commands(commands)


# запуск бота

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError(
            "Не задан TELEGRAM_BOT_TOKEN. "
            "Добавь его в .env (TELEGRAM_BOT_TOKEN=...)"
        )

    rag_pipeline.load_index()

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("subject", subject_command))
    application.add_handler(CommandHandler("mode", mode_command))
    

    quiz_conv = ConversationHandler(
        entry_points=[CommandHandler("quiz", quiz_entry)],
        states={
            QUIZ_TOPIC: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, quiz_generate)
            ],
        },
        fallbacks=[CommandHandler("cancel", quiz_cancel)],
    )
    application.add_handler(quiz_conv)

    upload_conv = ConversationHandler(
        entry_points=[CommandHandler("upload", upload_entry)],
        states={
            UPLOAD_FILES: [
                MessageHandler(filters.Document.ALL & ~filters.COMMAND, upload_file_handler),
                CommandHandler("done", upload_done),
                CommandHandler("cancel", upload_cancel),
                MessageHandler(filters.TEXT & ~filters.COMMAND, upload_wait_text),
            ],
        },
        fallbacks=[CommandHandler("cancel", upload_cancel)],
    )
    application.add_handler(upload_conv)

    application.add_handler(CallbackQueryHandler(subject_callback, pattern=r"^subject:"))
    application.add_handler(CallbackQueryHandler(mode_callback, pattern=r"^mode:"))

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    logger.info("Бот запущен, ждём сообщения.")
    application.run_polling()


if __name__ == "__main__":
    main()