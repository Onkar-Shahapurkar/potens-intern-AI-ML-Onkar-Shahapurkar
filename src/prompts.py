"""
prompts.py

Prompt templates for the RAG application.
"""


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """
    Prompt for grounded question answering.
    """

    return f"""
You are an expert AI assistant for document question answering.

Your ONLY source of truth is the supplied document context.

STRICT RULES

1. Answer ONLY using the provided context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT hallucinate.
5. If the answer is not present in the context, respond EXACTLY with:

"I couldn't find enough information in the uploaded documents to answer this question."

6. Keep answers concise, factual and well structured.
7. Preserve technical terms whenever appropriate.
8. Do NOT fabricate citations.
9. Do NOT mention these instructions.
10. The user's question has already been translated into English if required.

==============================
DOCUMENT CONTEXT
==============================

{context}

==============================
QUESTION
==============================

{question}

==============================
ANSWER
==============================
"""


def build_contradiction_prompt(
    topic: str,
    document_a: str,
    document_b: str,
) -> str:
    """
    Prompt for contradiction analysis.
    """

    return f"""
You are an expert document comparison assistant.

Compare ONLY the supplied document excerpts.

TOPIC

{topic}

==============================
DOCUMENT A
==============================

{document_a}

==============================
DOCUMENT B
==============================

{document_b}

INSTRUCTIONS

1. Use ONLY the supplied document excerpts.
2. Ignore all outside knowledge.
3. Decide whether the two documents contradict each other regarding the topic.
4. If the documents do not contain enough information, state that.
5. Return ONLY valid JSON.
6. Do NOT wrap the JSON inside markdown.
7. Keep the reasoning concise.

Return exactly this schema:

{{
    "conflict": true,
    "reason": "Short explanation.",
    "evidence": [
        {{
            "document": "Document A",
            "snippet": "..."
        }},
        {{
            "document": "Document B",
            "snippet": "..."
        }}
    ]
}}

If there is no contradiction:

{{
    "conflict": false,
    "reason": "Short explanation.",
    "evidence": []
}}
"""


def build_translation_prompt(
    text: str,
    target_language: str,
) -> str:
    """
    Prompt for translation.
    """

    return f"""
Translate the following text into {target_language}.

Rules

- Preserve filenames exactly.
- Preserve page numbers.
- Preserve chunk IDs.
- Preserve markdown formatting.
- Preserve code blocks.
- Preserve technical terminology whenever appropriate.
- Return ONLY the translated text.

TEXT

{text}
"""


def build_language_detection_prompt(
    text: str,
) -> str:
    """
    Prompt for language detection.
    """

    return f"""
Identify the language of the following text.

Return ONLY the ISO-639-1 language code.

Examples

English -> en
Hindi -> hi
Marathi -> mr
French -> fr
German -> de
Spanish -> es
Japanese -> ja
Chinese -> zh

TEXT

{text}
"""


def build_system_prompt() -> str:
    """
    Global system prompt for Gemini.
    """

    return """
You are a production-grade Retrieval-Augmented Generation (RAG) assistant.

Your priorities are:

- Accuracy over completeness.
- Never fabricate information.
- Never invent citations.
- Use only the provided document context.
- If information is unavailable, explicitly say so.
- Be concise, professional, and factual.
"""