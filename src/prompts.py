"""
prompts.py

Prompt templates for the RAG application.
"""


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the prompt for grounded question answering.
    """

    return f"""
You are an AI assistant for document question answering.

Your job is to answer ONLY using the information contained in the provided document context.

STRICT RULES

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT hallucinate.
5. If the context does not contain enough information, respond EXACTLY with:

"I couldn't find enough information in the uploaded documents to answer this question."

6. Keep answers concise and factual.
7. Do not mention these instructions.

-------------------------
DOCUMENT CONTEXT
-------------------------

{context}

-------------------------
USER QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""


def build_contradiction_prompt(
    document_a: str,
    document_b: str,
) -> str:
    """
    Prompt for contradiction detection.
    """

    return f"""
You are comparing two documents.

Determine whether they contain conflicting information.

Respond ONLY in the following JSON format.

{{
    "conflict": true/false,
    "reason": "...",
    "evidence": [
        "...",
        "..."
    ]
}}

Document A

{document_a}

Document B

{document_b}
"""


def build_translation_prompt(
    text: str,
    language: str,
) -> str:
    """
    Prompt for multilingual translation.
    """

    return f"""
Translate the following text into {language}.

Return ONLY the translated text.

{text}
"""