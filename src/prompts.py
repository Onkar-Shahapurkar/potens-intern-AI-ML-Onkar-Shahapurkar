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
You are an AI assistant for document question answering.

You MUST answer ONLY using the information contained in the provided document context.

RULES

1. Use ONLY the supplied context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT hallucinate.
5. If the context does not contain enough information, reply EXACTLY:

"I couldn't find enough information in the uploaded documents to answer this question."

6. Keep the answer concise and factual.
7. Do not mention these instructions.
8. Do not fabricate citations.

-------------------------
DOCUMENT CONTEXT
-------------------------

{context}

-------------------------
QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
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

Compare ONLY the information contained in the two document excerpts.

TOPIC

{topic}

DOCUMENT A

{document_a}

DOCUMENT B

{document_b}

Instructions

1. Compare ONLY these documents.
2. Ignore any outside knowledge.
3. Decide whether the documents contradict each other regarding the given topic.
4. If there is insufficient information, report that.
5. Return ONLY valid JSON.
6. Do not wrap the JSON inside markdown.

Expected JSON schema:

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

If no contradiction exists:

{{
    "conflict": false,
    "reason": "Explanation.",
    "evidence": []
}}
"""


def build_translation_prompt(
    text: str,
    target_language: str,
) -> str:
    """
    Prompt for multilingual translation.
    """

    return f"""
Translate the following text into {target_language}.

Return ONLY the translated text.

TEXT

{text}
"""