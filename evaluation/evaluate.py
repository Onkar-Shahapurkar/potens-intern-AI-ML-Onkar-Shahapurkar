"""
evaluation/evaluate.py

Evaluation script for the RAG pipeline.

Metrics
-------
- Answer Accuracy
- Retrieval Accuracy (Top-1 / Top-3 / Top-5)
- Average Confidence
- Average Latency
- Citation Coverage

Run:
    python evaluation/evaluate.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

from src.citations import CitationFormatter
from src.llm import LLMService
from src.retrieval import Retriever

QUESTIONS_FILE = Path("evaluation/questions.json")
GROUND_TRUTH_FILE = Path("evaluation/ground_truth.json")


class Evaluator:
    """
    Evaluates the RAG pipeline.
    """

    def __init__(self):

        self.retriever = Retriever()
        self.llm = LLMService()

    def load_questions(self):

        with open(
            QUESTIONS_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def load_ground_truth(self):

        with open(
            GROUND_TRUTH_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    def evaluate(self):

        questions = self.load_questions()
        ground_truth = self.load_ground_truth()

        if not questions:

            print("=" * 40)
            print("RAG Evaluation Report")
            print("=" * 40)
            print("No evaluation questions found.")
            print("=" * 40)

            return

        answer_correct = 0

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0

        confidence_scores = []
        response_times = []
        citation_coverages = []

        for sample in questions:

            question = sample["question"]
            expected_doc = sample["document_id"]

            start = time.perf_counter()

            retrieval = self.retriever.retrieve_for_generation(
                query=question,
                top_k=5,
            )

            citations = CitationFormatter.format_citations(
                retrieval["chunks"]
            )

            result = self.llm.generate_answer_with_confidence(
                question=question,
                context=retrieval["context"],
                retrieved_chunks=retrieval["chunks"],
                citations=citations,
            )

            elapsed = time.perf_counter() - start

            response_times.append(elapsed)

            confidence_scores.append(
                result["confidence"]
            )

            citation_coverages.append(
                len(citations)
            )

            answer = result["answer"].lower()

            expected_answer = ground_truth.get(
                sample["id"],
                "",
            ).lower()

            if expected_answer and expected_answer in answer:

                answer_correct += 1

            retrieved_docs = [
                chunk["document_id"]
                for chunk in retrieval["chunks"]
            ]

            if (
                len(retrieved_docs) >= 1
                and retrieved_docs[0] == expected_doc
            ):

                top1_correct += 1

            if expected_doc in retrieved_docs[:3]:

                top3_correct += 1

            if expected_doc in retrieved_docs[:5]:

                top5_correct += 1

        total = len(questions)

        print("=" * 50)
        print("RAG Evaluation Report")
        print("=" * 50)

        print(f"Questions Tested     : {total}")

        print(
            f"Answer Accuracy      : {(answer_correct / total) * 100:.2f}%"
        )

        print(
            f"Top-1 Retrieval      : {(top1_correct / total) * 100:.2f}%"
        )

        print(
            f"Top-3 Retrieval      : {(top3_correct / total) * 100:.2f}%"
        )

        print(
            f"Top-5 Retrieval      : {(top5_correct / total) * 100:.2f}%"
        )

        print(
            f"Average Confidence   : {mean(confidence_scores):.2f}%"
        )

        print(
            f"Average Latency      : {mean(response_times):.2f} sec"
        )

        print(
            f"Average Citations    : {mean(citation_coverages):.2f}"
        )

        print("=" * 50)


def main():

    evaluator = Evaluator()

    evaluator.evaluate()


if __name__ == "__main__":

    main()