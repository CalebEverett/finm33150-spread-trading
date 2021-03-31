import os

from canvasapi.quiz import QuizSubmissionQuestion
from typing import List, Dict

from csci_utils.canvas_utils import SubmissionManager
import futures_spreads


def answers_fn(questions: List[QuizSubmissionQuestion]) -> List[Dict]:
    """Returns answers to Canvas quiz questions."""

    answers = [dict(id=q.id, answer=q.answer) for q in questions]

    # Question 1
    for k in questions[0].answer.keys():
        answers[0]["answer"][k] = 42

    # Question 2
    answers[1]["answer"] = 42

    return answers


if __name__ == "__main__":
    url = os.getenv("CANVAS_URL")
    token = os.getenv("CANVAS_TOKEN")

    # sm = SubmissionManager(
    #     pset=1, min_quiz_score=10, canvas_url=url, canvas_token=token
    # )

    # sm.get_canvas_objects()
    # sm.atomic_quiz_submit(answers_fn, verbose=True)
    # sm.assignment_submit(verbose=True)
