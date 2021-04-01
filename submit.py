import os

from canvasapi import Canvas
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
    #     assignment_name="Assignment Futures Spreads",
    #     course_name="FINM 33150 1, 2 (Spring 2021) Regression Analysis and Quantitative Trading Strategies",
    #     canvas_url=url,
    #     canvas_token=token,
    # )

    # sm.get_canvas_objects()
    # print(sm.course)
    # sm.atomic_quiz_submit(answers_fn, verbose=True)
    # sm.assignment_submit(verbose=True)

    course_id = 33395
    canvas = Canvas(url, token)
    course = canvas.get_course(course_id)

    # for f in course.get_files():
    #     if f.folder_id == 842539:
    #         file = course.get_file(f.id)
    #         file.download(file.filename)

    # file = course.get_file(5394394)
    # file.download(file.filename)

    folder = course.get_folder(842539)

    for f in folder.get_files():
        try:
            f.download(f"notebooks/{f.filename}")
            print("downloaded", f.filename)
        except:
            print("failed", f.filename, f.url)
