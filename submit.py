import os

from canvasapi import Canvas

if __name__ == "__main__":
    url = os.getenv("CANVAS_URL")
    token = os.getenv("CANVAS_TOKEN")

    # sm.get_canvas_objects()
    # print(sm.course)
    # sm.atomic_quiz_submit(answers_fn, verbose=True)
    # sm.assignment_submit(verbose=True)

    course_id = 33395
    canvas = Canvas(url, token)
    course = canvas.get_course(course_id)
    file = course.get_file(5403112)
    file.download(file.filename)

    # for f in course.get_files():
    #     if f.folder_id == 842539:
    #         file = course.get_file(f.id)
    #         file.download(file.filename)

    # file = course.get_file(5394394)
    # file.download(file.filename)

    # folder = course.get_folder(842539)

    # for f in folder.get_files():
    #     try:
    #         f.download(f"notebooks/{f.filename}")
    #         print("downloaded", f.filename)
    #     except:
    #         print("failed", f.filename, f.url)
