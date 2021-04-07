import json
import os
import re
from pathlib import Path

from canvasapi import Canvas
from git import Repo

repo = Repo(".")

submit_file = Path(f"submissions/ceverett_{repo.head.commit.hexsha[:8]}.ipynb")


def write_submit_file():
    code_file = "futures_spreads.ipynb"

    with open(code_file, "r") as nb:
        nb_json = json.load(nb)

    for i, cell in enumerate(nb_json["cells"]):
        if cell["cell_type"] == "code":

            if len(cell["source"]) == 1:
                groups = re.search(r"(?<=\<include-)(.*?)(?=\>)", cell["source"][0])

                if groups:
                    with open(groups.group(0), "r") as m:
                        nb_json["cells"][i]["source"] = (
                            m.readlines() + nb_json["cells"][i]["source"]
                        )

            new_lines = []
            for code_line in cell["source"]:
                if "from futures_spreads import utils" not in code_line:
                    new_lines.append(code_line.replace("utils.", ""))

            nb_json["cells"][i]["source"] = new_lines

    if submit_file.exists():
        print("deleting")
        submit_file.unlink()

    with open(submit_file, "w") as f:
        json.dump(nb_json, f)


if __name__ == "__main__":
    url = os.getenv("CANVAS_URL")
    token = os.getenv("CANVAS_TOKEN")

    course_id = 33395
    canvas = Canvas(url, token)
    course = canvas.get_course(course_id)

    write_submit_file()

    assignment = course.get_assignment(340816)

    submission = assignment.submit(
        dict(
            submission_type="online_upload",
        ),
        file=submit_file,
    )
