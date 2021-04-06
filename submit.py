import json
import os

import requests
from canvasapi import Canvas
from git import Repo

if __name__ == "__main__":
    url = os.getenv("CANVAS_URL")
    token = os.getenv("CANVAS_TOKEN")

    course_id = 33395
    canvas = Canvas(url, token)
    course = canvas.get_course(course_id)

    repo = Repo(".")
    latest_tag = str(repo.tags[-1])
    comment = dict(version=latest_tag, is_dirty=repo.is_dirty())

    url = (
        f"https://{os.getenv('ACCESS_TOKEN')}:x-oauth-basic"
        f"@github.com/CalebEverett/finm33150-futures-spreads"
        f"/archive/refs/tags/{latest_tag}.zip"
    )

    zipfilepath = f"submissions/ceverett_{latest_tag}.zip"
    print(zipfilepath)

    with open(zipfilepath, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    assignment = course.get_assignment(340816)

    submission = assignment.submit(
        dict(
            submission_type="online_upload",
        ),
        comment=dict(text_comment=json.dumps(comment)),
        file=zipfilepath,
    )

    print(submission)
