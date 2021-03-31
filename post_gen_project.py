import json
import os
from base64 import b64encode
from pathlib import Path
from urllib.parse import urljoin

import requests
from git import Repo
from nacl import encoding, public

DOTENV_PATH = "/home/caleb/dotenvs/psets/.env"
INIT_REPO = "yes" == "yes"
CREATE_SECRETS = "no" == "yes"
MERGE_REMOTE = "no" == "yes"
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

BASE_URL = "https://api.github.com"
REPO_NAME = "CalebEverett/finm33150-futures-spreads"


def encrypt(public_key: str, secret_value: str) -> str:
    """Encrypt a Unicode string using the public key."""
    public_key = public.PublicKey(public_key.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return b64encode(encrypted).decode("utf-8")


def get_public_key():
    public_key_path = f"/repos/{REPO_NAME}/actions/secrets/public-key"
    headers = {"Authorization": f"token {GITHUB_ACCESS_TOKEN}"}
    r = requests.get(urljoin(BASE_URL, public_key_path), headers=headers)
    return r.json()["key_id"], r.json()["key"]


def create_secret(secret_name: str, secret_value: str, key_id: str, key) -> None:
    encrypted_value = encrypt(key, secret_name)
    secrets_path = f"/repos/{REPO_NAME}/actions/secrets/{secret_name}"
    headers = {
        "Authorization": f"token {GITHUB_ACCESS_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = json.dumps({"key_id": key_id, "encrypted_value": encrypted_value})
    r = requests.put(urljoin(BASE_URL, secrets_path), headers=headers, data=data)
    if r.status_code != 201:
        raise Exception("Problem creating secret.")


if DOTENV_PATH:
    Path(".env").write_text(Path(DOTENV_PATH).read_text())

    if CREATE_SECRETS:
        key_id, key = get_public_key()

        with open(DOTENV_PATH, "r") as ef:
            env_lines = ef.readlines()

            with open(".github/workflows/build.yml", "r+") as bf:
                build_lines = bf.readlines()
                for bi, bl in enumerate(build_lines):
                    if "env:" in bl:
                        break

                for ei, el in enumerate(env_lines):
                    name, value = el.split("=")
                    create_secret(name, value, key_id, key)

                    insert_string = (
                        f"  {name}: "
                        + "${{ secrets."
                        + name
                        + " }}\n"
                    )
                    build_lines.insert(bi + ei + 1, insert_string)

                bf.seek(0)
                bf.writelines(build_lines)

if INIT_REPO:
    repo = Repo.init()
    repo.git.add(all=True)
    repo.index.commit("Add initial project skeleton.")

if MERGE_REMOTE:
    repo.create_remote("origin", url="git@github.com:CalebEverett/finm33150-futures-spreads.git")
    remote = repo.remote("origin")
    remote.fetch()
    repo.git.merge("origin/master", allow_unrelated_histories=True)
    remote.push(refspec="master:master")
