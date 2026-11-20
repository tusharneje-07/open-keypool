"""Use open-keypool with an httpx client."""

import os

import httpx

from open_keypool import KeyPool, KeyPoolAuth


pool = KeyPool(
    [os.environ["OPENAI_KEY_1"], os.environ["OPENAI_KEY_2"]],
    provider="openai",
)

with httpx.Client(auth=KeyPoolAuth(pool)) as client:
    response = client.get("https://api.openai.com/v1/models")
    response.raise_for_status()
    print(response.json())
