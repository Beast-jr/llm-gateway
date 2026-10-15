"""List the Gemini models your key can use, and suggest a fast one.

Run from the project root:  python -m scripts.list_gemini_models
(needs GEMINI_API_KEY in .env; the key itself is never printed)
"""
import os
import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()


def main() -> None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is empty or missing in .env. Get a key at https://aistudio.google.com/apikey")
    client = genai.Client(api_key=key)
    models = []
    for m in client.models.list():
        if "generateContent" in (getattr(m, "supported_actions", None) or []):
            models.append(m.name.removeprefix("models/"))
    models.sort()
    print(f"{len(models)} models can answer questions with your key:\n")
    for name in models:
        print(("  * " if "flash" in name else "    ") + name)
    print("\n* = Flash family (fast and cheap). For a fallback, prefer a Flash-Lite model if you see one.")
    print("Put your pick in .env as:  GEMINI_MODEL=<name>")


if __name__ == "__main__":
    main()
