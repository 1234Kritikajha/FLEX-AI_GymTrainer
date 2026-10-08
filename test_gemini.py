import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY is missing from .env")

try:
    from google import genai
except ImportError:
    raise SystemExit("Install dependencies first: pip install -r requirements.txt")

client = genai.Client(api_key=key)

response = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
    contents="Reply with exactly: FLEX Gemini connection successful.",
)

print(response.text)
