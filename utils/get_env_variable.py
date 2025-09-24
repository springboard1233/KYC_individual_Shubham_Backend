from dotenv import load_dotenv
import os

load_dotenv()  # loads .env file

SECRET_KEY = os.getenv("SECRET_KEY")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")

print("SECRET_KEY:", SECRET_KEY)
print("GMAIL_PASSWORD:", GMAIL_PASSWORD)
