from flask import request, jsonify
#from functools import wraps
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file
SECRET_KEY = os.getenv("SECRET_KEY")



from functools import wraps
import jwt

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid token"}), 401

        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        if payload.get("role") != "admin":
            return jsonify({"error": "Admins only"}), 403

        return f(*args, **kwargs)
    return wrapper
