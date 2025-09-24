import jwt
#import datetime
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file
SECRET_KEY = os.getenv("SECRET_KEY")



#def generate_token(user_id):
#    payload = {
#        "user_id": user_id,
#        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # token expiry
#    }
#    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def generate_token(user_id, role , username):
    payload = {
        "user_id": user_id,
        "role": role,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


#def decode_token(token):
#    try:
#        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#        return payload["user_id"]
#    except jwt.ExpiredSignatureError:
#        return None
#    except jwt.InvalidTokenError:
#        return None

def decode_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"], payload.get("role", "user"), payload.get("username"," ")  # default role = user
    except jwt.ExpiredSignatureError:
        return None, None, None
    except jwt.InvalidTokenError:
        return None, None, None



# pip install PyJWT


