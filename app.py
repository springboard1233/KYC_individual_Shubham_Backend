from flask import Flask, request, jsonify, render_template ,send_from_directory
import os, random, time
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib, ssl
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from bson import ObjectId
from datetime import datetime#, timezone, timedelta
from pytz import timezone


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


#import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


from pdf2image import convert_from_bytes
from pdf2image import convert_from_path

from rapidfuzz import fuzz

import joblib

load_dotenv()  # loads .env file

#from dotenv import load_dotenv
#from sendgrid import SendGridAPIClient
#from sendgrid.helpers.mail import Mail


# Import your OCR functions
from ocr.ocr_ex import extract_rawtxt
from ocr.f import extract_details
from ocr.ocr_ex import extract_pantxt
from ocr.f import pan_details   # <-- create this for PAN


from utils.utils import verhoeff_check, validate_pan, hash_value, check_document, compute_fraud_score
from utils.jwt_session import *
from utils.admin import *

# ---------------- Flask App ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- Load ENV ----------------
#load_dotenv()

# ---------------- MongoDB Setup ----------------

uri = "mongodb+srv://starkadam999_db_user:7drXY49M@kyc-cluster.cuxa0fo.mongodb.net/?retryWrites=true&w=majority&appName=kyc-cluster"

client = MongoClient(uri)

#client = MongoClient("mongodb://localhost:27017/")
db = client['aadhaar_db']
aadhaar_audit = db['aadhaar_audit_trail']
pan_audit = db['pan_audit_trail']
users_collection = db['users']
otp_collection = db['otp_store']  # new collection for OTPs
#kyc_col = db["kyc_submissions"]
kyc_col = db["new_table"]


#db = client["kyc_db"]


# ---------- HTML Page Routes ----------
@app.route('/')
def index():
    return render_template('main.html')

#@app.route('/login')
#def login_page():
#    return render_template('login.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/admin-dashboard')
def admin_page():
    return render_template('admin.html')

@app.route("/api/logout", methods=["POST"])
def logout():
    # For stateless JWT, nothing to do on server
    return jsonify({"message": "Logged out successfully"})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)




# ---------- Helper: Send Email with SendGrid ----------
#def send_email1(receiver_email, otp):
#    message = Mail(
#        from_email="shubham.19jdai037@gmail.com",   # must be verified in SendGrid
#        to_emails=receiver_email,
#        subject="Your OTP Code",
#        html_content=f"<strong>Your OTP is {otp}. It is valid for 2 minutes.</strong>"
#    )
#    try:
#        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
#        response = sg.send(message)
#        print("Email sent:", response.status_code)  # should be 202
#        return True
#    except Exception as e:
#        print("Error sending email:", e)
#        return False

# save files to upload


# Pretrained CNN (ResNet50 as feature extractor)
#cnn_model = models.resnet50(pretrained=True)
cnn_model = resnet50(weights=ResNet50_Weights.DEFAULT)
cnn_model.fc = nn.Identity()
cnn_model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        emb = cnn_model(img)
    return emb.numpy()





# Precompute reference embeddings
AADHAAR_TEMPLATE = get_embedding("layout/aadhaar_card.png")
PAN_TEMPLATE = get_embedding("layout/pan_card.png")




from werkzeug.utils import secure_filename

def save_user_file(file, user_id, doc_type):
    # doc_type = 'aadhaar' or 'pan'
    filename = secure_filename(f"{user_id}_{doc_type}.jpg")
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filename , filepath



def bool_from_valid(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.lower()
        if "not valid" in v or "invalid" in v:
            return False
        if "valid" in v:
            return True
    return False




# --------------------------
# GNN Model
# --------------------------
class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x #F.softmax(x, dim=1)


# --------------------------
# Build Graph with Encoding
# --------------------------
def build_graph_from_docs(docs):
    node_features = []
    labels = []
    edge_index = [[], []]

    # Risk level mapping
    #risk_map = {"Low": 0, "Medium": 1, "High": 2} # may be wrong
    risk_map = {"Low": 0, "High": 1}

    for doc in docs:
        aadhaar = doc.get("aadhaar", {})
        pan = doc.get("pan", {})

        # --- dob → age ---
        dob_str = aadhaar.get("dob", None)
        age = 0
        if dob_str:
            try:
                dob = datetime.strptime(dob_str, "%d-%m-%Y")
                today = datetime.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            except:
                age = 0  # fallback if parsing fails

        # --- risk level to int ---
        risk_level = doc.get("risk_level", "Low")
        risk_int = risk_map.get(risk_level, 0)

        # --- feature vector ---
        features = [
            doc.get("overall_score", 0),
            aadhaar.get("fraud_score", 0),
            pan.get("fraud_score", 0),
            1 if aadhaar.get("valid") else 0,
            1 if pan.get("valid") else 0,
            1 if aadhaar.get("duplicate_found") else 0,
            1 if pan.get("duplicate_found") else 0,
            1 if aadhaar.get("AadhaarName_username") == "Match" else 0,
            1 if pan.get("panName_username") == "Match" else 0,
            1 if doc.get("AadhaarName_panName") == "Match" else 0,
            len(aadhaar.get("aadhar_manipulation", [])),
            len(pan.get("pan_manipulation", [])),
            age,
            risk_int,
        ]

        node_features.append(features)

        # label from risk level (binary: High=1 else 0)
        labels.append(1 if risk_level == "High" else 0)

    # --- edges ---
    for i, u in enumerate(docs):
        for j, v in enumerate(docs):
            if i >= j:
                continue

            # Aadhaar duplicate
            if (
                u.get("aadhaar", {}).get("number_hash")
                and u["aadhaar"]["number_hash"] == v.get("aadhaar", {}).get("number_hash")
            ):
                edge_index[0] += [i, j]
                edge_index[1] += [j, i]

            # PAN duplicate
            if (
                u.get("pan", {}).get("number_hash")
                and u["pan"]["number_hash"] == v.get("pan", {}).get("number_hash")
            ):
                edge_index[0] += [i, j]
                edge_index[1] += [j, i]

            # Fuzzy name similarity
            sim = fuzz.token_set_ratio(u.get("user_entered_name", ""), v.get("user_entered_name", ""))
            if sim > 85:
                edge_index[0] += [i, j]
                edge_index[1] += [j, i]

    # --- build PyG Data object ---
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


# --------------------------
# Run Pipeline
# --------------------------
def run_gnn_pipeline(docs, epochs=200):
    data = build_graph_from_docs(docs)

    model = FraudGNN(in_channels=data.num_node_features, hidden_channels=16, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    class_counts = torch.bincount(data.y) # updated but not run
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)



    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)

        #loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    with torch.no_grad():
        predictions = model(data).argmax(dim=1)

    return predictions, data.edge_index


def run_gnn_inference_for_user(user_id):
    # 1. Load trained model
    model = FraudGNN(in_channels=14, hidden_channels=16, out_channels=2)
    model.load_state_dict(torch.load("fraud_gnn_model.pt"))
    model.eval()

    # 2. Fetch all docs
    docs = list(kyc_col.find({}))
    if not docs:
        return None

    # 3. Build graph
    data = build_graph_from_docs(docs)

    # 4. Run inference
    with torch.no_grad():
        #probs = model(data)[:, 1].tolist()
        #preds = model(data).argmax(dim=1).tolist()

        logits = model(data)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)


    # 5. Match user_id
    for idx, doc in enumerate(docs):
        if str(doc.get("user_id")) == str(user_id):
            fraud_label = int(preds[idx])
            fraud_prob = float(probs[idx])

            # Update DB
            kyc_col.update_one(
                {"user_id": user_id},
                {"$set": {
                    "fraud_label": fraud_label,
                    "fraud_probability": fraud_prob
                }}
            )
            return {"fraud_label": fraud_label, "fraud_probability": fraud_prob}

    return None

def send_status_email(user_email, status):
    subject = f"Your KYC Status: {status}"
    if status == "Approved":
        body = "Congratulations! Your KYC request has been approved."
    elif status == "Rejected":
        body = "We are sorry. Your KYC request has been rejected."
    else:
        body = f"Your KYC status has been updated to {status}."

    message = f"Subject: {subject}\n\n{body}"

    sender_email = "shubham.19jdai037@gmail.com"
    password = os.getenv("GMAIL_PASSWORD")         # Gmail app password
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, user_email, message)
    except Exception as e:
        print("Error sending status email:", e)
        return False
    return True




#  helper function to send email
def send_email(receiver_email, otp):
    sender_email = "shubham.19jdai037@gmail.com"   # <-- change to yours
    password = os.getenv("GMAIL_PASSWORD")  #  load from env          # Gmail app password
    message = f"Subject: Login OTP\n\nYour OTP is {otp}. It is valid for 2 minutes."

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        print("Error sending email:", e)
        return False

    return True

    #with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
     #   server.login(sender_email, password)
      #  server.sendmail(sender_email, receiver_email, message)



def serialize_doc(doc):
    """Convert MongoDB ObjectId to string for JSON response"""
    if not doc:
        return doc
    doc = dict(doc)  # make a copy so we don’t mutate the original
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# helper funtion for name matching
def match_label(score):
    if score >= 85:
        return "Match"
    else:
        return "Not Match"


# --- Training API ---
@app.route("/train", methods=["POST"])
@admin_required
def train_model():
    try:
        # 1. Fetch data from DB
        docs = list(kyc_col.find({}))
        if not docs:
            return jsonify({"status": "error", "message": "No data found"}), 400

        # 2. Build graph
        data = build_graph_from_docs(docs)

        # 3. Train model
        model = FraudGNN(in_channels=data.num_node_features, hidden_channels=16, out_channels=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(200):
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()

        # 4. Predictions
        #with torch.no_grad():
        #    probs = model(data)[:, 1].tolist()  # probability of fraud
        #    preds = model(data).argmax(dim=1).tolist()
        #    print(preds)
        #    print(probs)

        # 4. Predictions
        with torch.no_grad():
            logits = model(data)
            probs = F.softmax(logits, dim=1)[:, 1].tolist()  # safe 0-1 probability
            preds = logits.argmax(dim=1).tolist()


        # 5. Update DB
        for doc, pred, prob in zip(docs, preds, probs):
            kyc_col.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "fraud_label": int(pred),
                    "fraud_probability": float(prob)
                }}
            )

        # 6. Save model + meta
        torch.save(model.state_dict(), "fraud_gnn_model.pt")
        joblib.dump(data.num_node_features, "model_meta.pkl")

        return jsonify({"status": "success", "message": "Model trained & DB updated"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500







@app.route("/api/admin/stats", methods=["GET"])
@admin_required
def admin_stats():
    # 1. Total KYC submissions
    total = kyc_col.count_documents({})

    # 2. Status counts for pie chart
    status_counts = {
        "Pending": kyc_col.count_documents({"status": "Pending"}),
        "Approved": kyc_col.count_documents({"status": "Approved"}),
        "Rejected": kyc_col.count_documents({"status": "Rejected"})
    }

    # 3. Risk level counts for bar chart
    risk_levels_raw = list(kyc_col.aggregate([
        {"$group": {"_id": "$risk_level", "count": {"$sum": 1}}}
    ]))
    # Convert to dict and ensure all three levels exist
    risk_levels = {"Low": 0, "Medium": 0, "High": 0}
    for item in risk_levels_raw:
        if item["_id"] in risk_levels:
            risk_levels[item["_id"]] = item["count"]

    return jsonify({
        "total": total,
        "status_counts": status_counts,  # pie chart data
        "risk_levels": risk_levels       # bar chart data
    })


@app.route("/api/admin/users", methods=["GET"])
@admin_required
def admin_users():
    # Admin token check
    #auth_header = request.headers.get("Authorization")
    #if not auth_header or not auth_header.startswith("Bearer "):
    #    return jsonify({"error": "Missing or invalid token"}), 401

    #token = auth_header.split(" ")[1]
    #user_id, role = decode_token(token)

    #if role != "admin":
    #    return jsonify({"error": "Unauthorized"}), 403

    users = list(kyc_col.find({}, {
    "_id": 0,
    "user_id": 1,
    "user_entered_name": 1,
    "status": 1,
    "risk_level": 1,
    "overall_score": 1,
    "fraud_probability": 1,
    "fraud_label": 1
    }))


    return jsonify(users)


def serialize_audit(doc):
    print("date: ",doc.get("date")," time: ", doc.get("time"))
    return {
        #"id": str(doc.get("_id")),  # convert ObjectId to string
        #"user_id": doc.get("user_id"),
        "username":doc.get("username"),
        "doc_type": doc.get("doc_type"),
        "date": doc.get("date"),
        "time": doc.get("time"),
        #"timestamp": doc.get("timestamp").isoformat() if isinstance(doc.get("timestamp"), datetime) else str(doc.get("timestamp")),
        "status": doc.get("status"),
        "fraud_score": doc.get("fraud_score")
        #"risk_level": doc.get("risk_level")
    }

@app.route("/api/admin/user/<user_id>", methods=["GET"])
@admin_required
def admin_user_details(user_id):
    # Admin token check
    #auth_header = request.headers.get("Authorization")
    #if not auth_header or not auth_header.startswith("Bearer "):
    #    return jsonify({"error": "Missing or invalid token"}), 401

    #token = auth_header.split(" ")[1]
    #_, role = decode_token(token)

    #if role != "admin":
    #    return jsonify({"error": "Unauthorized"}), 403

    user = kyc_col.find_one({"user_id": user_id}, {"_id": 0})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Fetch Aadhaar Audit Trails
    #aadhaar_audit_trails = list(aadhaar_audit.find({"user_id": user_id}, {"_id": 0}))
    #print(aadhaar_audit_trails)
    # Fetch PAN Audit Trails
    #pan_audit_trails = list(pan_audit.find({"user_id": user_id}, {"_id": 0}))
    #print(pan_audit_trails)

    aadhaar_audit_trails = [serialize_audit(d) for d in aadhaar_audit.find({"user_id": user_id})]
    pan_audit_trails = [serialize_audit(d) for d in pan_audit.find({"user_id": user_id})]

    # Include audit trails in response
    user['aadhaar_audit_trails'] = aadhaar_audit_trails
    user['pan_audit_trails'] = pan_audit_trails


    return jsonify(serialize_doc(user))


#@app.route("/api/admin/update_status", methods=["POST"])
#@admin_required
#def update_status():
#    data = request.get_json()
#    user_id = data.get("user_id")
#    status = data.get("status")
#    if status not in ["Pending", "Approved", "Rejected"]:
#        return jsonify({"error": "Invalid status"}), 400

#    result = kyc_col.update_one({"user_id": user_id}, {"$set": {"status": status}})
#    if result.matched_count == 0:
#        return jsonify({"error": "User not found"}), 404
#    return jsonify({"message": f"Status updated to {status}"})


#@app.route("/api/admin/update_status", methods=["POST"])
#@admin_required
#def update_status():
#    data = request.get_json()
#    user_id = data.get("user_id")
#    status = data.get("status")

#    if status not in ["Pending", "Approved", "Rejected"]:
#        return jsonify({"error": "Invalid status"}), 400

    # Update KYC status
#    result = kyc_col.update_one({"user_id": user_id}, {"$set": {"status": status}})
#    if result.matched_count == 0:
#        return jsonify({"error": "User not found"}), 404


    # Get user email
    #user = users_collection.find_one({"user_id": user_id})
#    user = users_collection.find_one({"_id": ObjectId(user_id)})
    # user = users_collection.find_one({"_id":user_id})
#    if user and "email" in user:
#        send_status_email(user["email"], status)

#    return jsonify({"message": f"Status updated to {status}"})


@app.route("/api/admin/update_status", methods=["POST"])
@admin_required
def update_status():
    data = request.get_json()
    user_id = data.get("user_id")
    status = data.get("status")

    if status not in ["Pending", "Approved", "Rejected"]:
        return jsonify({"error": "Invalid status"}), 400

    # Update KYC status (main table)
    result = kyc_col.update_one({"user_id": user_id}, {"$set": {"status": status}})
    if result.matched_count == 0:
        return jsonify({"error": "User not found"}), 404

    # Update latest Aadhaar audit record
    latest_audit = aadhaar_audit.find_one(
        {"user_id": user_id}, sort=[("_id", -1)]
    )
    if latest_audit:
        aadhaar_audit.update_one(
            {"_id": latest_audit["_id"]},
            {"$set": {"status": status}}
        )

    # Update latest pan audit record
    latest_audit = pan_audit.find_one(
        {"user_id": user_id}, sort=[("_id", -1)]
    )
    if latest_audit:
        pan_audit.update_one(
            {"_id": latest_audit["_id"]},
            {"$set": {"status": status}}
        )
    # Send email to user
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user and "email" in user:
        send_status_email(user["email"], status)

    return jsonify({"message": f"Status updated to {status}"})





# =============== Aadhaar APIs ===============
@app.route('/api/extract', methods=['POST'])
def extract_aadhaar():


     #  Extract JWT from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid token"}), 401

    token = auth_header.split(" ")[1]

    user_id, role, username = decode_token(token)

    if not user_id:
        return jsonify({"error": "Token expired or invalid"}), 401

################################################################
    #file = request.files.get('file')
    #if not file:
    #    return jsonify({"error": "No file uploaded"}), 400
    ##file.save(filepath)


    #doc_type = 'aadhaar'# or 'pan'
    #filename = secure_filename(f"{user_id}_{doc_type}.jpg")
    #filepath = os.path.join(UPLOAD_FOLDER,filename)
    #file.save(filepath)
###############################################################



    #from werkzeug.utils import secure_filename

    #import os

    ALLOWED_IMAGE_EXT = {'jpg', 'jpeg', 'png'}
    ALLOWED_PDF_EXT = {'pdf'}

    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    doc_type = 'aadhaar'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if ext in ALLOWED_IMAGE_EXT:
        # Save image directly
        filename = secure_filename(f"{user_id}_{doc_type}.{ext}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process image
        uploaded_emb = get_embedding(filepath)

    elif ext in ALLOWED_PDF_EXT:
        # Convert PDF in memory to JPG
        pdf_bytes = file.read()  # Read PDF bytes from upload
        try:
            pages = convert_from_bytes(
                pdf_bytes,
                dpi=300,
                first_page=1,
                last_page=1,
                poppler_path=r"C:\poppler\poppler-25.07.0\Library\bin"
            )
        except Exception as e:
            return jsonify({"error": f"Failed to convert PDF: {e}"}), 400

        first_page = pages[0]
        filepath = os.path.join(UPLOAD_FOLDER, f"{user_id}_{doc_type}.jpg")
        first_page.save(filepath, 'JPEG')

        # Process converted image
        uploaded_emb = get_embedding(filepath)

    else:
        return jsonify({"error": "Unsupported file type"}), 400

    #uploaded_emb = get_embedding(filepath)
    sim = cosine_similarity(AADHAAR_TEMPLATE, uploaded_emb)[0][0]
    sim  = float(sim)
    print("Similarity:", sim)





    text = extract_rawtxt(filepath)
    data = extract_details(text)

    data["layout"] = sim

    # Save extracted data in DB
    kyc_col.update_one(
        {"user_id": user_id},
        {"$set": {
            "aadhaar": data,
            "aadhaar_file_path": filepath,
            #"aadhaarFormat" : sim,
            "status": "Pending"
        }},
        upsert=True
    )
    return jsonify(data)




import uuid


@app.route("/save_aadhaar", methods=["POST"])
def save_aadhaar():

    #  Extract JWT from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid token"}), 401

    token = auth_header.split(" ")[1]

    user_id, role, username = decode_token(token)

    if not user_id:
        return jsonify({"error": "Token expired or invalid"}), 401


    #payload = decode_token(token)  # your helper
    #if not payload:
    #    return jsonify({"error": "Token expired or invalid"}), 401

    #user_id = payload["user_id"]   # take user_id from token
    #role = payload.get("role", "user")

    data = request.get_json()

    # Generate unique user_id if not present
    #user_id = data.get("user_id") or str(uuid.uuid4())

    aadhaar_num = data.get("AadhaarNumber")

    aadhaar_num = "".join(filter(str.isdigit, aadhaar_num))


    # 1. Validate Aadhaar
    is_valid = verhoeff_check(aadhaar_num)

    # 2. Hash & Duplicate Check
    aadhaar_hash = hash_value(aadhaar_num)
    duplicate = kyc_col.find_one({"aadhaar.number_hash": aadhaar_hash})
    #print("duplicate value:  ",duplicate)


    r = kyc_col.find_one(
        {"user_id": user_id},
        {"aadhaar_file_path": 1, "_id": 0}   # only return this field
    )

    if not r or "aadhaar_file_path" not in r:
        return jsonify({"error ! aadhaar filepath not found"}) , 404

    filepath = r["aadhaar_file_path"]

     # Run file quality/issue check
    valid, issues = check_document(filepath)
    # 3. Manipulation issues (optional: can be passed from extract API)
    #issues = data.get("issues", [])
    ok = len(issues) == 0

    username_match_aadhaar = fuzz.token_set_ratio(username, data.get("Name",""))
    username_match_aadhaar = match_label(username_match_aadhaar)

    WEIGHTS = {
    "invalid": 25,
    "duplicate": 25,      # assign appropriate score
    "manipulation": 25,
    "mismatch":25}


    # 4. Fraud Score with Weights
    fraud_score = 0
    if not is_valid:
        fraud_score += WEIGHTS["invalid"]
    if duplicate:
        fraud_score += WEIGHTS["duplicate"]
    if not ok:
        fraud_score += WEIGHTS["manipulation"]
    if username_match_aadhaar == "Not Match":
        fraud_score += WEIGHTS["mismatch"]

    if fraud_score > 100:
        fraud_score = 100

    layout = "valid" if (data.get("layout",0)) > 0.75 else "Not_Valid"
    # 5. Build Aadhaar object
    aadhaar_obj = {
        "aadhaar_name": data.get("Name", ""),
        "dob": data.get("DOB", ""),
        "gender": data.get("Gender", ""),
        "address": data.get("Address", ""),
        "number_hash": aadhaar_hash,
        "aadhaar_number":"XXXX XXXX " + aadhaar_num[-4:],
        "valid": "Yes" if is_valid else "No" ,
        "duplicate_found": bool(duplicate),
        "fraud_score": fraud_score,
        "aadhar_manipulation": issues,
        "AadhaarName_username":username_match_aadhaar,
        "Aadhaar_Layout": layout
    }

    # 6. Record to save
    record = {
        "user_id": user_id,
        "user_entered_name": username, #data.get("Name", ""),
        "aadhaar": aadhaar_obj,
        #"pan": None,
    }

    # 7. Recompute fraud score
    #score, risk, all_issues = compute_fraud_score(aadhaar_obj, None)
    #record["overall_score"] = score
    #record["risk_level"] = risk

    record["status"] = "Pending"

    # Define IST timezone (UTC+5:30)
    #IST = timezone(timedelta(hours=5, minutes=30))

    kyc_col.update_one({"user_id": user_id}, {"$set": record}, upsert=True)
    #from datetime  import datetime

    IST = timezone("Asia/Kolkata")
    now = datetime.now(IST)

    audit_doc = {
    "user_id": user_id,
    "username" : username,
    "doc_type": "aadhaar",
    "date": now.strftime("%d-%b-%Y"),   # e.g. "20-Sep-2025"
    "time": now.strftime("%H:%M:%S"),   # e.g. "15:45:12"
    #"timestamp": datetime.now(IST).strftime("%d-%b-%Y %H:%M:%S"),
    "status": record.get("status", "Pending"),
    "fraud_score": fraud_score
    #"risk_level": record.get("risk_level", "Unknown"),
    #"fraud_probability": record.get("fraud_probability", 0),
    #"fraud_label": record.get("fraud_label", 0)
}


    aadhaar_audit.insert_one(audit_doc)


    # Generate JWT
    #token = generate_token(user_id)

    return jsonify({
        "message": "Aadhaar saved",
        #"data": record
    })




#@app.route('/api/extract_pan', methods=['POST'])
#def extract_pan():
#    file = request.files['file']
#    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#    file.save(filepath)

    #from ocr.ocr_ex import extract_pantxt
    #from ocr.f import pan_details

#    text = extract_pantxt(filepath)
#    data = pan_details(text)
#    return jsonify(data)

@app.route('/api/extract_pan', methods=['POST'])
def extract_pan():
    # 1. Extract JWT from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid token"}), 401

    token = auth_header.split(" ")[1]
    user_id, role, username = decode_token(token)

    if not user_id:
        return jsonify({"error": "Token expired or invalid"}), 401

    # 2. Get uploaded file
    #file = request.files.get('file')
    #if not file:
    #    return jsonify({"error": "No file uploaded"}), 400

    # 3. Save file with unique name
    #doc_type = 'pan'
    #filename = secure_filename(f"{user_id}_{doc_type}.jpg")
    #filepath = os.path.join(UPLOAD_FOLDER, filename)
    #file.save(filepath)


    #uploaded_emb = get_embedding(filepath)

    ALLOWED_IMAGE_EXT = {'jpg', 'jpeg', 'png'}
    ALLOWED_PDF_EXT = {'pdf'}

    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    doc_type = 'pan'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    if ext in ALLOWED_IMAGE_EXT:
        # Save image directly
        filename = secure_filename(f"{user_id}_{doc_type}.{ext}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process image
        uploaded_emb = get_embedding(filepath)

    elif ext in ALLOWED_PDF_EXT:
        # Convert PDF in memory to JPG
        pdf_bytes = file.read()  # Read PDF bytes from upload
        try:
            pages = convert_from_bytes(
                pdf_bytes,
                dpi=300,
                first_page=1,
                last_page=1,
                poppler_path=r"C:\poppler\poppler-25.07.0\Library\bin"
            )
        except Exception as e:
            return jsonify({"error": f"Failed to convert PDF: {e}"}), 400

        first_page = pages[0]
        filepath = os.path.join(UPLOAD_FOLDER, f"{user_id}_{doc_type}.jpg")
        first_page.save(filepath, 'JPEG')

        # Process converted image
        uploaded_emb = get_embedding(filepath)

    else:
        return jsonify({"error": "Unsupported file type"}), 400


    sim = cosine_similarity(PAN_TEMPLATE, uploaded_emb)[0][0]
    print("Similarity:", sim)



    # 4. Extract details using OCR
    text = extract_pantxt(filepath)       # your OCR function
    data = pan_details(text)              # your parser

    data["layout"] = float(sim)
    # 5. Save extracted data in DB
    kyc_col.update_one(
        {"user_id": user_id},
        {"$set": {
            "pan": data,
            "pan_file_path": filepath,
            #"pan_manipulation": issues,
            "status": "Pending"
        }},
        upsert=True
    )

    # 6. Return response
    return jsonify(data)




@app.route("/save_pan", methods=["POST"])
def save_pan():
    # Extract JWT from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid token"}), 401

    token = auth_header.split(" ")[1]
    user_id, role, username = decode_token(token)
    #user_id = decode_token(token)
    if not user_id:
        return jsonify({"error": "Token expired or invalid"}), 401

    data = request.get_json()
    pan_num = data.get("PANNumber")

     # 1. Validate PAN
    is_valid = validate_pan(pan_num)

    # 2. Hash & Duplicate Check
    pan_hash = hash_value(pan_num)
    duplicate = kyc_col.find_one({"pan.number_hash": pan_hash})

    r = kyc_col.find_one(
        {"user_id": user_id},
        {"pan_file_path": 1, "_id": 0}   # only return this field
    )

    if not r or "pan_file_path" not in r:
        return jsonify({"error": "pancard File path not found"}), 404

    filepath = r["pan_file_path"]

    valid, issues = check_document(filepath)
    # 3. Manipulation issues
    #issues = data.get("issues", [])
    ok = len(issues) == 0

    username_match_pan = fuzz.token_set_ratio(username, data.get("Name",""))
    username_match_pan = match_label(username_match_pan)

    WEIGHTS = {
    "invalid": 25,
    "duplicate": 25,      # assign appropriate score
    "manipulation": 25,
    "mismatch":25}

    #def match_label(score):
    #if score >= 85:
    #    return "Match"
    #else:
    #    return "Not Match"

    # 4. Fraud Score with Weights
    fraud_score = 0
    if not is_valid:
        fraud_score += WEIGHTS["invalid"]
    if duplicate:
        fraud_score += WEIGHTS["duplicate"]
    if not ok:
        fraud_score += WEIGHTS["manipulation"]
    if username_match_pan == "Not Match":
        fraud_score += WEIGHTS["mismatch"]

    if fraud_score > 100:
        fraud_score = 100

    layout = "valid" if (data.get("layout",0)) > 0.75 else "Not_Valid"

    # 5. PAN object
    pan_obj = {
        "pan_name": data.get("Name", ""),
        "father_name": data.get("FatherName", ""),
        "number_hash": pan_hash,
        "pan_number" : "xxxxxxxx"+ pan_num[-2:],
        "valid": is_valid,
        "duplicate_found": bool(duplicate),
        "fraud_score": fraud_score,
        "pan_manipulation": issues,
        "panName_username": username_match_pan,
        "pan_layout": layout
    }



    existing = kyc_col.find_one({"user_id": user_id})
    if not existing:
        return jsonify({"error": "No Aadhaar record found"}), 400

    existing["pan"] = pan_obj

    # 7. Fuzzy Name Matching
    aadhaar_name = existing.get("aadhaar", {}).get("aadhaar_name", "")
    pan_name = pan_obj.get("pan_name", "")

    #name_on_aadhaar_pan = fuzz.token_set_ratio(aadhaar_name,data.get("Name",""))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([aadhaar_name, pan_name], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])


    existing["AadhaarName_panName"] = "Match" if similarity >= 0.75 else "Not Match"#match_label(name_on_aadhaar_pan)

    score, risk = compute_fraud_score(existing.get("aadhaar"), pan_obj)
    existing["overall_score"] = score
    existing["risk_level"] = risk
    existing["status"] = "Pending"
    existing["fraud_probability"] = 0
    existing["fraud_label"] = 0

    # Define IST timezone (UTC+5:30)
    #IST = timezone(timedelta(hours=5, minutes=30))

    kyc_col.update_one({"user_id": user_id}, {"$set": existing}, upsert=True)



    IST = timezone("Asia/Kolkata")
    now = datetime.now(IST)

    audit_doc = {
    "user_id": user_id,
    "username":username,
    "doc_type": "pan",
    "date": now.strftime("%d-%b-%Y"),   # e.g. "20-Sep-2025"
    "time": now.strftime("%H:%M:%S"),   # e.g. "15:45:12"
    #"timestamp": datetime.now(IST).strftime("%d-%b-%Y %H:%M:%S"),
    "status": existing.get("status", "Pending"),
    "fraud_score": existing.get("overall_score", 0),
    #"risk_level": existing.get("risk_level", "Unknown"),
    #"fraud_probability": existing.get("fraud_probability", 0),
    #"fraud_label": existing.get("fraud_label", 0)
    }

    pan_audit.insert_one(audit_doc)


    ## call INference gnn api
    result = run_gnn_inference_for_user(user_id)

    if result:
        return jsonify({
            #"status": "successfully uploaded",
            "message": "PAN saved"# and GNN prediction updated",
            #"fraud_label": result["fraud_label"],
            #"fraud_probability": result["fraud_probability"]
        })
    else:
        return jsonify({"status": "error", "message": "Prediction failed"})

    #return jsonify({"message": "PAN saved", "data": existing})
    # Serialize before returning
    #return jsonify({"message": "PAN saved"}) #"data": serialize_doc(existing)})




@app.route("/api/user/result", methods=["GET"])
def get_user_result():
    # 1. Extract JWT from header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid token"}), 401

    token = auth_header.split(" ")[1]
    user_id, role, username = decode_token(token)

    if not user_id:
        return jsonify({"error": "Token expired or invalid"}), 401

    # 2. Fetch KYC record from DB
    record = kyc_col.find_one({"user_id": user_id})
    if not record:
        return jsonify({"message": "No KYC record found"}), 404

    # Selectively pick fields to show user
    #user_result = {
    #    "Name": record.get("user_entered_name", username),
    #    "DOB": record.get("aadhaar", {}).get("dob", ""),
    #    "Gender": record.get("aadhaar", {}).get("gender", ""),
    #    #"PAN": record.get("pan", {}).get("pan_number", ""),
    #    "Status": record.get("status", "Pending")
    #}

    # Aadhaar/PAN image filenames (extract only basename)
    import os
    aadhaar_filename = os.path.basename(record.get("aadhaar_file_path", "")) if record.get("aadhaar_file_path") else ""
    pan_filename = os.path.basename(record.get("pan_file_path", "")) if record.get("pan_file_path") else ""

    user_result = {
        "user_entered_name": record.get("user_entered_name", username),
        "status": record.get("status", "Pending"),

        # Aadhaar
        "aadhaar_name": record.get("aadhaar", {}).get("aadhaar_name", ""),
        "aadhaar_dob": record.get("aadhaar", {}).get("dob", ""),
        "aadhaar_gender": record.get("aadhaar", {}).get("gender", ""),
        "aadhaar_address": record.get("aadhaar", {}).get("address", ""),
        "aadhaar_number": record.get("aadhaar", {}).get("aadhaar_number", ""),
        "aadhaar_image": f"/uploads/{aadhaar_filename}" if aadhaar_filename else "",

        # PAN
        "pan_name": record.get("pan", {}).get("pan_name", ""),
        "pan_father_name": record.get("pan", {}).get("father_name", ""),
        "pan_number": record.get("pan", {}).get("pan_number", ""),
        "pan_image": f"/uploads/{pan_filename}" if pan_filename else ""
    }

    return jsonify({"message": "KYC result fetched", "data": user_result})

    # 3. Serialize ObjectId before returning
    #return jsonify({"message": "KYC result fetched", "data": serialize_doc(record)})


@app.route('/api/record', methods=['GET'])
def fetch_records():
    docs = list(aadhaar_collection.find({}, {"_id": 0}))
    return jsonify(docs)

# =============== SIGNUP APIs ===============

#@app.route('/api/signup', methods=['POST'])
#def signup():
#    data = request.get_json()
#    firstname = data.get("firstname")
#    lastname = data.get("lastname")
#    email = data.get("email")
#    password = data.get("password")

    # Check if user with the email already exists
#    if users_collection.find_one({"email": email}):
#        return jsonify({"message": "Email already exists"}), 400

#    otp = str(random.randint(100000, 999999))
#    otp_doc = {
#        "email": email,
#        "firstname": firstname,
#        "lastname": lastname,
#        "password": generate_password_hash(password),
#        "otp": otp,
#        "time": time.time()
#    }
#    otp_collection.update_one({"email": email}, {"$set": otp_doc}, upsert=True)

#    send_email(email, otp)
#    return jsonify({"message": "OTP sent to email. Please verify."})

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    firstname = data.get("firstname")
    lastname = data.get("lastname")
    email = data.get("email")
    password = data.get("password")
    confirm_password = data.get("confirm_password")

    # Check password match
    if password != confirm_password:
        return jsonify({"message": "Passwords do not match"}), 400

    # Check if user with the email already exists
    if users_collection.find_one({"email": email}):
        return jsonify({"message": "Email already exists"}), 400

    otp = str(random.randint(100000, 999999))
    otp_doc = {
        "email": email,
        "firstname": firstname,
        "lastname": lastname,
        "password": generate_password_hash(password),
        "otp": otp,
        "time": time.time()
    }
    otp_collection.update_one({"email": email}, {"$set": otp_doc}, upsert=True)

    send_email(email, otp)
    return jsonify({"message": "OTP sent to email. Please verify."})



@app.route('/api/verify-otp', methods=['POST'])
def verify_signup_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")

    record = otp_collection.find_one({"email": email})
    if not record:
        return jsonify({"message": "No OTP found"}), 400

    # Check OTP and expiry (2 minutes)
    if record["otp"] == otp and time.time() - record["time"] <= 120:
        users_collection.insert_one({
            "firstname": record["firstname"],
            "lastname": record["lastname"],
            "email": email,
            "password": record["password"],
            "role":"user"
        })
        otp_collection.delete_one({"email": email})
        return jsonify({"message": "Signup successful!"})

    return jsonify({"message": "Invalid or expired OTP"}), 400


@app.route('/api/resend-otp', methods=['POST'])
def resend_signup_otp():
    data = request.get_json()
    email = data.get("email")

    record = otp_collection.find_one({"email": email})
    if not record:
        return jsonify({"message": "No signup found"}), 400

    otp = str(random.randint(100000, 999999))
    otp_collection.update_one({"email": email}, {"$set": {"otp": otp, "time": time.time()}})
    send_email(email, otp)
    return jsonify({"message": "OTP resent successfully!"})

# =============== LOGIN APIs ===============

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    # Find user by email
    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid credentials"}), 401

    # Generate OTP and save to OTP collection
    otp = str(random.randint(100000, 999999))
    otp_collection.update_one(
        {"email": user["email"]},
        {"$set": {"otp": otp, "time": time.time()}},
        upsert=True
    )
    send_email(user["email"], otp)
    return jsonify({"message": "OTP sent to email. Please verify."})


@app.route('/api/login-verify-otp', methods=['POST'])
def verify_login_otp():
    data = request.get_json()
    email = data.get("email")
    otp = data.get("otp")

    record = otp_collection.find_one({"email": email})
    user = users_collection.find_one({"email": email})

    username = user['firstname']  + ' ' + user['lastname']

    if record and record["otp"] == otp and time.time() - record["time"] <= 120:
        otp_collection.delete_one({"email": email})

        token = generate_token(str(user["_id"]), user["role"], username)

        return jsonify({
            "message": "Login successful!",
            "token": token,
            "role": user["role"] # "username": user['firstname'] + ' ' + user['lastname']
        })
        #return jsonify({"message": "Login successful!"})
    return jsonify({"message": "Invalid or expired OTP"}), 400

@app.route('/api/resend-login-otp', methods=['POST'])
def resend_login_otp():
    data = request.get_json()
    email = data.get("email")

    record = otp_collection.find_one({"email": email})
    if not record:
        return jsonify({"message": "No login session found"}), 400

    otp = str(random.randint(100000, 999999))
    otp_collection.update_one({"email": email}, {"$set": {"otp": otp, "time": time.time()}})
    send_email(email, otp)
    return jsonify({"message": "Login OTP resent successfully!"})

# ---------- Run Flask ----------
if __name__ == '__main__':
    app.run(debug=True)
