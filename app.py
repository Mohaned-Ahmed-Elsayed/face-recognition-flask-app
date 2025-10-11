import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN spam logs
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
import cv2
import threading
import uuid
import webbrowser
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from deepface import DeepFace
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import csv
from datetime import datetime

# -------- Flask setup --------
app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_DIR = "unknown_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

camera = cv2.VideoCapture(0)

# -------- Email settings --------
FROM_EMAIL = os.getenv("FROM_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")

# -------- Load face model once --------
FACE_MODEL = DeepFace.build_model("Facenet512")
recognition_running = True

DB = {}

# -------- Normalize embeddings --------
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# -------- Face augmentation --------
def augment_face(image_path, save_dir, person_name):
    img = cv2.imread(image_path)
    if img is None:
        return
    img = cv2.resize(img, (160, 160))
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"{person_name}_orig.jpg"), img)

    # brightness
    for i, alpha in enumerate([0.7, 1.3]):
        bright = cv2.convertScaleAbs(img, alpha=alpha, beta=30)
        cv2.imwrite(os.path.join(save_dir, f"{person_name}_bright{i}.jpg"), bright)

    # rotation
    for i, angle in enumerate([-15, 15]):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        cv2.imwrite(os.path.join(save_dir, f"{person_name}_rot{i}.jpg"), rotated)

    # flip
    flipped = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(save_dir, f"{person_name}_flip.jpg"), flipped)

# -------- DB builder with embeddings --------
def build_db():
    global DB
    DB = {}
    for person in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        DB[person] = []
        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)

            if not os.path.isfile(img_path) or os.path.getsize(img_path) == 0:
                print(f"⚠️ Skipping invalid file: {img_file}")
                continue

            try:
                rep = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet512",
                    enforce_detection=False,
                )
                if rep and "embedding" in rep[0]:
                    emb = normalize(np.array(rep[0]["embedding"]))
                    DB[person].append(emb)
            except Exception as e:
                print(f"⚠️ Embedding error for {img_file}: {e}")

    print("✅ Database built with", sum(len(v) for v in DB.values()), "faces.")

# -------- Recognition Logging --------
LOG_FILE = "logs.csv"

def log_recognition(name, dist):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, name, f"{dist:.4f}"])

#------------recognition------
def run_recognition_loop():
    global last_detections

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frame_counter = 0
    recognition_interval = 10  # run deep recognition every 10 frames

    while recognition_running:
        success, frame = camera.read()
        if not success:
            continue

        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        new_detections = []

        # run DeepFace recognition only every Nth frame
        if frame_counter % recognition_interval == 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                label = "Unknown"
                color = (0, 0, 255)
                best_dist = 100

                try:
                    rep = DeepFace.represent(face_img, model_name="Facenet512", enforce_detection=False)
                    if rep and "embedding" in rep[0]:
                        embedding = normalize(np.array(rep[0]["embedding"]))
                        best_match = None

                        for person, embeddings in DB.items():
                            for e in embeddings:
                                dist = np.linalg.norm(embedding - e)
                                if dist < best_dist:
                                    best_dist = dist
                                    best_match = person

                        if best_match and best_dist < 0.8:
                            label = best_match
                            color = (0, 255, 0)
                        else:
                            filename = f"{uuid.uuid4().hex}.jpg"
                            temp_path = os.path.join(UNKNOWN_DIR, filename)
                            cv2.imwrite(temp_path, face_img)
                            send_email_with_throttle(temp_path, "http://localhost:5000/")

                        log_recognition(label, best_dist)
                except Exception as e:
                    print("Recognition error:", e)

                new_detections.append((x, y, w, h, label, color))
        else:
            # Only update boxes without re-recognizing (smooth movement)
            for (x, y, w, h) in faces:
                new_detections.append((x, y, w, h, label,color))

        if new_detections:
            last_detections = new_detections

        frame_counter += 1



# -------- Email alerts with throttling --------
last_email_time = 0
def send_email_with_throttle(image_path, add_url, cooldown=30):
    global last_email_time
    now = datetime.now().timestamp()
    if now - last_email_time < cooldown:
        return  # skip if still cooling down
    last_email_time = now
    send_email(image_path, add_url)

def send_email(image_path, add_url):
    msg = MIMEMultipart("related")
    msg["Subject"] = "🚨 Unknown Face Detected"
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    html = f"""
    <html>
      <body style="font-family:Arial,sans-serif;">
        <h2>🚨 Unknown Face Detected</h2>
        <p>An unrecognized person was seen. Review below:</p>
        <img src="cid:face_image" width="300" style="border-radius:10px;margin:10px 0;" />
        <p><b>Check the app to add this person.</b></p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    with open(image_path, "rb") as f:
        img = MIMEImage(f.read(), _subtype="jpeg")
        img.add_header("Content-ID", "<face_image>")
        msg.attach(img)

    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(FROM_EMAIL, APP_PASSWORD)
    server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
    server.quit()

# -------- Video stream with recognition --------
frame_count = 0
def gen_frames():
    global last_detections
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.resize(frame, (320, 240))

        # Draw last known detections (updated by thread)
        for (x, y, w, h, label, color) in last_detections:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# -------- Routes --------
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/logs')
def view_logs():
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="") as f:
            reader = csv.reader(f)
            logs = list(reader)[::-1]  # newest first
    return render_template("logs.html", logs=logs)

# -------- Add person from file --------
@app.route("/add_person/file", methods=["POST"])
def add_person_file():
    global DB
    name = request.form.get("name", "Unknown").strip()
    file = request.files.get("file")

    if not file:
        return jsonify({"status": "error", "msg": "no file uploaded"}), 400

    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(person_dir, filename)
    file.save(save_path)

    augment_face(save_path, person_dir, name)
    build_db()
    return jsonify({"status": "ok", "msg": f"{name} added from file"})


# -------- Run App --------
if __name__ == "__main__":
    build_db()
    last_detections = []
    # ✅ Start recognition in the background
    recognition_thread = threading.Thread(target=run_recognition_loop, daemon=True)
    recognition_thread.start()
    import atexit
    def cleanup():
      global recognition_running, camera
      recognition_running = False
      camera.release()
      print("🧹 Cleaned up camera and stopped recognition thread.")
    atexit.register(cleanup)

    # ✅ Launch Flask app and open it automatically
    url = "http://127.0.0.1:5000/"
    threading.Timer(1, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)








