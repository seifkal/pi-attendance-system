import mysql.connector
from mysql.connector import Error
import os
import json
import re
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles  # <--- NEW IMPORT

import uvicorn

app = FastAPI()

# --- CONFIGURATION ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'zain1909',
    'database': 'attendance_db'
}

# --- Setup Folders ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DB_FOLDER = os.path.join(BASE_DIR, "databases")
STATIC_DIR = os.path.join(BASE_DIR, "static") # <--- NEW PATH

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- MOUNT STATIC FILES (CRITICAL STEP) ---
# This allows the HTML to access /static/css/style.css
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Helper: Get Database Connection ---
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# --- API Endpoints ---

@app.post("/api/upload_report")
async def upload_report(request: Request):
    try:
        data = await request.json()
        
        # Save JSON Backup
        safe_time = re.sub(r'[^a-zA-Z0-9]', '_', data['date'])
        filename = f"Report_{safe_time}.json"
        file_path = os.path.join(DB_FOLDER, filename)
        os.makedirs(DB_FOLDER, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        # Save to MySQL
        conn = get_db_connection()
        if not conn: return JSONResponse(status_code=500, content={"message": "DB Connection Failed"})
        
        cursor = conn.cursor()
        
        cursor.execute("INSERT INTO sessions (name, date_str, duration) VALUES (%s, %s, %s)", 
                       (data['session_name'], data['date'], data['duration']))
        session_id = cursor.lastrowid
        
        student_values = []
        for s in data['students']:
            student_values.append((session_id, s['name'], s['status'], s['first_seen'], s['checks'], s['attention_score']))
            
        cursor.executemany("INSERT INTO reports (session_id, student_name, status, first_seen, checks, attention_pct) VALUES (%s, %s, %s, %s, %s, %s)", student_values)
        
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "Success", "id": session_id}
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: int):
    try:
        conn = get_db_connection()
        if not conn: return JSONResponse(status_code=500, content={"message": "DB Connection Failed"})
        cursor = conn.cursor()
        
        # Force delete reports first, then session
        cursor.execute("DELETE FROM reports WHERE session_id = %s", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "Deleted", "id": session_id}
    except Exception as e:
        print(f"Error deleting: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/sessions")
async def get_sessions():
    conn = get_db_connection()
    if not conn: return []
    cursor = conn.cursor(dictionary=True) 
    cursor.execute("SELECT * FROM sessions ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

@app.get("/api/session/{session_id}")
async def get_session_details(session_id: int):
    conn = get_db_connection()
    if not conn: return []
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM reports WHERE session_id = %s", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    formatted = []
    for r in rows:
        formatted.append({
            "name": r['student_name'],
            "status": r['status'],
            "first_seen": r['first_seen'],
            "checks": r['checks'],
            "attention_pct": r['attention_pct'],
            "low_attention": r['attention_pct'] < 50
        })
    return formatted

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)