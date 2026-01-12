import requests
import json
import random
import time

# Ensure this matches where your server.py is running
SERVER_URL = "http://127.0.0.1:8000/api/upload_report"

STUDENTS = ["Fairuz", "lionel messi", "Tom Cruise","Albert Einstein" , "Musa Al-Taamari", "Leonardo da Vinci","The Rock" , "Amr Diab"]

def generate_and_send(session_name, date_str, base_attention):
    print(f"📤 Sending: {session_name}...")
    
    report = {
        "session_name": session_name,
        "date": date_str,
        "duration": 90,
        "students": []
    }
    
    for name in STUDENTS:
        # 20% Chance the student is ABSENT
        if random.random() < 0.20:
            student_data = {
                "name": name,
                "status": "Absent",
                "first_seen": "--:--",
                "checks": 0,
                "attention_score": 0
            }
        else:
            # Student is PRESENT
            # Create random realistic data around the base average
            score = max(0, min(100, random.uniform(base_attention - 15, base_attention + 15)))
            
            student_data = {
                "name": name,
                "status": "Present",
                "first_seen": f"09:{random.randint(10, 59)}",
                "checks": random.randint(5, 12),
                "attention_score": round(score, 1)
            }

        report["students"].append(student_data)

    try:
        res = requests.post(SERVER_URL, json=report)
        if res.status_code == 200:
            print("   ✅ Success! Data saved to MySQL.")
        else:
            print(f"   ❌ Failed: {res.status_code} - {res.text}")
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
        print("      (Is server.py running?)")

if __name__ == "__main__":
    print("--- Simulating Class Reports ---")
    
    # Send 3 different classes with varying difficulty/attention levels
    generate_and_send("Computer Architecture", "2026-01-12 09:30", 88.0)
    time.sleep(1)
    generate_and_send("Linear Algebra", "2026-01-12 13:00", 45.0)
    time.sleep(1)
    generate_and_send("Embedded Systems", "2026-01-12 15:30", 72.5)
    
    print("\n-----------------------------------")
    print("Done! Check your dashboard at: http://localhost:8000")