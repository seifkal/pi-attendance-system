# Raspberry Pi 5 Deployment Guide - Attendance System

## Quick Start

### 1. Prepare Your Raspberry Pi 5

**Hardware needed:**
- Raspberry Pi 5 (4GB or 8GB RAM recommended)
- Pi Camera Module 3 or USB webcam
- MicroSD card (32GB+, Class 10)
- Power supply (5V/5A USB-C)
- Heatsink and fan (recommended)

**Install Raspberry Pi OS:**
- Use Raspberry Pi Imager
- Install "Raspberry Pi OS (64-bit)" - **NOT** the Lite version
- Enable SSH and set up WiFi during imaging

### 2. Transfer Files to Pi

**From your Mac:**
```bash
# Compress the project (exclude large files)
cd /Users/saifqal
tar -czf project.tar.gz \
    --exclude='Project/datasets' \
    --exclude='Project/venv*' \
    --exclude='Project/runs' \
    Project/

# Transfer to Pi (replace 'raspberrypi.local' with your Pi's hostname/IP)
scp project.tar.gz pi@raspberrypi.local:~/

# SSH into Pi
ssh pi@raspberrypi.local

# Extract
cd ~
tar -xzf project.tar.gz
cd Project
```

### 3. Install Dependencies on Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
sudo apt install -y python3-pip python3-opencv python3-picamera2

# Create virtual environment
python3 -m venv venv_pi
source venv_pi/bin/activate

# Install packages
pip install insightface onnxruntime

# Test installation
python3 -c "import insightface; print('✓ InsightFace OK')"
```

### 4. Register Students on Pi

**Option A: Use webcam on Pi**
```bash
source venv_pi/bin/activate

python scripts/register_student.py \
    --name "Student Name" \
    --id "STU001" \
    --webcam \
    --num-samples 7 \
    --db databases/students.db
```

**Option B: Transfer database from Mac**
```bash
# On Mac: Copy your existing database
scp databases/students.db pi@raspberrypi.local:~/Project/databases/
scp databases/embeddings.pkl pi@raspberrypi.local:~/Project/databases/
```

### 5. Run Attendance System

**With display (if Pi connected to monitor):**
```bash
source venv_pi/bin/activate
python scripts/pi_attendance.py --db databases/students.db
```

**Headless mode (no display needed):**
```bash
python scripts/pi_attendance.py --db databases/students.db --no-display
```

**For testing (run for 2 minutes):**
```bash
python scripts/pi_attendance.py --duration 120
```

---

## Performance Optimization

### Frame Processing Rate

**Default:** Process every 3rd frame
- Camera runs at ~30 FPS (smooth display)
- Recognition runs at ~10 FPS (sufficient for attendance)

**Adjust if needed:**
```bash
# Process every 2nd frame (faster recognition, more CPU load)
python scripts/pi_attendance.py --process-every 2

# Process every 5th frame (lower CPU load, slower recognition)
python scripts/pi_attendance.py --process-every 5
```

### Expected Performance

**Raspberry Pi 5 with default settings:**
- Camera FPS: 25-30 FPS
- Processing FPS: 8-12 FPS
- CPU usage: 40-60%
- Temperature: 50-65°C (with heatsink)

**If performance is poor:**
1. Ensure heatsink/fan is installed
2. Increase `--process-every` to 5 or 10
3. Lower camera resolution (edit script, change to 416x416)
4. Close other applications

---

## Camera Setup

### Using Pi Camera Module

**Connect:**
1. Power off Pi
2. Connect ribbon cable to camera port
3. Power on Pi
4. Test: `libcamera-hello`

**Use in script:**
```bash
python scripts/pi_attendance.py  # Automatically uses Pi Camera
```

### Using USB Webcam

**Force USB camera:**
```bash
python scripts/pi_attendance.py --usb-camera
```

---

## Attendance Database

### View Attendance Records

```bash
# On Pi, open SQLite database
sqlite3 databases/students.db

# SQL queries
SELECT * FROM students;
SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 20;
SELECT student_id, COUNT(*) as attendance_count 
FROM attendance 
GROUP BY student_id;

# Exit
.quit
```

### Export Attendance to CSV

```python
# In Python on Pi
from student_database import StudentDatabase

db = StudentDatabase("databases/students.db")
db.export_attendance_csv("attendance_report.csv")
db.close()
```

---

## Auto-Start on Boot (Optional)

**Create systemd service:**

```bash
# Create service file
sudo nano /etc/systemd/system/pi-attendance.service
```

**Add content:**
```ini
[Unit]
Description=Pi Attendance System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Project
ExecStart=/home/pi/Project/venv_pi/bin/python /home/pi/Project/scripts/pi_attendance.py --no-display --db /home/pi/Project/databases/students.db
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable pi-attendance
sudo systemctl start pi-attendance

# Check status
sudo systemctl status pi-attendance

# View logs
sudo journalctl -u pi-attendance -f
```

---

## Troubleshooting

### "No module named insightface"
```bash
source venv_pi/bin/activate
pip install insightface onnxruntime
```

### "Cannot open camera"
```bash
# Test Pi Camera
libcamera-hello

# Test USB camera
v4l2-ctl --list-devices

# Check permissions
sudo usermod -a -G video $USER
```

### Low FPS / High CPU
- Increase `--process-every` to 5 or 10
- Ensure cooling (heatsink + fan)
- Close other applications
- Reduce camera resolution in script

### Database errors
```bash
# Check database exists
ls -la databases/

# Check permissions
chmod 664 databases/students.db
chmod 664 databases/embeddings.pkl
```

---

## Next Steps

Once attendance is working on Pi:
1. Test with multiple students
2. Benchmark performance over extended periods
3. Add attention monitoring (Phase 4)
4. Add phone detection (Phase 4)
5. Create web dashboard for viewing attendance

---

## File Transfer Tips

**Transfer only what you need:**
```bash
# Databases
scp databases/*.db databases/*.pkl pi@raspberrypi.local:~/Project/databases/

# Scripts
scp scripts/*.py pi@raspberrypi.local:~/Project/scripts/

# Models (if trained on Mac)
scp models/best.pt pi@raspberrypi.local:~/Project/models/
```

**Sync changes during development:**
```bash
# Use rsync for efficient updates
rsync -avz --exclude='venv*' --exclude='datasets' \
    /Users/saifqal/Project/ pi@raspberrypi.local:~/Project/
```
