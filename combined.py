"""
main.py
Unified Face Attendance + Cheating Detection System
--------------------------------------------------
- Runs attendance for 10 seconds using webcam
- Then automatically switches to YOLO cheating detection
"""

import time
import logging
import os
from Attendance.attendance_taker import FaceRecognizer
from exam import CheatingDetector


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("🚀 Starting Unified Attendance + Cheating Detection System")

    # -----------------------------
    # STEP 0: Get Subject Input
    # -----------------------------
    subject_name = input("📚 Enter the subject name: ").strip().capitalize()

    # Define subject folder path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    subject_folder = os.path.join(base_dir, "Attendance", "students", subject_name)

    if not os.path.exists(subject_folder):
        logging.error(f"⚠️ Subject folder '{subject_name}' not found at: {subject_folder}")
        return

    # -----------------------------
    # STEP 1: Run Attendance Phase
    # -----------------------------
    logging.info(f"🕒 Running Attendance Mode for subject '{subject_name}' (10 seconds)...")

    attendance = FaceRecognizer(subject_name=subject_name)  # pass subject name
    cap = attendance.start_camera()

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > 10:
            logging.info("✅ Attendance phase complete. Moving to Cheating Detection...")
            break
        attendance.process_frame(cap)

    attendance.stop_camera(cap)

    # -----------------------------
    # STEP 2: Run Cheating Detection Phase
    # -----------------------------
    logging.info(f"🧠 Starting Cheating Detection for subject '{subject_name}'...")
    detector = CheatingDetector(subject_name=subject_name)  # pass subject
    detector.run()

    logging.info("🏁 Unified System Completed Successfully.")


if __name__ == "__main__":
    main()
