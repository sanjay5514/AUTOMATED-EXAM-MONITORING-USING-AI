Ensuring fairness during examinations has become increasingly challenging with the rise of modern devices such as smartphones, smartwatches, and wireless earphones. Traditional invigilation methods depend heavily on human supervision, which can be affected by fatigue, distraction, and limited attention.

This project introduces a Smart Surveillance System for Exam Integrity, an embedded AI solution designed to automate and enhance examination monitoring using real-time video analytics.

The system uses a Raspberry Pi 4 Model B (8 GB) as the primary processing unit and an Imou Ranger 2 IP camera for real-time video acquisition via RTSP. The incoming video stream is processed locally using OpenCV, dlib, and YOLOv11:

dlib handles facial recognition for student identity verification.

YOLOv11 detects cheating behaviours such as mobile phone usage, earphones, and suspicious movements.

All computations run on-device to ensure low latency, data privacy, and offline functionality. Whenever a violation is detected, the system sends an automatic alert to the invigilator via Python SMTP email, including the student’s details, timestamp, and an image snapshot.

Dataset creation and model training were carried out using Roboflow and Google Colab, resulting in a high-accuracy detection model optimized for embedded deployment.

This project showcases the potential of embedded AI to transform conventional exam supervision into a smart, automated, and privacy-preserving process, supporting transparency and strengthening academic integrity.
