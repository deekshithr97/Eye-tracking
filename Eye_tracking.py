import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from pygaze.libtracker import EyeTracker

def calculate_eye_angle(eye_contour):
    """
    Calculate the angle of the eye using its contour.
    """
    (x, y), (MA, ma), angle = cv2.fitEllipse(eye_contour)
    return angle

def process_frame(frame, tracker, data):
    """
    Process each frame to detect eye movement, calculate angles, and store data.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_contours, _ = cv2.findContours(cv2.Canny(eye_roi, 50, 150), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if eye_contours:
                # Find the largest contour (likely the eye)
                largest_contour = max(eye_contours, key=cv2.contourArea)
                angle = calculate_eye_angle(largest_contour)

                # Draw contour and angle on the frame
                eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                cv2.ellipse(frame, cv2.fitEllipse(largest_contour), (0, 255, 0), 2)
                cv2.putText(frame, f"Angle: {angle:.2f}", (eye_center[0] - 50, eye_center[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                # Record data
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data.append({'Timestamp': timestamp, 'Eye Angle': angle})

    return frame, data

def save_data_to_excel(data, filename="eye_tracking_data.xlsx"):
    """
    Save the eye tracking data to an Excel file.
    """
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)

def main():
    """
    Main function to track eye movement, calculate angles, and save data.
    """
    tracker = EyeTracker()  # PyGaze EyeTracker instance
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        processed_frame, data = process_frame(frame, tracker, data)
        cv2.imshow('Eye Tracking', processed_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save data to Excel
    save_data_to_excel(data)

if __name__ == "__main__":
    main()
