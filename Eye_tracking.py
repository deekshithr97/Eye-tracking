import cv2
import numpy as np
import math
import datetime
from openpyxl import Workbook
import os

# Function to calculate angle using Perimetry chart reference
def calculate_angle_perimetry(eye, center):
    dx = eye[0] - center[0]
    dy = eye[1] - center[1]
    angle = math.degrees(math.atan2(-dy, dx))  # Negative dy to match the Perimetry chart (top is 90°)
    if angle < 0:
        angle += 360  # Normalize to 0-360 degrees
    return angle

# Load Haar cascades for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create an Excel workbook and add a sheet
wb = Workbook()
ws = wb.active
ws.title = "Eye Tracking Data"
ws.append(["Timestamp", "Left Eye Angle", "Right Eye Angle"])  # Column headers

# Start video capture
cap = cv2.VideoCapture(0)

# Define an explicit path for the Excel file
file_path = os.path.join(os.getcwd(), "Eye_Tracking_Data_Perimetry.xlsx")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        left_eye_angle = None
        right_eye_angle = None

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            eye_centers = []

            for (ex, ey, ew, eh) in eyes:
                # Eye region
                eye = roi_gray[ey:ey + eh, ex:ex + ew]
                eye_color = roi_color[ey:ey + eh, ex:ex + ew]

                # Eye center
                eye_center = (ex + ew // 2, ey + eh // 2)
                eye_centers.append(eye_center)

                # Draw eye rectangle and center
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
                cv2.circle(roi_color, eye_center, 5, (0, 255, 0), -1)

                # Gaze estimation: simple approximation based on bright spots
                _, threshold = cv2.threshold(eye, 55, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 5:
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)
                        if radius > 3:
                            gaze_point = (int(cx), int(cy))
                            cv2.circle(eye_color, gaze_point, 5, (0, 0, 255), -1)

                            # Calculate angle relative to the Perimetry chart
                            angle = calculate_angle_perimetry(gaze_point, eye_center)
                            if len(eye_centers) == 1:  # Assuming the first eye is left
                                left_eye_angle = angle
                            elif len(eye_centers) == 2:  # Assuming the second eye is right
                                right_eye_angle = angle

        # Store the data with timestamp
        if left_eye_angle is not None and right_eye_angle is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ws.append([timestamp, left_eye_angle, right_eye_angle])
            print(f"Time: {timestamp}, Left Angle: {left_eye_angle:.2f}, Right Angle: {right_eye_angle:.2f}")

        cv2.imshow('Eye Tracking (Perimetry Chart)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Save the Excel file
    try:
        wb.save(file_path)
        print(f"Excel file saved successfully at: {file_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
    cap.release()
    cv2.destroyAllWindows()
