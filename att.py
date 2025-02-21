import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd


class AttendanceSystem:
    def __init__(self, images_path='Students'):
        # Create directories if they don't exist
        if not os.path.exists(images_path):
            os.makedirs(images_path)
            print(f"Created {images_path} directory. Please add student images there.")
        if not os.path.exists('attendance'):
            os.makedirs('attendance')

        self.images_path = images_path
        self.known_faces = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Load known faces from the images directory"""
        for image_file in os.listdir(self.images_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                # Load image
                current_img = face_recognition.load_image_file(f'{self.images_path}/{image_file}')
                # Get encoding
                encoding = face_recognition.face_encodings(current_img)[0]
                # Get name from filename (without extension)
                name = os.path.splitext(image_file)[0]
                
                self.known_faces.append(encoding)
                self.known_names.append(name)
                print(f'Loaded: {name}')

    def mark_attendance(self, name):
        """Mark attendance in CSV file"""
        date = datetime.now().strftime('%Y-%m-%d')
        time = datetime.now().strftime('%H:%M:%S')
        filename = f'attendance/attendance_{date}.csv'

        # Create new CSV if it doesn't exist
        if not os.path.exists(filename):
            df = pd.DataFrame(columns=['Name', 'Time'])
            df.to_csv(filename, index=False)

        # Read existing attendance
        df = pd.read_csv(filename)
        
        # Only mark attendance if not already marked
        if name not in df['Name'].values:
            new_row = {'Name': name, 'Time': time}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(filename, index=False)
            print(f'Marked attendance for {name}')

    def run(self):
        """Run the attendance system"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            # Resize frame for faster processing
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            
            # Convert BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Process each face
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                # Compare with known faces
                matches = face_recognition.compare_faces(self.known_faces, face_encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_names[first_match_index]
                    self.mark_attendance(name)

                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw box and name
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)

            # Show frame
            cv2.imshow('Attendance System', img)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Attendance System...")
    print("Please ensure student images are in the 'Students' directory")
    print("Press 'q' to quit")
    
    system = AttendanceSystem(images_path='D:\OOPM viva\project\Students')
    system.run()