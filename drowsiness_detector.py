import cv2
import mediapipe as mp
import pygame

# Initialize MediaPipe and Pygame
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
pygame.mixer.init()

# Load your custom alarm sound
pygame.mixer.music.load("E:\ML PROJECT\DriverDrowsiness/ala.mp3")

# Eye landmark indices (MediaPipe 468 landmarks)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Start webcam
cap = cv2.VideoCapture(0)

# Drowsiness detection threshold
score = 0
threshold = 5  # Number of frames with closed eyes

# Helper functions
def euclidean_distance(pt1, pt2):
    return ((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2) ** 0.5

def get_ear(eye_landmarks):
    # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    vertical1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    vertical2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    horizontal = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE]

            left_ear = get_ear(left_eye)
            right_ear = get_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.23:
                score += 1
            else:
                score = 0
                pygame.mixer.music.stop()

            if score > threshold:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
