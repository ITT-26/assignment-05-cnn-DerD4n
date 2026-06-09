import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # The exact margin used in your training script
    MARGIN = 0.35 

    # Start webcam
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
        
        print("Starting camera... Press 'ESC' to exit.")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip image for a natural selfie-view, convert to RGB for MediaPipe
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect hands
            results = hands.process(rgb_image)

            h, w, _ = image.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the skeleton joints
                    mp_drawing.draw_landmarks(
                        image, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Calculate the tight bounding box around the detected landmarks
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, y_min = min(x_min, x), min(y_min, y)
                        x_max, y_max = max(x_max, x), max(y_max, y)
                    
                    # Draw tight bounding box (Red)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(image, 'Raw Detection', (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Calculate the expanded margin box (Exactly how your CNN crops it)
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    center_x = x_min + box_width / 2.0
                    center_y = y_min + box_height / 2.0
                    
                    # Force a square aspect ratio based on the longest side, plus margin
                    box_side = max(box_width, box_height) * (1.0 + MARGIN)
                    
                    left = int(max(0, round(center_x - box_side / 2.0)))
                    top = int(max(0, round(center_y - box_side / 2.0)))
                    right = int(min(w, round(center_x + box_side / 2.0)))
                    bottom = int(min(h, round(center_y + box_side / 2.0)))

                    # Draw the final crop box (Green)
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(image, f'CNN Input Crop (Margin: {MARGIN})', (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the output window
            cv2.imshow('MediaPipe Hand Debugger', image)
            
            # Exit loop if the 'ESC' key is pressed
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()