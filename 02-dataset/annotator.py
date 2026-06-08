import json
import uuid

import cv2
import sys
import os

# Annotator from PatrickGraf99

class ImageProcessor:

    def __init__(self):
        video_id = 0
        if len(sys.argv) > 1:
            video_id = int(sys.argv[1])

        # Create a video capture object for the webcam
        self.cap = cv2.VideoCapture(video_id)
        ret, self.frame = self.cap.read()

        self.WINDOW_NAME = "annotater"
        cv2.namedWindow(self.WINDOW_NAME)
        self.frozen = False
        self.annot_points = []
        self.annotating = False
        self.img_frozen = None
        self.img_frozen_original = None
        self.bboxes = []
        self.labels = []

    def update(self):
        # Do nothing if frame is frozen...
        if self.frozen:
            return
        # ...or being annotated
        if self.annotating:
            return
        # Capture a frame from the webcam
        ret, self.frame = self.cap.read()
        if not ret:
            print('Something went wrong with capturing a camera image.')
            return
        cv2.imshow(self.WINDOW_NAME, self.frame)


    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        sys.exit()

    def freeze(self):
        # Switch between frozen and unfrozen state
        if self.annotating:
            print('Freezing is not allowed during annotation')
            return
        if self.frozen:
            self.frozen = False
            return
        self.frozen = True
        # Store the original extra so we can save it later without any cirles
        self.img_frozen = self.frame.copy()
        self.img_frozen_original = self.img_frozen.copy()
        cv2.imshow(self.WINDOW_NAME, self.img_frozen)

    def handle_mouse_input(self, event, x, y, flags, param):
        # Ignore mouse if the frame is not frozen
        if self.annotating:
            return
        if not self.frozen:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add a circle and append the click to annot points
            print(f'Registered click at {x}, {y}')
            cv2.circle(self.img_frozen, (x, y), 3, (0, 0, 255), -1) # Draw a circle on the freeze
            cv2.imshow(self.WINDOW_NAME, self.img_frozen)
            # Force refresh displayed img
            cv2.waitKey(1)
            self.annot_points.append((x, y))
            # Once we have 2 points we can annotate the area
            if len(self.annot_points) == 2:
                self.annotate()

    def annotate(self):
        print('Calculating bbox')
        # Some math to calc the values for our bbox
        self.annotating = True
        top_left = self.annot_points[0]
        bottom_right = self.annot_points[1]
        # Let's also draw a rectangle around our box for visual clarity
        cv2.rectangle(self.img_frozen, top_left, bottom_right, (0, 0, 255), 2)
        cv2.imshow(self.WINDOW_NAME, self.img_frozen)
        self.annot_points = []

        x = top_left[0]
        y = top_left[1]

        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        bbox = [
            x / self.img_frozen.shape[1],
            y / self.img_frozen.shape[0],
            width / self.img_frozen.shape[1],
            height / self.img_frozen.shape[0]
        ]
        label = input('Please enter a label for the captured hand:')
        # Append our box and label
        self.bboxes.append(bbox)
        self.labels.append(label)
        print(f'Added {label}')
        print(f'Total annotations: {len(self.labels)}')
        self.annotating = False

    def save_annot(self):
        # Check if file exists and create it if not
        if not os.path.exists('annot-name.json'):
            with open('annot-name.json', 'w') as file:
                json.dump({}, file, indent=4)
        # Save the original freeze-frame with a random uuid
        annot_id = str(uuid.uuid4())
        cv2.imwrite(f'{annot_id}.jpg', self.img_frozen_original)

        # Try loading the json in the annot file
        with open('annot-name.json', 'r') as file:
            try:
                json_data = json.load(file)
            except json.decoder.JSONDecodeError:
                # Use an empty dict as a fallback if file is empty
                json_data = {}
        json_data[f'{annot_id}'] = {
            'bboxes': self.bboxes,
            'labels': self.labels
        }
        # Then write all our data
        with open('annot-name.json', 'w') as file:
            json.dump(json_data, file, indent=4)
        # And reset everything
        self.reset()

    def reset(self):
        self.annot_points = []
        self.annotating = False
        self.frozen = False
        self.img_frozen = None
        self.img_frozen_original = None
        self.labels = []
        self.bboxes = []

    def start(self):
        cv2.setMouseCallback(self.WINDOW_NAME, self.handle_mouse_input)
        while True:
            self.update()

            # Q to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
            # F to freeze a frame
            if key == ord('f'):
                self.freeze()

            # S to save
            if key == ord('s'):
                self.save_annot()

annotater = ImageProcessor()
annotater.start()