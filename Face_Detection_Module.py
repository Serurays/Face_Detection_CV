import cv2
import mediapipe as mp
import time


class Face_Detector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(self.min_detection_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_face(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.face_detection.process(img_rgb)

        bounding_box_list = []

        for d_id, detection in enumerate(self.results.detections):
            h, w, c = img.shape
            bounding_box_class = detection.location_data.relative_bounding_box
            bounding_box = int(bounding_box_class.xmin * w), int(bounding_box_class.ymin * h), \
                int(bounding_box_class.width * w), int(bounding_box_class.height * h)

            bounding_box_list.append([d_id, bounding_box, detection.score])

            if draw:
                img = self.fancy_draw(img, bounding_box)
                cv2.putText(img, f"{int(detection.score[0] * 100)}%", (bounding_box[0], bounding_box[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 3, (100, 200, 100), 2)

        return img, bounding_box_list

    def fancy_draw(self, img, bbox, rectangle_thickness=1, line_thickness=5, line_length=30):
        x, y, w, h = bbox

        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (200, 100, 100), rectangle_thickness)

        cv2.line(img, (x, y), (x + line_length, y), (200, 100, 100), line_thickness)
        cv2.line(img, (x, y), (x, y + line_length), (200, 100, 100), line_thickness)

        cv2.line(img, (x1, y), (x1 - line_length, y), (200, 100, 100), line_thickness)
        cv2.line(img, (x1, y), (x1, y + line_length), (200, 100, 100), line_thickness)

        cv2.line(img, (x, y1), (x + line_length, y1), (200, 100, 100), line_thickness)
        cv2.line(img, (x, y1), (x, y1 - line_length), (200, 100, 100), line_thickness)

        cv2.line(img, (x1, y1), (x1 - line_length, y1), (200, 100, 100), line_thickness)
        cv2.line(img, (x1, y1), (x1, y1 - line_length), (200, 100, 100), line_thickness)

        return img


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0

    while True:
        success, img = cap.read()

        if not success:
            print('Video ended.')
            break

        detector = Face_Detector()
        img, bbox_list = detector.find_face(img)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, f"FPS: {fps}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (70, 120, 150), 2)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
