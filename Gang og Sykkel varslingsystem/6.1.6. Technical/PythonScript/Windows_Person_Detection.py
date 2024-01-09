import cv2
import imutils
import numpy as np
from multiprocessing import Process, Queue
from flask import Flask
from flask_mqtt import Mqtt
import time  # Import the time module

def load_model():
    protopath = "model/object_detection/MobileNetSSD_deploy.prototxt"
    modelpath = "model/object_detection/MobileNetSSD_deploy.caffemodel"
    return cv2.dnn.readNetFromCaffe(protopath, modelpath)
def process_frame(frame, detector, classes, confidence_threshold):
    frame = imutils.resize(frame, width=720)
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    detector.setInput(blob)
    detections = detector.forward()

    people_cnt = 0
    bike_cnt = 0

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] in ["person", "bicycle"]:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                color = (0, 0, 255) if CLASSES[idx] == "person" else (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                if CLASSES[idx] == "person":
                    people_cnt += 1
                else:
                    bike_cnt += 1

    return frame, people_cnt, bike_cnt

def camera_processing(camera_index, output_queue):
    try:
        detector = load_model()
        classes = ["background", "aeroplane", "bicycle", "bird", "boat", ...]
        confidence_threshold = 0.7
        cap = cv2.VideoCapture(camera_index)
        while True:
            ret, frame = cap.read()
            if ret:
                processed_frame, people_count, bike_count = process_frame(frame, detector, classes, confidence_threshold)
                # cv2.imshow(f"Camera {camera_index}", processed_frame)  # Commented out for testing
                output_queue.put((camera_index, people_count, bike_count))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    except Exception as e:
        print(f"Error in camera_processing: {e}")

def mqtt_processing(input_queue):
    app = Flask(__name__)
    app.config['MQTT_BROKER_URL'] = 'broker.hivemq.com'
    app.config['MQTT_BROKER_PORT'] = 1883
    app.config['MQTT_KEEPALIVE'] = 5
    app.config['MQTT_TLS_ENABLED'] = False
    mqtt = Mqtt(app)

    @mqtt.on_connect()
    def handle_connect(client, userdata, flags, rc):
        mqtt.subscribe('Dugeleg/VisionTest/#')

    @mqtt.on_message()
    def handle_mqtt_message(client, userdata, message):
        print(f"Data from mqtt: {message.payload.decode()}")

    while True:
        while not input_queue.empty():
            camera_index, people_count, bike_count = input_queue.get()
            mqtt.publish(f"Dugeleg/VisionTest/Camera{camera_index}/Personer", str(people_count))
            mqtt.publish(f"Dugeleg/VisionTest/Camera{camera_index}/Sykkler", str(bike_count))
        time.sleep(0.1)

# Load the model
#protopath = "model/object_detection/MobileNetSSD_deploy.prototxt"
#modelpath = "model/object_detection/MobileNetSSD_deploy.caffemodel"
#detector = cv2.dnn.readNetFromCaffe(protopath, modelpath)

# Class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7

if __name__ == '__main__':
    output_queue = Queue()

    camera1_process = Process(target=camera_processing, args=(0, output_queue))
    camera2_process = Process(target=camera_processing, args=(1, output_queue))
    mqtt_process = Process(target=mqtt_processing, args=(output_queue,))

    camera1_process.start()
    camera2_process.start()
    mqtt_process.start()

    camera1_process.join()
    camera2_process.join()
    mqtt_process.join()

    cv2.destroyAllWindows()
