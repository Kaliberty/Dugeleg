import jetson_inference as inference
import jetson_utils as utils
import cv2
import threading
import numpy as np
import paho.mqtt.client as mqtt
import time
import RPi.GPIO as GPIO


# Global variables
sensor_node = 1
LocalPeople = 0
LocalBikes = 0
LocalVehicles = 0
ExternalPeople = 0
ExternalBikes = 0
ExternalVehicles = 0
MQTT_Duty_Cycle = 100  # Duty cycle in milliseconds
show_camera_feed = True  # Set this to False if you don't want to display the camera feed

# GPIO stuff
GPIO_Test = False
PWM_Light = True
Transistor_Type = "PNP" #PNP or NPN

if GPIO_Test == True:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(7, GPIO.OUT)

if PWM_Light:
    # Setup PWM if not already done
    GPIO.setmode(GPIO.BOARD)
    pwm = GPIO.PWM(32, 500)
    pwm.start(1)


class CSI_Camera:
    def __init__(self):
        self.video_capture = None
        self.frame = None
        self.grabbed = False
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(gstreamer_pipeline_string, cv2.CAP_GSTREAMER)
            self.grabbed, self.frame = self.video_capture.read()
        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)

    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        if self.video_capture is not None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                if grabbed:
                    # Flip the frame here
                    frame = cv2.flip(frame, 1)  # Change 1 to 0 for vertical flip

                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        if self.read_thread is not None:
            self.read_thread.join()

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=5, flip_method=2):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class CameraThread(threading.Thread):
    def __init__(self, camera_source, is_csi_camera=False):
        threading.Thread.__init__(self)
        self.camera_source = camera_source
        self.is_csi_camera = is_csi_camera
        self.frame = None
        self.grabbed = False
        self.running = False

        if is_csi_camera:
            self.camera = CSI_Camera()
            self.camera.open(gstreamer_pipeline(sensor_id=camera_source, flip_method=0))
            self.camera.start()
        else:
            self.camera = cv2.VideoCapture(camera_source)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def run(self):
        self.running = True
        while self.running:
            if self.is_csi_camera:
                self.grabbed, self.frame = self.camera.read()
            else:
                self.grabbed, self.frame = self.camera.read()

    def stop(self):
        self.running = False
        if self.is_csi_camera:
            self.camera.stop()
            self.camera.release()
        else:
            self.camera.release()

    def get_frame(self):
        return self.grabbed, self.frame

def on_connect(client, userdata, flags, rc):
    global sensor_node
    print("Connected with result code "+str(rc))
    # Subscribe to the opposite sensor's topics based on Sensor_Node value
    if sensor_node == 1:
        other_node = 2
    else:
        other_node = 1

    client.subscribe(f"Dugeleg/Varslingsystem/Sensor{other_node}/#")

def on_message(client, userdata, msg):
    global ExternalPeople, ExternalBikes, ExternalVehicles, sensor_node

    topic = msg.topic
    payload = msg.payload.decode()
    print(f"Received '{payload}' from '{topic}' topic")

    if sensor_node == 1:
        # Extract the sensor data from the MQTT message
        if topic == "Dugeleg/Varslingsystem/Sensor2/People":
            ExternalPeople = int(payload)
        elif topic == "Dugeleg/Varslingsystem/Sensor2/Bikes":
            ExternalBikes = int(payload)
        elif topic == "Dugeleg/Varslingsystem/Sensor2/Vehicles":
            ExternalVehicles = int(payload)
    else:
        # Extract the sensor data from the MQTT message
        if topic == "Dugeleg/Varslingsystem/Sensor1/People":
            ExternalPeople = int(payload)
        elif topic == "Dugeleg/Varslingsystem/Sensor1/Bikes":
            ExternalBikes = int(payload)
        elif topic == "Dugeleg/Varslingsystem/Sensor1/Vehicles":
            ExternalVehicles = int(payload)

def setup_mqtt_client():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("broker.hivemq.com", 1883, 60)
    return client

def run_cameras(mqtt_client):
    global LocalPeople, LocalBikes, LocalVehicles, MQTT_Duty_Cycle, show_camera_feed, sensor_node, GPIO_Test, PWM_Light
    left_camera_thread = CameraThread(0, is_csi_camera=True)
    right_camera_thread = CameraThread(1, is_csi_camera=True)

    left_camera_thread.start()
    right_camera_thread.start()

    net = inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

    if show_camera_feed:
        cv2.namedWindow("Dual Camera Object Detection", cv2.WINDOW_AUTOSIZE)

    categories_to_track = ["bicycle", "person", "car", "truck"]
    object_counts = {category: 0 for category in categories_to_track}

    scale_factor = 0.5

    try:
        while True:
            grabbed_left, left_image = left_camera_thread.get_frame()
            grabbed_right, right_image = right_camera_thread.get_frame()

            if grabbed_left and grabbed_right:
            
                width = int(left_image.shape[1] * scale_factor)
                height = int(left_image.shape[0] * scale_factor)
                left_image = cv2.resize(left_image, (width, height))
                right_image = cv2.resize(right_image, (width, height))


                left_image_rgba = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGBA)
                left_image_cuda = utils.cudaFromNumpy(left_image_rgba)
                detections_left = net.Detect(left_image_cuda)

                right_image_rgba = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGBA)
                right_image_cuda = utils.cudaFromNumpy(right_image_rgba)
                detections_right = net.Detect(right_image_cuda)

                for category in categories_to_track:
                    object_counts[category] = 0

                for detection in detections_left + detections_right:
                    class_id = net.GetClassDesc(detection.ClassID)
                    if class_id in categories_to_track:
                        object_counts[class_id] += 1

                # Update global variables with object counts
                LocalPeople = object_counts['person']
                LocalBikes = object_counts['bicycle']
                LocalVehicles = object_counts['car'] + object_counts['truck']

                # Call Light_Test function to update light state
                if GPIO_Test == True:
                    Light_Test()

                if PWM_Light == True:
                    control_light_based_on_time()

                # Dynamic topic prefix based on Sensor_Node
                topic_prefix = f"Dugeleg/Varslingsystem/Sensor{sensor_node}"

                # Publish MQTT messages with a duty cycle
                if (time.time() * 1000) % MQTT_Duty_Cycle < 30:  # Adjust the 30ms threshold as needed
                    mqtt_client.publish(f"{topic_prefix}/Pedestrians", str(LocalPeople))
                    mqtt_client.publish(f"{topic_prefix}/Bikes", str(LocalBikes))
                    mqtt_client.publish(f"{topic_prefix}/Vehicles", str(LocalVehicles))

                if show_camera_feed:
                    combined_image = np.hstack((cv2.cvtColor(utils.cudaToNumpy(left_image_cuda), cv2.COLOR_RGBA2BGR),
                                                cv2.cvtColor(utils.cudaToNumpy(right_image_cuda), cv2.COLOR_RGBA2BGR)))
                    cv2.imshow("Dual Camera Object Detection", combined_image)
                    if cv2.waitKey(30) & 0xFF == 27:
                        break

            else:
                time.sleep(0.03)  # Sleep for a brief moment

    except KeyboardInterrupt:  # Handle Ctrl+C gracefully
        print("Stopping cameras and exiting...")
    finally:
        left_camera_thread.stop()
        right_camera_thread.stop()
        left_camera_thread.join()
        right_camera_thread.join()
        if show_camera_feed:
            cv2.destroyAllWindows()

def Light_Test():
    global LocalPeople, LocalBikes, LocalVehicles, ExternalPeople, ExternalBikes, ExternalVehicles, Transistor_Type
    Light = (LocalPeople + ExternalPeople >= 1) and (LocalBikes + ExternalBikes >= 1)
    if Transistor_Type == "NPN":
        GPIO.output(7, Light)
    elif Transistor_Type == "PNP":
        GPIO.output(7, not Light)
    else:
        GPIO.output(7, False)

def control_light_based_on_time():
    global LocalPeople, LocalBikes, LocalVehicles, ExternalPeople, ExternalBikes, ExternalVehicles, pwm, PWM_Light, Transistor_Type

    Light = (LocalPeople + ExternalPeople >= 1) and (LocalBikes + ExternalBikes >= 1)

    cTime = time.strftime("%H:%M")
    Sunrise = "09:30"
    Sunset = "20:30"
    
    if Transistor_Type == "NPN":
        if Light == True:
            if (cTime > Sunrise) and (cTime < Sunset):
                pwm.ChangeDutyCycle(90)
            else:
                pwm.ChangeDutyCycle(50)
        else:
            pwm.ChangeDutyCycle(0)
    elif Transistor_Type == "PNP":
        if Light == True:
            if (cTime > Sunrise) and (cTime < Sunset):
                pwm.ChangeDutyCycle(10)
            else:
                pwm.ChangeDutyCycle(50)
        else:
            pwm.ChangeDutyCycle(100)

if __name__ == "__main__":
    mqtt_client = setup_mqtt_client()
    mqtt_client.loop_start()
    run_cameras(mqtt_client)
    mqtt_client.loop_stop()
