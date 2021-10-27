# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from imutils.video import VideoStream 
import numpy as np
import argparse
import imutils
import time
import cv2
import time, board, busio
import adafruit_mlx90640
import datetime
import paho.mqtt.client as mqtt
import json
import requests
from scipy import ndimage

THINGSBOARD_HOST = '10.170.69.226'
# ACCESS_TOKEN = 'RASPBERRY_PI_DEMO_TOKEN'
url_post= 'http://10.170.69.226:8080/api/v1/RASPBERRY_PI_DEMO_TOKEN/telemetry'
headers= {'Content-Type':'application/json',}

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(0.1)

# Kết nối camera Nhiệt
i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
mlx_shape = (24, 32)

# Phóng to kích thước ma trận nhiệt
mlx_interp_val = 10  
mlx_interp_shape = (mlx_shape[0] * mlx_interp_val, mlx_shape[1] * mlx_interp_val)  

count = 0


#Thingsboard
sensor_data={'temperature':0}
client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, 1883)
client.loop_start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	current_time = datetime.datetime.now()
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (100, 100)), 1.0,
		(100, 100), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	#
	count += 1

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < args["confidence"]:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)

		vface_temp = None
		x = startX - 10 if startX - 10 > 10 else startX + 10
		y = startY - 10 if startY - 10 > 10 else startY + 10
		'''26/10/2021: Modify the size of the frame'''
		cv2.rectangle(frame, (startX, startY), (endX - 20, endY - 50),
			(0, 0, 255), 2)
		#cv2.putText(frame, text, (startX, y),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		if count % 5 == 0: 
			# Đọc ảnh nhiệt từ camera
			frame1 = np.zeros(mlx_shape[0] * mlx_shape[1])
			mlx.getFrame(frame1)

			# Lật ảnh và phóng ảnh nhiệt
			data_array = np.fliplr(np.reshape(frame1, mlx_shape))  # reshape, flip data
			data_array = ndimage.zoom(data_array, mlx_interp_val)  # interpolate

			''' 26/10/2021: This code is to calculate the maximum of temperature (a pixel)'''
			vface_temp = round(np.max(data_array[y:y+h,x:w+x]), 2)
			#vface_temp = str(round(np.average(data_array[y:y+h,x:w+x]), 2)) # Average
			print('Nhiet do: {} Thoi gian: {}'.format(str(vface_temp), current_time))
			cv2.putText(frame, str(vface_temp), (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
			sensor_data['temperature']=vface_temp;
			test=requests.post(url_post, headers=headers,data=json.dumps(sensor_data))
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

client.loop_stop()
client.disconnect()
# do a bit of cleanup

cv2.destroyAllWindows()
vs.stop()
