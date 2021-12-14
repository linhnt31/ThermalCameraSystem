# USAGE
# python main_detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

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
import base64

from scipy import ndimage

# f=open("test.txt","w")
THINGSBOARD_HOST = '203.162.10.115'
ACCESS_TOKEN = 'RASPBERRY_PI_DEMO_TOKEN'

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
'''29/10/2021

check = 0
#Thingsboard: temperature and images '''
sensor_data={'temperature':0}
images = {}

client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, 1883)
client.loop_start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	
	'''29/10/2021: Get a image and push it to thingsboard'''
	frame = vs.read()
	frame = imutils.resize(frame, width=320, height=240)
	# grab the frame dimensions and convert it to a blob
	#print("Frame: {}".format(frame.shape))  # (300, 400, 3)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (100, 100)), 1.0,
		(100, 100), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	# Return detected objects including coordinates of x, y in the center of the object and the height and width of the box 
	detections = net.forward()

	# print("After NN: {}".format(detections.shape)) # (1, 1, 98, 7)
	#
	count += 1
	# loop over the detections
	for i in range(0, detections.shape[2]):
		detections

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
		# x = startX - 10 if startX - 10 > 10 else startX + 10
		# y = startY - 10 if startY - 10 > 10 else startY + 10
		x = startX + 100
		y = startY + 25
		'''26/10/2021: Modify the size of the frame'''
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		#cv2.putText(frame, text, (startX, y),
		#	cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		if count % 1 == 0: 
			# Đọc ảnh nhiệt từ camera
			frame1 = np.zeros(mlx_shape[0] * mlx_shape[1])
			mlx.getFrame(frame1)

			# Lật ảnh và phóng ảnh nhiệt
			data_array = np.fliplr(np.reshape(frame1, mlx_shape))  # reshape, flip data
			data_array = ndimage.zoom(data_array, mlx_interp_val)  # interpolate

			''' 26/10/2021: This code is to calculate the maximum of temperature (a pixel)'''
			try:
				vface_temp = round(np.max(data_array[y:y+h,x:w+x]), 2)
				vface_temp = round((vface_temp - 38 + 2.5) * 1.3 + 36.5, 2)
			except:
				continue
			#vface_temp = str(round(np.average(data_array[y:y+h,x:w+x]), 2)) # Average
			#print('Nhiet do: {} Thoi gian: {}'.format(str(vface_temp), current_time))
			cv2.putText(frame, str(vface_temp), (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
			sensor_data['temperature'] = vface_temp;
			client.publish('v1/devices/me/telemetry',json.dumps(sensor_data));
			# print(type(json.dumps(sensor_data)))

			
			current_time = datetime.datetime.now()
			img_name = "Person_{}.png".format(current_time)
			cv2.imwrite(img_name, frame)
			tbImg = open(img_name,'rb')
			image_read=tbImg.read()
			# # print(tbImg)
			png_as_text = str(base64.b64encode(image_read)).strip()
			# # print(type(png_as_text))
			# # print(png_as_text)
			images["img"] = str(png_as_text)[2:-1]
			# # f.write(str(images))
			# f.close()
			
			# print(type(images))
			# print(type(json.dumps(images)))
			
			client.publish('v1/devices/me/telemetry', json.dumps(images)) 
			
				
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
