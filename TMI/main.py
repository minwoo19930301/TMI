# code to start the code in conda cmd
# python main.py --input input/( put file name ex:source.mov) --output output/(put file name - ex:result.mp4) --yolo yolo-coco



# search for ### for modified parts
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob

files = glob.glob('output/*.png')
for f in files:
   os.remove(f)

from sort import *
tracker = Sort()
memory = {}


### initialize counter before the loop of framing - which will count cars that were in box area for more than 3 secs
### if more than 3 cars are in the same area for more than 3sec, this means traffic jam in that square
counter = 0
### initialize just_counter for counting cars that stayed for more than 1 secs
### added this value because yolo was not perfect to analyze a certain car for more than 1 secs
### and sometimes recognizes the same car as a diffent new car, in certain cases such as 
### black cars in night time or freeze bugs in camera video or specular trucks that are not weighted yet. 
### so if there are more than 5~6 cars in an area for more than 1 sec this also is identified as traffic jam
just_counter =0

### our target box square area and make dictionary to put the car ids as key and how much frames they were in box as value
### this area plot changes by depending on the the camera zoom in/out gratitude and the angle of which road it is shooting.
#square = [(231, 600), (511, 735)]
square = [(194, 600), (511, 735)]
in_target_box = {}

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

### check if the car is in the box area
def in_box(target_pt1,target_pt2,object_pt):
	return object_pt[0]>target_pt1[0] and object_pt[0]<target_pt2[0] and object_pt[1]>target_pt1[1] and object_pt[1]<target_pt2[1]  

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
	
	dets = []
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			dets.append([x, y, x+w, y+h, confidences[i]])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	dets = np.asarray(dets)
	tracks = tracker.update(dets)

	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = {}
	for track in tracks:
		boxes.append([track[0], track[1], track[2], track[3]])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]

	if len(boxes) > 0:
		i = int(0)
		for box in boxes:
			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))

			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

			color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			
			
			if indexIDs[i] in previous:
				previous_box = previous[indexIDs[i]]
				(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
				(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
				p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
				p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
				cv2.line(frame, p0, p1, color, 3)
				cv2.rectangle(frame, (x, y), (w, h), color, 2)
				text = "{}".format(indexIDs[i])
				cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				
				### if the car is in the box then put the car id as key and add 1 in the value, initialize as 0 if None
				### then make the square as red
				### if not, then return the value back as 0 
				if in_box(square[0],square[1],p0):
					in_target_box[indexIDs[i]] = in_target_box.get(indexIDs[i],0) + 1
					cv2.rectangle(frame, (x, y), (w, h), (0,0,255), 2)
				else:
					in_target_box[indexIDs[i]]=0
			
		
			#print(	in_target_box)


			# text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			i += 1

	### for each item in in_target_box, if the value is in the box for more then 90 frames(3sec) then add counter
	### if more than 1 sec count as just_counter
	for key,value in in_target_box.items():
			#print(key,":",value)
			if value>90:
				counter +=1 
			if value>30:
				just_counter +=1
	#print("--------")

	### draw square for our target box
	cv2.rectangle(frame,square[0],square[1],(0, 255, 255),2)
	### draw counter and data related with the time
	#cv2.putText(frame, "cars which overlapsed 3 secs in box : "+str(counter), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 170, 0), 1)
	cv2.putText(frame, 'Time: ' + str(round(frameIndex / 30, 2)) + ' sec of ' + str(round( total/ 30, 1))
                    + ' sec', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0), 1)
	cv2.putText(frame, "Frame: " + str(frameIndex) + ' of ' + str(total), (10, 230), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 170, 0), 1)
	
	### if there are more than 3 cars that were in the box for more than 3 secs then there will be tailgating
	### if there are more than 5 cars that stayed more than 1 sec in such area, then there will be tailgating
	if counter > 3:
		cv2.putText(frame, "Tailgate Alert", (10,300), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
	if just_counter > 5:
		cv2.putText(frame, "Tailgate Alert", (10,300), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
	### initialize the counter and just_counter for next frame status in the loop
	counter = 0
	just_counter=0
	# saves image file
	### output the png file to check as image
	cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

	# increase frame index
	frameIndex += 1
	if frameIndex % 10 == 0:
		print(str(frameIndex)+"/"+str(total))
	if frameIndex >= 4000:
		print("[INFO] cleaning up...")
		writer.release()
		vs.release()
		exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()