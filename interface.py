import numpy as np
import cv2
from CNN import predict

font = cv2.FONT_HERSHEY_SIMPLEX
textScale = 1
textThickness = 1
textColor = (255, 255, 255)
bgColor = (0, 0, 0)
textLocation = (20, 400)
xPadding = 5
yPadding = 10
predictionInterval = 1

def calculatePt2(text):
	(textSize, _) = cv2.getTextSize(text, font, textScale, textThickness)
	(pt2_x1, pt2_y1) = textLocation
	(pt2_x2, pt2_y2) = textSize
	pt2_x = pt2_x1 + pt2_x2 + xPadding
	pt2_y = pt2_y1 - pt2_y2 - yPadding
	return (pt2_x, pt2_y)

def main():
	text = ''
	cap = cv2.VideoCapture(0)
	index = 1

	while True:
		ret, frame = cap.read()
		(height, width, _) = frame.shape

		if index % predictionInterval == 0:
			prediction = predict(frame)
			if prediction != None:
				text += prediction
		index += 1

		if len(text) != 0:
			(pt1_x, pt1_y) = textLocation
			pt1_x -= xPadding
			pt1_y += yPadding
			pt1 = (pt1_x, pt1_y)

			pt2 = calculatePt2(text)
			(pt2_x, _) = pt2
			while pt2_x > width - pt1_x:
				text = text[1:]
				pt2 = calculatePt2(text)
				(pt2_x, _) = pt2

			cv2.rectangle(frame, pt1, pt2, (0, 0, 0), -1)
			cv2.putText(frame, text, textLocation, font, textScale, textColor, textThickness)

		cv2.imshow('frame', frame)
		k = cv2.waitKey(1) & 0xFF
		if k == ord('q'):
			break
		elif k == 8: #backspace
			text = text[0:-1]

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
