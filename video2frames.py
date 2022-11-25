import cv2
import os

def FrameCapture(basepath,vidpath,imgpath):
	cap = cv2.VideoCapture(basepath+vidpath)

	count =0
	while(cap.isOpened()):
		ret, frame = cap.read()
		print(frame, ret)
		if ret:
			cv2.imwrite(basepath+imgpath+"frame%d.jpg" % count, frame)
			print(".\{}\frame{}.jpg".format(imgpath,count))
			count += 1
		else:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	FrameCapture("C:/Users/User/Documents/GitHub/calib_challenge/unlabeled/","5.hevc","5/")
	FrameCapture("C:/Users/User/Documents/GitHub/calib_challenge/unlabeled/","6.hevc","6/")
	FrameCapture("C:/Users/User/Documents/GitHub/calib_challenge/unlabeled/","7.hevc","7/")
	FrameCapture("C:/Users/User/Documents/GitHub/calib_challenge/unlabeled/","8.hevc","8/")
	FrameCapture("C:/Users/User/Documents/GitHub/calib_challenge/unlabeled/","9.hevc","9/")
