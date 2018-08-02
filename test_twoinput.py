import os,sys
import cv2
import numpy as np
from PIL import Image
import pyflow



# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

if __name__ == '__main__':
		img1_path = sys.argv[1]
		img2_path = sys.argv[2]
		img1_out_path = sys.argv[3]
		print (img1_path, img2_path)
		
		im1 = np.array(Image.open(img1_path))
		im2 = np.array(Image.open(img2_path))
		im1 = im1.astype(float) / 255.
		im2 = im2.astype(float) / 255.
		
		u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)

		flow = np.concatenate((u[..., None], v[..., None]), axis=2)
		print (flow.shape)
		hsv = np.zeros(im1.shape, dtype=np.uint8)
		hsv[:, :, 0] = 255
		hsv[:, :, 1] = 255
		mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		hsv[..., 0] = ang * 180 / np.pi / 2
		hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		cv2.imwrite(img1_out_path, rgb)
		# cv2.imshow('abc',rgb)
		# cv2.waitKey()
		# exit()

		

        
