import os
import cv2
def image_set(dir):
	images = []
	for root, _, files in os.walk(dir):
		for file in files:
			images.append(os.path.join(root, file))
	return sorted(images)

def trans(img, pos, flip):
	img = cv2.resize(img, (286, 286), interpolation = cv2.INTER_CUBIC)
	x, y = pos
	img = img[x: x+256, y: y+256, :]
	if flip:
		img = img[:,::-1,:].copy()
	return img