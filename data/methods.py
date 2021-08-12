import os
def image_set(dir):
	images = []
	for root, dirs, files in os.walk(dir):
		for file in files:
			images.append(os.path.join(root, file))
	return sorted(images)
