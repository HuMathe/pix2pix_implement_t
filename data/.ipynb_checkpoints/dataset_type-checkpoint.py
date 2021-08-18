import cv2
import torch
import random
from data.methods import image_set, trans
from torchvision import transforms
class typeI_dataset(torch.utils.data.Dataset):
	def __init__(self, config):
		data_path = 'datasets/' + config['dataset']
		if config['istrain'] :
			data_path += '/train/'
		else:
			data_path += '/test/'
		self.data_path = data_path
		self.images = image_set(data_path)
		return 
	
	def __getitem__(self, idx):
		"""
			return 2 val x and y:
				x: label
				y: groud truth
		"""
		path = self.images[idx]
		#print(path)
		img_raw = cv2.imread(path)
		x, y = img_raw[:,256:,:], img_raw[:,:256,:]
		pos = random.randint(0, 30), random.randint(0, 30)
		flip = random.random() > 0.5
		x, y = trans(x, pos, flip), trans(y, pos, flip)
		x, y = transforms.ToTensor()(x), transforms.ToTensor()(y)
		#print(x.shape)
		x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
		y = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(y)
		return x, y
	def __len__(self):
		return len(self.images)