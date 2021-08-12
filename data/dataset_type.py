import cv2
import torch
from methods import image_set
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
		img_raw = cv2.imread(path)
		x, y = img_raw[:,256:,:], img_raw[:,:256,:]
		x, y = transforms.ToTensor()(x), transforms.ToTensor()(y)
		#print(x.shape)
		x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
		y = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(y)
		return x, y
	def __len__(self):
		return len(self.images)