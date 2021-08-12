import torch
from dataset_type import *
def fetch_data(config): # create a dataloader
	"""
		1. create a dataset (a specific class)
		2. create a dataloader and return it
	"""
	dataset = collect_data(config)
	return torch.utils.data.DataLoader(
				dataset, 
				batch_size = config['batch_size'], 
				shuffle = config['istrain'], 
				num_worker = config['num_workers']
			)

def collect_data(config):
	data_class = None
	if config['dstype'] == 1: # /d/train, /d/val, /d/test (facades)
		data_class = typeI_dataset
	return data_class(config)
