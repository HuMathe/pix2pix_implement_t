import torch 
import numpy as np
from torch import nn 
from models.methods import weight_init

class D_Markovian(nn.Module): # Markovian discriminator
	def __init__(self, config):
		input_c = config['in_channels']
		output_c = config['out_channels']
		ndf = config['ndf']
		n_layers = config['n_layers']
		super(D_Markovian, self).__init__()
		self.layer = nn.Sequential(nn.Conv2d(input_c + output_c, ndf, 4, stride = 2, padding = 1),
								   nn.LeakyReLU(0.2, True))
		laer = plaer = 1
		for i in range(1, n_layers):
			plaer = laer
			laer = min(8, 2**i)
			self.layer.add_module('conv'+str(i), nn.Conv2d(ndf * plaer, ndf * laer, 4, stride = 2, padding = 1))
			self.layer.add_module('bnorm'+str(i), nn.BatchNorm2d(ndf * laer))
			self.layer.add_module('relu'+str(i), nn.LeakyReLU(0.2, True))

		plaer = laer
		laer = min(8, 2**i)
		self.layer.add_module('conv'+str(n_layers), nn.Conv2d(ndf * plaer, ndf * laer, 4, stride = 1, padding = 1))
		self.layer.add_module('bnorm'+str(n_layers), nn.BatchNorm2d(ndf * laer))
		self.layer.add_module('relu'+str(n_layers), nn.LeakyReLU(0.2, True))		
		
		self.layer.add_module('conv_', nn.Conv2d(ndf * laer, 1, 4, stride = 1, padding = 1))
		self.layer.add_module('sigmoid', nn.Sigmoid())
		weight_init(self)
		return 

	def forward(self, x):
		return self.layer(x)