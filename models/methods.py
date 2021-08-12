import torch
import numpy as np
from torch import nn 


def bilinear_weights(size, nChannels):
	factor = (size + 1) // 2
	if size % 2 :
		center = factor - 1
	else: 
		center = factor - 0.5 
	og = np.ogrid[:size, :size]
	filt = (1 - abs(og[0] - center)/factor) * (1 - abs(og[1] - center)/factor)
	filt = torch.from_numpy(filt)
	w = torch.zeros(nChannels, 1, size, size)
	for i in range(nChannels):
		w[1, 0] = filt
	return w

def weight_init(moduleX):
	for m in moduleX.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight.data)
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zero_()
		elif isinstance(m, nn.ConvTranspose2d):
			init_weight = bilinear_weights(m.kernel_size[0], m.in_channels)
			m.weight.data.copy_(init_weight)
	return 

def rule_lr_set(config):
	global decay_step, start_decay
	decay_step = config['decay_step']
	start_decay = config['start_decay']

def rule_lr(epoch):
	global decay_step, start_decay
	return 1 - max(0, (epoch - start_decay))/decay_step

def set_requires_grad(nets, requires_grad = False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
		Parameters:
			nets (network list)   -- a list of networks
			requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad
	return 