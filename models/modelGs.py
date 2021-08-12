import torch 
import numpy as np
from torch import nn
from methods import weight_init


class G_Unet(nn.Module): # UNet
	def __init__(self, input_c, output_c, ngf):
		super(G_Unet, self).__init__()
		self.encoder_1 = nn.Conv2d(input_c, ngf, 4, stride = 2, padding = 1, bias = False)
		self.encoder_2 = nn.Sequential(nn.LeakyReLU(0.2, True),
									   nn.Conv2d(ngf, ngf * 2, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 2))
		self.encoder_3 = nn.Sequential(nn.LeakyReLU(0.2, True),
									   nn.Conv2d(ngf * 2, ngf * 4, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 4))
		self.encoder_4 = nn.Sequential(nn.LeakyReLU(0.2, True),
									   nn.Conv2d(ngf * 4, ngf * 8, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 8))
		self.encoder_5 = nn.Sequential(nn.LeakyReLU(0.2, True),
									   nn.Conv2d(ngf * 8, ngf * 8, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 8))
		self.encoder_6 = nn.Sequential(nn.LeakyReLU(0.2, True),
									   nn.Conv2d(ngf * 8, ngf * 8, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 8))
		self.encoder_7 = nn.Sequential(nn.LeakyReLU(0.2, True),
									   nn.Conv2d(ngf * 8, ngf * 8, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 8))
		self.encoder_8 = nn.Sequential(nn.LeakyReLU(0.2, True),
									   nn.Conv2d(ngf * 8, ngf * 8, 4, stride = 2, padding = 1, bias = False))
		self.decoder_1 = nn.Sequential(nn.ReLU(True),
									   nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 8),
									   nn.Dropout(0.5))
		self.decoder_2 = nn.Sequential(nn.ReLU(True),
									   nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 8),
									   nn.Dropout(0.5))
		self.decoder_3 = nn.Sequential(nn.ReLU(True),
									   nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 8),
									   nn.Dropout(0.5))
		self.decoder_4 = nn.Sequential(nn.ReLU(True),
									   nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 8))
		self.decoder_5 = nn.Sequential(nn.ReLU(True),
									   nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 4))
		self.decoder_6 = nn.Sequential(nn.ReLU(True),
									   nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf * 2))
		self.decoder_7 = nn.Sequential(nn.ReLU(True),
									   nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, stride = 2, padding = 1, bias = False),
									   nn.BatchNorm2d(ngf))
		self.decoder_8 = nn.Sequential(nn.ReLU(True), 
									   nn.ConvTranspose2d(ngf * 2, output_c, 4, stride = 2, padding = 1),
									   nn. Tanh())
		weight_init(self)


	def forward(self, x):
		e1 = self.encoder_1(x)
		e2 = self.encoder_2(e1)
		e3 = self.encoder_3(e2)
		e4 = self.encoder_4(e3)
		e5 = self.encoder_5(e4)
		e6 = self.encoder_6(e5)
		e7 = self.encoder_7(e6)
		e8 = self.encoder_8(e7)
		d1 = torch.cat((self.decoder_1(e8), e7), 1)
		d2 = torch.cat((self.decoder_2(d1), e6), 1)
		d3 = torch.cat((self.decoder_3(d2), e5), 1)
		d4 = torch.cat((self.decoder_4(d3), e4), 1)
		d5 = torch.cat((self.decoder_5(d4), e3), 1)
		d6 = torch.cat((self.decoder_6(d5), e2), 1)
		d7 = torch.cat((self.decoder_7(d6), e1), 1)
		d8 = self.decoder_8(d7)
		return d8

