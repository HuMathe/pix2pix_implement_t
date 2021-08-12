import torch
from modelGs import *
from modelDs import *
from methods import rule_lr, rule_lr_set, set_requires_grad

def netModel(config):
	"""
		a model including :
			> netG: generator
			> netD: discriminator
	"""
	model = collect_model(config)
	if config['istrain']:
		model.train()
	else:
		model.eval()
	return 

def collect_model(config):
	if config['netG_type'] == 'Unet':
		netG = G_Unet(config)
	
	if config['istrain'] == False:
		netD = None
	elif config['netD_type'] == 'Markovian':
		netD = D_Markovian(config)
	return modelGAN(netG, netD, config)

class modelGAN():
	def __init__(self, netG, netD, config):
		
		if config['use_cuda']:
			self.device = torch.device('cuda:'+str(config['gpu_id']))
		else:
			self.device = torch.device('cpu')
		self.lambda_ = config['lambda_']
		self.netG = netG.to(self.device)
		self.netD = netD.to(self.device)
		if config['istrain']:
			rule_lr_set(config)
			self.criterion_GAN = nn.MSELoss().to(self.device)
			self.criterion_L1 = nn.L1Loss().to(self.device)
			self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr = config['lr'], betas = config['betas'])
			self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr = config['lr'], betas = config['betas'])
			self.schedulerG = torch.optim.lr_scheduler.LambdaLR(self.optimizerG, lr_lambda = rule_lr)
			self.schedulerD = torch.optim.lr_scheduler.LambdaLR(self.optimizerD, lr_lambda = rule_lr)
	
	def train(self):
		self.netD.train()
		self.netG.train()
		return 
	
	def eval(self):
		self.netG.eval()
		self.netD.eval()
		return 
	
	def learn(self, data):
		"""
			data = (x, y)
				x: label
				y: image
		"""	

		# train discriminator
		x, y = data
		x = x.to(self.device)
		y = y.to(self.device)
		z = self.netG(x)
		set_requires_grad(self.netD, True)
		self.optimizerD.zero_grad()
		# train with fake image
		fake_input = torch.cat((x, z), 1)
		fake_output = self.netD(fake_input.detach())
		lD_fake = self.criterion_GAN(fake_output, False)

		# train with real image
		real_input = torch.cat((x, y), 1)
		real_output = self.netD(real_input)
		lD_real = self.criterion_GAN(real_output, True)

		lD = (lD_fake + lD_real) * 0.5
		lD.backward()

		self.optimizerD.step()

		# train generator
		set_requires_grad(self.netD, False)
		self.optimizerG.zero_grad()

		fake_input = torch.cat((x, z), 1)
		fake_output = self.netD(fake_input)
		lG = self.criterion_GAN(fake_output, True)
		lGL1 = self.criterion_L1(fake_output, y) * self.lambda_
		lossG = lG + lGL1
		lossG.backward()
		self.optimizerG.step()

		return lD.data.item(), lG.data.item(), lGL1.data.item()
	
	def adjust_lr(self):
		self.schedulerG.step()
		self.schedulerD.step()
		return 
	
	def save(self, epoch, root, filen = 'checkpoint.pth.tar'):
		torch.save(
			{
				'epoch': epoch,
				'netG': self.netG.state_dict(),
				'netD': self.netD.state_dict(),
				'optimG': self.optimizerG.state_dict(),
				'optimD': self.optimizerD.state_dict(),
			}, 
			root + filen
		)
		return 