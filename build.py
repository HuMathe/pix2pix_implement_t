import models
from header import *
from torchvision.transforms import Normalize, ToTensor

save_path = 'results\\'
data_path = 'datasets\\acades\\'
dirs = ('train', )


def img2batch(img):
	img = ToTensor()(img)
	img = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img).numpy()
	return torch.FloatTensor((img, ))
def batch2img(batch):
	img = np.transpose(batch[0].cpu().numpy(), (1, 2, 0))
	img = (img*0.5 + 0.5)*255//1
	return np.uint8(img)
def fakeimg(netG, x):
	netG.eval()
	x = img2batch(x)
	Gxz = netG(x)
	img = batch2img(Gxz.detach())
	return img

def init():
	global netG 
	netG = models.modelGs.G_Unet(
		{
			'in_channels': 3,
			'out_channels': 3,
			'ngf': 64,
		}
	).to(torch.device("cpu"))
	mop = 'saved_models/facadesG_v1.pth.tar'
	state = torch.load(mop)
	netG.load_state_dict(state['netG'])
	print('init done...')


def process(rimg):
	global netG
	img = fakeimg(netG, rimg[:,256:,:])
	return img
	


def run_demo(dir):
	_dir = data_path + dir + '\\'
	__dir = save_path + dir + '\\'
	for path, _, files in os.walk(_dir):
		for file in files:
			if not file.endswith('.jpg'):
				continue
			img_path = os.path.join(path, file)
			print('processing: %s' % img_path, end = ' | ')
			img = cv2.imread(img_path)
			img = process(img)
			cv2.imwrite(os.path.join(__dir, file), img)
			print('saved result in: ' + os.path.join(__dir, file))
	return 


init()
for dir in dirs:
	run_demo(dir)



