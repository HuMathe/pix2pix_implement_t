from header import *


config_path = 'config/facades_train_config.yaml'

print("loading config from: %s"%config_path)
with open(config_path, 'rb') as f:
	config = f.read()
config = load(config)
print_config(config)
use_cuda = torch.cuda.is_available()
if config['use_cuda'] and not use_cuda:
	print("ERROR: no gpu found, please check your device")
	print("using CPU instead...")
	config['use_cuda'] = False
dataset = fetch_data(config)
model = netModel(config) # the last line (netG.train)
lD_, lG_, lGL1_ = [], [], []

print('training...')
for epoch in range(config['gap'], config['end_epoch']):
	start_time = time.time()
	print("eopch [%3d/%3d]: "%(epoch + 1, config['end_epoch']), end = '')
	lD, lG, lGL1 = [], [], []
	# train
	for data in dataset:
		ld, lg, lgl1 = model.learn(data)
		lD.append(ld)
		lG.append(lg)
		lGL1.append(lgl1)
	print('%.1fs %.3fD %.3fG %.3fGL1' % (time.time()-start_time, sum(lD)/len(lD), sum(lG)/len(lG), sum(lGL1)/len(lGL1)))
	# save model
	lD_.extend(lD)
	lG_.extend(lG)
	lGL1_.extend(lGL1)
	np.save('lossD', np.array(lD_))
	np.save('lossG', np.array(lG_))
	np.save('lossL1', np.array(lGL1_))
	model.save(epoch, config['save_path'])
	model.adjust_lr()
print('training done...')
