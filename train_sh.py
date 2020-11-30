# coding: utf-8

import os
import sys
import argparse
import datetime
import torch
import torchvision
from torchsummary import summary
from trainer import Trainer
from model.HSCNN import HSCNN
from model.DeepSSPrior import DeepSSPrior
from model.HyperReconNet import HyperReconNet
from model.HyperMixNet import HyperMixNet
from model.layers import MSE_SAMLoss
from data_loader import PatchMaskDataset
from utils import RandomCrop, RandomHorizontalFlip, RandomRotation
from utils import ModelCheckPoint, Draw_Output


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--batch_size', '-b', default=64, type=int, help='Training and validatio batch size')
parser.add_argument('--epochs', '-e', default=150, type=int, help='Train eopch size')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='HSCNN', type=str, help='Model Name')
parser.add_argument('--block_num', '-bn', default=9, type=int, help='Model Block Number')
args = parser.parse_args()


dt_now = datetime.datetime.now()
batch_size = args.batch_size
epochs = args.epochs
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 32
data_name = args.dataset
model_name = args.model_name
block_num = args.block_num


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


img_path = f'{data_name}'
train_path = os.path.join(img_path, 'train_patch_data')
test_path = os.path.join(img_path, 'test_patch_data')
mask_path = os.path.join(img_path, 'mask_data')
callback_path = os.path.join(img_path, 'callback_path')
callback_mask_path = os.path.join(img_path, 'mask_show_data')
callback_result_path = os.path.join('result', f'{data_name}', f'{dt_now.month:02d}{dt_now.day:02d}', f'{model_name}_{block_num}')
os.makedirs(callback_result_path, exist_ok=True)
ckpt_path = os.path.join('ckpt_path', f'{data_name}', f'{dt_now.month:02d}{dt_now.day:02d}')


train_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
test_transform = None
train_dataset = PatchMaskDataset(train_path, mask_path, transform=train_transform, concat=concat_flag)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = PatchMaskDataset(test_path, mask_path, transform=test_transform, concat=concat_flag)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


model_obj = {'HSCNN': HSCNN, 'DeepSSPrior': DeepSSPrior, 'HyperReconNet': HyperReconNet, 'HyperMixNet': HyperMixNet}
activation = {'HSCNN': 'leaky', 'DeepSSPrior': 'relu', 'HyperReconNet': 'relu', 'HyperMixNet': 'relu'}


if model_name in list(model_obj.keys()):
    model = model_obj(input_ch, 31, block_num=block_num, activation=activation)
else:
    print('Enter Model Name')
    sys.exit(0)


model.to(device)
if model_name == 'HyperMixNet':
    criterion = MSE_SAMLoss(alpha=.5, beta=1 - .5).to(device)
else:
    criterion = torch.nn.MSELoss().to(device)
param = list(model.parameters())
optim = torch.optim.Adam(lr=1e-3, params=param)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 25, .5)


summary(model, (input_ch, 64, 64))
print(model_name)


# callback_dataset = PatchMaskDataset(callback_path, callback_mask_path, concat=concat_flag)
# draw_ckpt = Draw_Output(callback_dataset, data_name, save_path=callback_result_path, filter_path=filter_path)
ckpt_cb = ModelCheckPoint(ckpt_path, model_name + f'_{block_num}',
                          mkdir=True, partience=1, varbose=True)
trainer = Trainer(model, criterion, optim, scheduler=scheduler, callbacks=[ckpt_cb])
trainer.train(epochs, train_dataloader, test_dataloader)
