# coding: utf-8


import os
import sys
import torch
from torchsummary import summary
from data_loader import PatchEvalDataset
from model.HSCNN import HSCNN
from model.DeepSSPrior import DeepSSPrior
from model.HyperReconNet import HyperReconNet
from model.HyperMixNet import HyperMixNet
from evaluate import PSNRMetrics, SAMMetrics
from evaluate import ReconstEvaluater
from pytorch_ssim import SSIM


parser = argparse.ArgumentParser(description='Evaluate Model')
parser.add_argument('--dataset', '-d', default='Harvard', type=str, help='Select dataset')
parser.add_argument('--concat', '-c', default='False', type=str, help='Concat mask by input')
parser.add_argument('--model_name', '-m', default='HSCNN', type=str, help='Model Name')
parser.add_argument('--block_num', '-b', default=9, type=int, help='Model Block Number')
args = parser.parse_args()


device = 'cuda' if touch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
data_name = args.dataset
if args.concat == 'False':
    concat_flag = False
    input_ch = 1
else:
    concat_flag = True
    input_ch = 32
model_name = args.model_name
block_num = args.block_num
img_path = f'{data_name}'
test_path = os.path.join(img_path, 'eval_data')
mask_path = os.path.join(img_path, 'eval_mask_data')
ckpt_dir = os.path.join(f'ckpt_path/{data_name}', f'{model_name}_{block_num}')
ckpt_list = os.listdir(ckpt_dir)
ckpt_list.sort()
ckpt_path = os.path.join(ckpt_dir, ckpt_list[-1])
output_path = os.path.join('result', data_name, f'{model_name}_{block_num}')
output_img_path = os.path.join(output_path, 'output_img')
output_mat_path = os.path.join(output_path, 'output_mat')
output_csv_path = os.path.join(output_path, 'output.csv')
os.makedirs(output_path, exist_ok=True)
os.makedirs(output_img_path, exist_ok=True)
os.makedirs(output_mat_path, exist_ok=True)


model_obj = {'HSCNN': HSCNN, 'DeepSSPrior': DeepSSPrior, 'HyperReconNet': HyperReconNet, 'HyperMixNet': HyperMixNet}
activation = {'HSCNN': 'leaky', 'DeepSSPrior': 'relu', 'HyperReconNet': 'relu', 'HyperMixNet': 'relu'}


if __name__ == '__main__':

    test_dataset = PatchEvalDataset(test_path, mask_path, transform=None, concat=concat_flag)
    model_name = args.model_name
    if model_name in list(model_obj.keys()):
        model = model_obj(input_ch, 31, block_num=block_num, activation=activation)
    else:
        print('Enter Model Name')
        sys.exit(0)

    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    psnr_evaluate = PSNRMetrics().to(device).eval()
    ssim_evaluate = SSIM().to(device).eval()
    sam_evaluate = SAMMetrics().to(device).eval()
    evaluate_fn = [psnr_evaluate, ssim_evaluate, sam_evaluate]

    evaluate = ReconstEvaluater(data_name, output_img_path, output_mat_path, output_csv_path)
    evaluate.metrics(model, test_dataset, evaluate_fn, ['PSNR', 'SSIM', 'SAM'])
