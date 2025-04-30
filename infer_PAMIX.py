import os
import torch
import warnings
import numpy as np
from PIL import Image
from torchvision import transforms
from models.PAMIX_Family import PAMIX

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS" #设备排序
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #设置第一块GPU可见
device = torch.device("cuda")
print(device)

num_experts = 4
model_weight_path = 'weights/PAMIX_BUS-BRA.pth'
model = PAMIX(num4class=2, num4expert=num_experts).to(device)
model.load_state_dict(torch.load(model_weight_path))
model.eval()

diagnosed_image_path = 'examples/0009--0009-r-0_malignant/USImage.png'
diagnosed_mask_path = 'examples/0009--0009-r-0_malignant/Mask.png'

I = Image.open(diagnosed_image_path).convert('RGB')
M = Image.open(diagnosed_mask_path).convert('1')

x0_p, y0_p, x0_d, y0_d = M.getbbox()
W, H = M.size
wp, hp = x0_p, y0_p
wd, hd = W - x0_d, H - y0_d

_img_scale_matrix = []
for iidx in range(num_experts):
    x1_p, y1_p, x1_d, y1_d = np.uint16(x0_p - wp / (num_experts - 1) * iidx), \
        np.uint16(y0_p - hp / (num_experts - 1) * iidx), np.uint16(x0_d + wd / (num_experts - 1) * iidx), \
        np.uint16(y0_d + hd / (num_experts - 1) * iidx)
    IMG_S = I.crop((x1_p, y1_p, x1_d, y1_d)).resize((224, 224), Image.BICUBIC)
    IMG_S = transforms.ToTensor()(IMG_S)
    _img_scale_matrix.append(IMG_S)
_img_scale_matrix = torch.stack(_img_scale_matrix, dim=0)
_img_scale_matrix = torch.unsqueeze(_img_scale_matrix, dim=0).to(device)

with torch.set_grad_enabled(False):
    OUT, _, _ = model(_img_scale_matrix)
OUT = torch.nn.functional.softmax(OUT, 1)
Prediction = torch.argmax(OUT, 1).detach().cpu().numpy()[0]
Probability = OUT.detach().cpu().numpy()[0,-1]

class_space = {0: 'Malignant',1: 'Benign'}
print('Prediction:{} {}'.format(class_space[Prediction], Prediction))
print('Probability:{}'.format(Probability))