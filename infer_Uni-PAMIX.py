import os
import torch
import warnings
from PIL import Image
from torchvision import transforms
from models.PAMIX_Family import UniPAMIX

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICES_ORDER"]="PCI_BUS_IS" #设备排序
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #设置第一块GPU可见
device = torch.device("cuda")

model_weight_path = 'weights/Uni-PAMIX_BUS-BRA.pth'
model = UniPAMIX(num4class=2).to(device)
model.load_state_dict(torch.load(model_weight_path))
model.eval()

diagnosed_image_path = 'examples/0009--0009-r-0_malignant/USImage.png'

IMG = Image.open(diagnosed_image_path).convert('RGB')
IMG = IMG.resize((224, 224), Image.BICUBIC)
IMG = transforms.ToTensor()(IMG)
IMG = torch.unsqueeze(IMG, dim=0).to(device)

with torch.set_grad_enabled(False):
    OUT, _ = model(IMG)
OUT = torch.nn.functional.softmax(OUT, 1)
Prediction = torch.argmax(OUT, 1).detach().cpu().numpy()[0]
Probability = OUT.detach().cpu().numpy()[0,-1]

class_space = {0: 'Malignant',1: 'Benign'}
print('Prediction:{} {}'.format(class_space[Prediction], Prediction))
print('Probability:{}'.format(Probability))