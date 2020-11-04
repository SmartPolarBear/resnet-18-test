import torch
import torchvision
import tensorrt
from torch2trt import torch2trt

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model = model.cuda().eval().half()

model.load_state_dict(torch.load('best_model_resnet18.pth'))

device = torch.device('cuda')

data = torch.zeros((1, 3, 224, 224)).cuda().half()

model_trt = torch2trt(model, [data], fp16_mode=True)

torch.save(model_trt.state_dict(), 'best_model_trt.pth')