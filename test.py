import torchvision.models
from torch import nn

model = torchvision.models.vgg16()
model.classifier[6] = nn.Linear(in_features=1024, out_features=1000, bias=True)

print(model)