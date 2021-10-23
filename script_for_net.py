import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

model = torch.load("path to model", map_location=torch.device('cpu'))
model.eval()
model.cuda()
print(os.getcwd())
def is_it():
    val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def image_loader(image_name):
        image = Image.open(image_name)
        image = val_transforms(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image.cuda()  #

    image = image_loader("dataset/sample.jpg")
    classes = ("flower", "not flower")
    pred = model(image)
    prediction = int(torch.argmax(pred, dim=1))
    return classes[prediction]



