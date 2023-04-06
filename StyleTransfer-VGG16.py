import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
from matplotlib import pyplot as plt
from torch.autograd import Variable
import copy

transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor()])

use_gpu = torch.cuda.is_available()
cnn = models.vgg16(pretrained=True).features
if use_gpu:
    cnn = cnn.cuda()


# Define the function to load the image
def imgLoad(path=None):
    img = Image.open(path)
    img = transform(img)
    img = img.unsqueeze(0)
    return img


ToPIL = torchvision.transforms.ToPILImage()


# Define the function to show the image
def imgShow(img, title=None):
    img = img.clone().cpu()
    img = img.view(3, 224, 224)
    img = ToPIL(img)

    if title is not None:
        plt.title(title)
    plt.imshow(img)
    plt.show()


content_img = imgLoad("TestPicture/image6.jpg")
content_img = Variable(content_img).cpu()
style_img = imgLoad("TestPicture/image8.jpg")
style_img = Variable(style_img).cpu()
print(content_img.size())
print(style_img.size())

imgShow(content_img, title='Content Image')
imgShow(style_img, title='Style Image')


# define content loss function
class Content_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Content_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input):
        self.loss = self.loss_fn(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


#  define style loss function
class Style_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()
        self.gram = gram_matrix()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.loss_fn(self.G, self.target)
        return self.output

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


# define the gram_matrix
class gram_matrix(torch.nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        return gram.div(a * b * c * d)


content_layer = ["Conv_5", "Conv_6"]
style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4", "Conv_5"]

content_losses = []
style_losses = []

content_weight = 1
style_weight = 1000

new_model = torch.nn.Sequential()

model = copy.deepcopy(cnn)

gram = gram_matrix()

index = 1
for layer in list(model):
    if isinstance(layer, torch.nn.Conv2d):
        name = "Conv_" + str(index)
        new_model.add_module(name, layer)
        if name in content_layer:
            target = new_model(content_img).clone()
            content_loss = Content_loss(content_weight, target)
            new_model.add_module("content_loss_" + str(index), content_loss)
            content_losses.append(content_loss)

        if name in style_layer:
            target = new_model(style_img).clone()
            target = gram(target)
            style_loss = Style_loss(style_weight, target)
            new_model.add_module("style_loss_" + str(index), style_loss)
            style_losses.append(style_loss)

    if isinstance(layer, torch.nn.ReLU):
        name = "Relu_" + str(index)
        new_model.add_module(name, layer)
        index = index + 1

    if isinstance(layer, torch.nn.MaxPool2d):
        name = "MaxPool_" + str(index)
        new_model.add_module(name, layer)

print(new_model)

input_img = content_img.clone()
parameter = torch.nn.Parameter(input_img.data)
optimizer = torch.optim.LBFGS([parameter])

n_epoch = 20

run = [0]


def closure():
    optimizer.zero_grad()
    style_score = 0
    content_score = 0
    parameter.data.clamp_(0, 1)
    new_model(parameter)
    for sl in style_losses:
        style_score += sl.backward()

    for cl in content_losses:
        content_score += cl.backward()

    run[0] += 1
    if run[0] % 10 == 0:
        print('{} Style Loss : {:4f} Content Loss: {:4f}'.format(run[0], style_score.item(), content_score.item()))

    return style_score + content_score


# Begin our transfer Process
while run[0] <= n_epoch / 10:
    optimizer.step(closure)

parameter.data.clamp_(0, 1)
plt.figure()
imgShow(parameter.data, title="Output Image")
