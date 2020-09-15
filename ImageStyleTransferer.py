from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
import re
import os


class ImageStyleTransferer():

    def __init__(self,LayerParameters, MaxImageSize, IterationNumber, ShowEvery, LearningRate, BetaParameter=1e6):

        self.VGG = models.vgg19(pretrained=True).features

        for param in self.VGG.parameters():
            param.requires_grad_(False)

        self.__device = torch.device("cpu")

        self.__layerParameters = LayerParameters

        self.__maxImageSize = MaxImageSize

        self.__iterationNumber = IterationNumber

        self.__showEvery = ShowEvery

        self.__learningRate = LearningRate

        self.__beta = BetaParameter

        self.__layers = {'0': 'conv1_1',
                        '5': 'conv2_1', 
                        '10': 'conv3_1', 
                        '19': 'conv4_1',
                        '21': 'conv4_2',  
                        '28': 'conv5_1'}

    # private metgods
    
    def __loadImage(self,img_path,shape=None):

        image = Image.open(img_path).convert('RGB')
    
        if max(image.size) > self.__maxImageSize:
            size = self.__maxImageSize
            image = image.resize((self.__maxImageSize,self.__maxImageSize))
        else:
            size = max(image.size)

        if shape is not None:
            size = shape
        
        in_transform = transforms.Compose([transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                 (0.229, 0.224, 0.225))])

        image = in_transform(image)[:3,:,:].unsqueeze(0)
    
        return image

    def __imageConvert(self,tensor):
    
        if self.__device.type != "cpu":
            image = tensor.to("cpu")
        else:
            image = tensor

        image = image.clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image

    def __getFeatures(self ,image):
        
        features = {}

        x = image

        for name, layer in self.VGG._modules.items():
            x = layer(x)
            if name in self.__layers:
                features[self.__layers[name]] = x
            
        return features

    def __gramMatrix(self,tensor):

        _, d, h, w = tensor.size()
    
        tensor = tensor.view(d, h * w)
    
        gram = torch.mm(tensor, tensor.t())
    
        return gram

    # set methods

    def SetDevice(self, Device):
        if self.__device != Device:
            self.__device = Device
            self.VGG.to(self.__device)

    def SetLayerParamenters(self, Parameters):
        self.__layerParameters = Parameters

    def SetMaximumImageSize(self, Size):
        self.__maxImageSize = Size

    def SetIterationNumber(self, Iterations):
        self.__iterationNumber = Iterations

    def SetShowEveryValue(self, ShowEvery):
        self.__showEvery = ShowEvery

    def SetLearningRate(self,LearningRate):
        self.__learningRate = LearningRate

    def SetBeta(self, BetaValue):
        self.__beta = BetaValue

    # process method
    def Process(self,ContentImagePath,StyleImagePath,ImageSavingFolder="Result\\"):

        if not os.path.exists(ImageSavingFolder):
            os.makedirs(ImageSavingFolder)

        formatRE = re.compile("\.[\w|\d]{2,4}")

        contentImageName = formatRE.sub("",ContentImagePath.split("\\")[-1])
        styleImageName = formatRE.sub("",StyleImagePath.split("\\")[-1])

        content = self.__loadImage(ContentImagePath)
        style = self.__loadImage(StyleImagePath, shape=content.shape[-2:])
        target = content.clone()

        if self.__device.type != "cpu":
            content = content.to(self.__device)
            style = style.to(self.__device)
            target = target.to(self.__device)

        target = target.requires_grad_(True)

        content_features = self.__getFeatures(content)
        style_features = self.__getFeatures(style)

        style_grams = {layer: self.__gramMatrix(style_features[layer]) for layer in style_features}

        style_weights = {'conv1_1': self.__layerParameters[0],
                         'conv2_1': self.__layerParameters[1],
                         'conv3_1': self.__layerParameters[2],
                         'conv4_1': self.__layerParameters[3],
                         'conv5_1': self.__layerParameters[4]}

        content_weight = 1
        style_weight = self.__beta

        if self.__showEvery == 0:
            self.__showEvery = self.__iterationNumber

        optimizer = optim.Adam([target], lr=self.__learningRate)

        for i in range(1,self.__iterationNumber + 1):
            tar__getFeatures = self.__getFeatures(target)
   
            content_loss = torch.mean((tar__getFeatures['conv4_2'] - content_features['conv4_2']) ** 2)
    
            style_loss = 0

            for layer in style_weights:
                target_feature = tar__getFeatures[layer]
                target_gram = self.__gramMatrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

                style_loss += layer_style_loss / (d * h * w)

            total_loss = content_weight * content_loss + style_weight * style_loss
    
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print(f"Iteration : {i} Total loss: {total_loss.item()}")

            if  i % self.__showEvery == 0:
                
                fig = plt.figure(frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(self.__imageConvert(target), aspect='auto')
                fig.savefig(f"{ImageSavingFolder}{contentImageName}_{styleImageName}_styled_{i}.png")
                plt.close(fig)

