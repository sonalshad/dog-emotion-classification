import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from transformers import ViTModel

# import necessary libraries

class ResNet(nn.Module):

    def __init__(self, version='resnet50'):
        super(ResNet, self).__init__()
        
        if version =='resnet34': 
            hidden_dim = 512
            resnet = models.resnet34(weights='DEFAULT') #IMAGENET1K_V2 currently
        else: 
            hidden_dim = 2048   
            resnet = models.resnet50(weights='DEFAULT')
        
        # freezing parameters
        for param in resnet.parameters():
            param.requires_grad = False

        # convolutional layers of resnet
        layers = list(resnet.children())[:8]
        self.top_model = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, 2) #to see if it works with cross-entropy
        # self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.top_model(x))
        x = nn.AdaptiveAvgPool2d((1,1))(x)
        x = x.view(x.size(0), -1) # flattening
        x = self.fc(x)
        return x

class DINOv2(nn.Module):

    ''' This class adds a linear layer to the pre-trained DINOv2 model for
        classification. 
        
        Input should be pre-processed images.
        
        ## USAGE ##
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        input = processor(images=image, return_tensors="pt")
        model = DINOv2WithLinearHead()
        model.forward(input['pixel_values'])                    
    '''
    def __init__(self, num_labels=2):
        
        super().__init__()
    
        # Base Model and Linear Layer
        ## Double check that weights are frozen
        self.dino_model = ViTModel.from_pretrained("facebook/dino-vits16", 
                          add_pooling_layer=False)
        for param in self.dino_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.dino_model.config.hidden_size, num_labels)

    def forward(self, input_pixels):
        outputs = self.dino_model(input_pixels) # this layer requires 'pixel_values' as input
        pooled_output = outputs.last_hidden_state[:, 0] 
        logits = self.classifier(pooled_output)
        return logits


    