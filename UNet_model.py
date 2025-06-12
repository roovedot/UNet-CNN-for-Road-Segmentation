import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        # Double convolution block
        self.conv = nn.Sequential(

            # 3x3 same convolution layer (output size will be the same as input size)
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=3, # "window" (filter) size    
                      stride=1, # pixels moved by the window in each step 
                      padding=1, # empty pixels around the image
                      bias=False, # no bias since BatchNorm would cancel it out
                      ),
            nn.BatchNorm2d(out_channels), # Normalizes the output of the convolution
            nn.ReLU(inplace=True), # Activation function (linear growth from 0, all negative is 0). inplace=True overwrites the tensor

            # Repeat, just take out_channels (output from the previous layer) as input
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        
        )#self.conv

        # Method that applies the layers in the sequential block
        def forward(self, x):
            return self.conv(x)  # Apply the convolutional layers to the input tensor x