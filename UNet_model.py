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
        
# UNet model class
# Main class, defines the architecture except the skip connections
class UNet(nn.Module):
    def __init__(
            self, in_channels=3, # RGB image input
            out_channels=1, # binary mask output
            features=[64, 128, 256, 512], #
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList() # List of upsampling layers
        self.downs = nn.ModuleList() # List of downsampling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling layer

        # Down part of the UNet (basically: convolution -> pooling for each feature size) 4 steps of convolution + downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # Double Convolution Maps the 3 features of RGB to 64 (First value in features)
            in_channels = feature # Overwrite in_channels with the output size of the previous layer

        # Up Part:
        for feature in reversed(features): # go through the features in reverse, from big to small

            # Upsampling: Transpose convolution (deconvolution) to double the size
            self.ups.append(
                # ConvTranspose2d with kernel 2 and stride 2  is the standard "learnable upsampling" layer, it doubles the size of the image
                nn.ConvTranspose2d(
                    feature*2, # *2 because the previous layer has 2x the number of features 
                    feature, # halves the number of features
                    kernel_size=2,
                    stride=2
                )
            )

            ## Concatenate

            # pass through the double conv
            self.ups.append(DoubleConv(feature*2, feature))
        
        # Bottom of the U
        # Pass trough double conv and double the last feature size
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) 

        # Final output layer after up-ladder
        # 1x1 convolution just maps the last feature size to the output channels
        self.finalConv = nn.Conv2d(features[0], out_channels, kernel_size=1)

