import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Double convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        r"""Double 3x3 Same Convolution block with Batch Normalization and ReLU activation."""
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
class UNet(nn.Module):
    def __init__( # Constructor, set default parameters
            self, in_channels=3, # RGB image input
            out_channels=1, # binary mask output
            features=[64, 128, 256, 512], 
    ):
        super(UNet, self).__init__() #Initialize the parent class nn.Module

        self.ups = nn.ModuleList() # List of upsampling layers
        self.downs = nn.ModuleList() # List of downsampling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling layer

        # Down part of the UNet (basically: convolution -> pooling for each feature size) 4 steps of convolution + downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) # Double Convolution Maps the 3 features of RGB to 64 (First value in features)
            in_channels = feature # Overwrite in_channels with the output size of the previous layer

        # Bottom of the U
        # Pass trough double conv and double the last feature size
        self.bottleneck = DoubleConv(features[-1], features[-1]*2) 

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

        # Final output layer after up-ladder
        # 1x1 convolution just maps the last feature size to the output channels
        self.finalConv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # BUILD THE UNET
    def forward(self, x):
        skip_connections = []

        # DOWN LADDER

        # For each down step, in this case 4 steps
        for down in self.downs:
            x = down(x) # Apply DoubleConv (blue arrows)
            skip_connections.append(x) # Save the output for the skip connection (grey arrow)
            x = self.pool(x) # Maxpooling downsample (red arrow)
        
        # BOTTOM

        # Bottleneck (bottom of the U)
        x = self.bottleneck(x) # Just a DoubleConv, no up or downsampling and no skip connection

        # UP LADDER

        # Reverse Skip connections (order from botto to top)
        skip_connections = skip_connections[::-1] # reverse list order

        # Iterate on the up steps with step size 2
        #   In each up, there is 2 items:
        #       1. Upsampling (ConvTranspose2d)
        #       2. DoubleConv (which concatenates the skip connection)
        for idx in range(0, len(self.ups), 2):

            # Upsampling (ConvTranspose2d)
            x = self.ups[idx](x) 

            # Iterate with step 1 on the skip connections
            skip_connection = skip_connections[idx // 2] # Get the corresponding skip connection

            # Protection against input sizes not divisible by 16
            #   If input is n not divisible by 16, (by 2 4 times),
            #   Maxpool will flatten the image (eg. 161x161 to 80x80)
            #   Upsampling will double the size (eg. 80x80 to 160x160)
            #   But the skip_connection will still be 161x161
            if x.shape != skip_connection.shape:
                # Resize x to match the original shape (other options woud be adding padding or cropping, would be useful to compare performance between the three)
                #   tensor.shape is an array of [batch_size, channels, height, width]
                #   [2:] will grab all items from index 2 (height) to the end 
                x = TF.resize(x, size=skip_connection.shape[2:]) 

            # Concatenate along the channel dimension
            concatInput = torch.cat((skip_connection, x), dim=1) 

            # DoubleConv
            x = self.ups[idx + 1](concatInput)
        
        # FINAL OUTPUT LAYER
        return self.finalConv(x)

# DEBUG
def test():
    # Testing edge cases
    x = torch.randn((3, 2, 479, 521))  # Batch size of 3, 2 channels, 479x521 image
    model = UNet(in_channels=2, out_channels=1) # Default parameters: in_channels=3, out_channels=1, features=[64, 128, 256, 512]

    preds = model(x)  # Forward pass (Because Unet is a subclass of nn.Module, this calls the forward() method)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")  # Should be [3, 1, 479, 521] if out_channels=1

if __name__ == "__main__":
    test()