import torch
from torch import nn
import math


class InceptionResidualBlock(nn.Module):
    def __init__(self, input_nc=16):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc,
                      out_channels=input_nc,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc,
                      out_channels=input_nc,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_nc,
                      out_channels=input_nc,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=input_nc,
                      out_channels=input_nc,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_nc,
                      out_channels=input_nc,
                      kernel_size=5,
                      stride=1,
                      padding=2),

            nn.ReLU(),
        )

        self.conv_concat = nn.Sequential(
            nn.Conv2d(in_channels=input_nc * 3,
                      out_channels=input_nc,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x)
        x3 = self.conv_block_3(x)

        x_cat = torch.cat((x1, x2, x3), dim=1)

        x_4 = self.conv_concat(x_cat)

        y = torch.add(x, x_4)

        return y
    
class  Extraction(nn.Module):

    def __init__(self,wh,input_Eb,output_Eb,WH,input_Ea=3,output_nc=3,input_inception = 8) -> None:
        super().__init__()

        self.w,self.h = wh
        W,H = WH
        self.c1 = math.floor((input_Ea*H*W)/(self.h*self.w))


        self.EA = nn.Sequential(
            nn.Conv2d(in_channels=input_Ea,
            out_channels= input_inception,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            InceptionResidualBlock(input_nc=input_inception),
            InceptionResidualBlock(input_nc=input_inception),
            nn.Conv2d(in_channels=input_inception,
            out_channels= input_Ea,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),

        )

        self.EB = nn.Sequential(
            nn.Conv2d(in_channels=input_Eb,
            out_channels= 12,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12,
            out_channels= output_Eb,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
        )

        self.EC = nn.Sequential(
            nn.Conv2d(in_channels=(self.c1+output_Eb),
            out_channels= 48,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=48,
            out_channels= 24,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24,
            out_channels= output_nc,
            kernel_size=3,
            stride=1,
            padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x,secret_key):

        a = self.EA(x)
        # Resizing a using torch.nn.functional.interpolate
        a = torch.reshape(a,(a.shape[0],self.c1,self.h,self.w))

        b = self.EB(secret_key)
 
        x_cat = torch.cat((a,b), dim=1)
 
        y = self.EC(x_cat)

        return y
    


class D_Watermark(nn.Module):
    def __init__(self, in_channels=1):
        super(D_Watermark, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
    

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)





def use_weights(checkpoint_fpath, model, optimizer, freez_layer=False):
    checkpoint = torch.load(checkpoint_fpath)
    pretrained_dict = checkpoint['state_dict']

    # Load weights into the model
    model_dict = model.state_dict()
    
    # Filter weights that exist in the model and have matching dimensions
    new_pretrained_dict = {}
    equal_model = True
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_pretrained_dict[k] = v
        else:
            new_pretrained_dict[k] = model_dict[k]
            equal_model = False

    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)

    # if equal_model:
    #     optimizer.load_state_dict(checkpoint['optimizer'])

    if len(freez_layer):
        # Freeze all layers
        # for param in model.parameters():
        #     param.requires_grad = False

        # # Unfreeze the first and last layers
        # layers = list(model.children())
        # if layers:
        #     # Ensure there's something to optimize by unfreezing the first and last layers
        #     if len(layers) > 1:
        #         for param in layers[0].parameters():
        #             param.requires_grad = True
        #         for param in layers[-1].parameters():
        #             param.requires_grad = True
        #     else:
        #         # If there's only one layer, unfreeze it
        #         for param in layers[0].parameters():
        #             param.requires_grad = True


        for param in model.parameters():
            param.requires_grad = False

        if 'ec' in freez_layer:
            # اگر می‌خواهید لایه‌های خاصی را آزاد بگذارید:
            for param in model.EC.parameters():
                param.requires_grad = True

        if 'eb' in freez_layer:
            # اگر می‌خواهید لایه‌های خاصی را آزاد بگذارید:
            for param in model.EB.parameters():
                param.requires_grad = True


        if 'ea' in freez_layer:
            # اگر می‌خواهید لایه‌های خاصی را آزاد بگذارید:
            for param in model.EA.parameters():
                param.requires_grad = True

        # Update optimizer for parameters that require gradients
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    return model, optimizer



def use_H_weights(checkpoint_fpath, model, optimizer):

    checkpoint = torch.load(checkpoint_fpath)
    pretrained_dict = checkpoint['state_dict']

    # Load weights into the model
    model_dict = model.state_dict() 
    
    new_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_pretrained_dict[k] = v
        else:
            new_pretrained_dict[k] = model_dict[k]

    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)


        # Collect all the parameter names and freeze all of them initially
    layer_names = []
    for name, param in model.named_parameters():
        layer_names.append(name)
        param.requires_grad = False  # Freeze all layers initially

    # Track number of layers with 'weight' in their name to identify trainable layers

    nlayer = int(len(layer_names)*2/3)
    trainable_layers = [ layer for i,layer in enumerate(layer_names)
                         if  i<nlayer]
   
    # Go through the layers in reverse and unfreeze the last `n` weight layers



    # Unfreeze the identified layers
    for name, param in model.named_parameters():
        if name in trainable_layers:
            param.requires_grad = True

    return model,optimizer