import os 
import numpy as np 
import cv2
from PIL import Image 

import matplotlib.pyplot as plt 

import torch 
import torchvision


# Creating a custom dataset class 
class ImageDataset(torch.utils.data.Dataset): 
    def __init__(self, dir, transform=None,convert='RGB'): 
        self.data_dir = dir
        self.images = os.listdir(dir) 
        self.transform = transform 
        self.convert = convert
  
    # Defining the length of the dataset 
    def __len__(self): 
        return len(self.images) 
  
    # Defining the method to get an item from the dataset 
    def __getitem__(self, index): 
        image_path = os.path.join(self.data_dir, self.images[index]) 

        image = Image.open(image_path)

        if self.convert == 'RGB':
            image = image.convert('RGB')
        else:
            image = image.convert('L')

        image = np.array(image)


        # Applying the transform 
        if self.transform: 
            image = self.transform(image) 
          
        return image
    

# Defining a custom transformer class 
class CustomTransform(object): 
    def __init__(self, imgsize=(256,256),a2b=True): 
        self.a2b = a2b 
        self.imgsize = imgsize
      
    # Defining the transform method 
    def __call__(self, image): 
        
        # Splitting the image into two parts 
        split = int(image.shape[1] * 0.5) 
        
        image1 = cv2.resize(image[:, :split],self.imgsize) 
        image2 = cv2.resize(image[:, split:],self.imgsize) 

        
          
        # Returning the two parts of the image 
        if self.a2b:
            return image1, image2
        else:
            return image2, image1
        
class CustomTensor(object):

    def __init__(self,tozero=True) -> None:
        self.tozero = tozero

    def __call__(self, image):
        A,B = image

        

        ## transpose for model
        
        if len(A.shape) == 3:
            A = np.transpose(A, (2, 0, 1)).astype(np.float32)
            B = np.transpose(B, (2, 0, 1)).astype(np.float32)

        else:
            A = np.expand_dims(A, axis=0).astype(np.float32)
            B = np.expand_dims(B, axis=0).astype(np.float32)
        
        
        ## normalize image between [-1 1]
        if self.tozero:
            A = A / 255
            B = B / 255
        else:
            A = A / 127.5 -1
            B = B / 127.5 -1

        ## convert to tensor
        A = torch.tensor(A, dtype=torch.float32)
        B = torch.tensor(B, dtype=torch.float32)

        return A,B

# Defining a custom augmentation class 
class CustomFlip(object): 
    def __init__(self, flip_prob=0.5): 
        self.flip_prob = flip_prob 


    # Defining the transform method 
    def __call__(self, image):
        A,B = image
    

        # Flipping the image horizontally 
        if np.random.random() < self.flip_prob: 
            A = A[:,::-1].copy()
            B = B[:,::-1].copy()

        # Returning the augmented image 
        return A,B
     

class CustomScale(object):
     
    def __init__(self,prob=0.2):
        self.prob = prob

    def __call__(self, image):
        A,B = image

        height, width = A.shape[:2]
        h,w = int(height*self.prob)+height,int(width*self.prob)+width

        A = cv2.resize(A,(h,w))
        B = cv2.resize(B,(h,w))

        x, y = np.random.uniform(low=0,high=int(h-height)), np.random.uniform(low=0,high=int(w-width))  
        # return image[:, int(x):int(x)+256, int(y):int(y)+256]

        A = A[int(x):int(x)+height, int(y):int(y)+width]
        B = B[int(x):int(x)+height, int(y):int(y)+width]



        return A,B
    

class CustomJitter(object):
    
    def __init__(self,jitter_prob=0.5): 

        self.jitter_prob = jitter_prob 

    def __call__(self, image):
        
        A,B = image
        if np.random.random() < self.jitter_prob: 
             
            noise = np.random.randint(-10, 10, size=A.shape, dtype=np.int8)
            # Adding random noise to the image 
            A = A + noise 

        return A,B





def get_train_loader(data_path,
                     imgsize,
                     batch_size,
                     train_transform=None,
                     num_workers=None,
                     a2b = True,
                     tozero=True,
                     convert='RGB'):

    if num_workers is None:
        num_workers = os.cpu_count()

    if train_transform is None:
        train_transform = torchvision.transforms.Compose([ 
        CustomTransform(imgsize=imgsize,a2b=True), 
        # CustomFlip(),
        # CustomScale(), 
        # CustomJitter(),
        CustomTensor(tozero),

        ]) 

  
    
    train_dataset = ImageDataset(data_path,
                                transform=train_transform,
                                convert=convert)

    # Creating the train and test dataloaders 
    train_dataloader = torch.utils.data.DataLoader( 
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        
    ) 


    return train_dataloader


def get_test_loader(data_path,
                    imgsize,
                    batch_size,
                    num_workers=None,
                    a2b = True,
                    tozero=True,
                    convert='RGB'):

    if num_workers is None:
        num_workers = os.cpu_count()

    test_transform = torchvision.transforms.Compose([ 
        CustomTransform(imgsize=imgsize,a2b=a2b),
        CustomTensor(tozero) 
        
    ]) 

    # Creating the train and test datasets 

    test_dataset = ImageDataset(data_path,
                                transform=test_transform,
                                convert=convert) 

    test_dataloader = torch.utils.data.DataLoader( 
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        
    ) 

    return test_dataloader



def get_watermark_img(img_path,
                    imgsize,
                    convert='L',
                    tozero=True,
                    round=True):
    
    image = Image.open(img_path)
    image = image.convert(convert)

    image = np.array(image)
    image = cv2.resize(image,imgsize)

    if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    else:
            image = np.expand_dims(image, axis=0).astype(np.float32)

        
        
    ## normalize image between [-1 1]
    if tozero:
        image = image / 255
    else:
        image = image / 127.5 -1

    if round:
        image = np.round(image)


    return torch.tensor(image, dtype=torch.float32)


    
