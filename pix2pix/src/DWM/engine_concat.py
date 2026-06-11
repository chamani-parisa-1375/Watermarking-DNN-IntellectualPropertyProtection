import torch
import os
import shutil
from  tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import cv2
import torch.nn.functional as F 
from torch.autograd import Variable

class Callback:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

    def save_ckp(self,state, validation_loss, save_result):

        os.makedirs(save_result,exist_ok=True)
        os.makedirs(save_result,exist_ok=True)
        path = rf'{save_result}/checkpoint.pt'
        torch.save(state, path)
     
        if validation_loss < self.min_validation_loss:
            best = rf'{save_result}/best.pt'
            shutil.copyfile(path, best)




    def load_ckp(self,checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']
    
def calculate_nc(original_watermark, extracted_watermark):
    """
    Calculate the Normalized Correlation (NC) between the original and extracted watermarks.

    Parameters:
    original_watermark (torch.Tensor): The original watermark bits.
    extracted_watermark (torch.Tensor): The extracted watermark bits.

    Returns:
    float: The Normalized Correlation (NC).
    """
    # Ensure both watermarks are of the same size
    assert original_watermark.size() == extracted_watermark.size(), "Watermarks must be of the same size"
    
    # Flatten the tensors to ensure they are 1D
    original_watermark = original_watermark.flatten().float()
    extracted_watermark = extracted_watermark.flatten().float()
    
    # Calculate the numerator (dot product of the two vectors)
    numerator = torch.sum(original_watermark * extracted_watermark)
    
    # Calculate the denominator (product of the Euclidean norms of the vectors)
    denominator = torch.sqrt(torch.sum(original_watermark ** 2)) * torch.sqrt(torch.sum(extracted_watermark ** 2))
    
    # Calculate NC
    nc = numerator / denominator
    
    return nc

def calculate_bcr(original_watermark, extracted_watermark):


    """
    Calculate the Bit Correct Ratio (BCR) between the original and extracted watermarks.

    Parameters:
    original_watermark (torch.Tensor): The original watermark bits.
    extracted_watermark (torch.Tensor): The extracted watermark bits.

    Returns:
    float: The Bit Correct Ratio (BCR).
    """
    # Ensure both watermarks are of the same size
    assert original_watermark.size() == extracted_watermark.size(), "Watermarks must be of the same size"
    original = torch.round(original_watermark)
    extracted = torch.round(extracted_watermark)
    # Calculate the number of correctly extracted bits
    correct_bits = torch.sum(original == extracted)
    
    # Total number of bits in the watermark
    total_bits = original.numel()
    
    # Calculate BCR
    bcr = correct_bits / total_bits
    
    return bcr    

def calculate_psnr(original: torch.Tensor, compressed: torch.Tensor) -> float:
    mse = F.mse_loss(original, compressed)
    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0  # Assuming the pixel values are normalized to the range [0, 1]
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr

def save_fig(input, target,watermark, output, ext_watermark, save_result, mode, epock='',round_wm=False):
    

    plt.figure(figsize=(20,20))
    plt.subplot(2,3,1)
    input = np.transpose(input[0].cpu().numpy(),(1,2,0))
    plt.imshow(input,cmap='gray')
    plt.title('original_img')
    plt.axis('off')

    plt.subplot(2,3,2)
    target = np.transpose(target[0].cpu().numpy(),(1,2,0))
    plt.imshow(target,cmap='gray')
    plt.title('target_img')
    plt.axis('off')

    plt.subplot(2,3,3)
    if watermark.shape[1] == 1:
        watermark = watermark[0,0].cpu().numpy()
        plt.imshow(watermark,cmap='gray')
    else:
        watermark = np.transpose(watermark[0].cpu().numpy(),(1,2,0))
        plt.imshow(watermark)
    plt.title('orginal_watarmark')
    plt.axis('off')


    plt.subplot(2,3,5)
    output = np.transpose(output[0].cpu().numpy(),(1,2,0))
    plt.imshow(output,cmap='gray')
    plt.title('output')
    plt.axis('off')
    

    plt.subplot(2,3,6)
    if ext_watermark.shape[1] == 1:
        ext_watermark = ext_watermark[0,0].cpu().numpy()

        if round_wm:
            ext_watermark = np.round(ext_watermark)
        plt.imshow(ext_watermark,cmap='gray')
    else:
        ext_watermark = np.transpose(ext_watermark[0].cpu().numpy(),(1,2,0))
        plt.imshow(ext_watermark)
    plt.title('ext_watarmark')
    plt.axis('off')

    plt.savefig(f'{save_result}/{mode}_batch_ep{epock}.png')
    plt.close()

# Example usage
# Assuming input, target, output, and watermark are your tensors with appropriate shapes
# save_fig(input, target, output, watermark, "path_to_save_directory", "mode", "epoch_number")


def train_wm(epoch,
               num_epochs,
               train_dataloader,
               img_wm,
               secret_key,
               model_H: torch.nn.Module,
               optimizer_H: torch.optim.Optimizer,
               model_D: torch.nn.Module,
               optimizer_D: torch.optim.Optimizer,
               model_E: torch.nn.Module,
               optimizer_E: torch.optim.Optimizer,
               adversarial_loss,
               content_loss,
               task_loss,
               accuracy_fn,
               calculate_watermark,
               device,
               save_result,
               save_step = 10,
               round_wm=False):
    
    ## create save result folder
    save_result = f'{save_result}/train'
    if not os.path.exists(save_result):os.mkdir(save_result)

    gloss_list, dloss_list = [],[]
    wloss_list,tloss_list = [],[]
    wacc_list,tacc_list = [],[]

    model_H.train()
    model_D.train()
    model_E.train()
    train_desc = f'train:[{epoch}/{num_epochs}]'
    with tqdm(train_dataloader, desc=train_desc,
    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as train_tqdm:
        
        for i,(input_img, target_img) in enumerate(train_tqdm):
   
            # Create a batch of watermarks
            watermark_batch = img_wm.unsqueeze(0).repeat(input_img.size(0), 1, 1, 1)
            secret_key_batch = secret_key.unsqueeze(0).repeat(input_img.size(0), 1, 1, 1)
            
            input_img, target_img = input_img.to(device), target_img.to(device)
            watermark_batch,secret_key_batch = watermark_batch.to(device),secret_key_batch.to(device)

            ## create watermark full zero and random key 
            wm_zero = torch.rand_like(watermark_batch).to(device)
            secret_x = torch.rand_like(secret_key_batch).to(device)

            real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
            fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            input_h = torch.cat([input_img,watermark_batch],dim=1)
            generated_image = model_H(input_h)

            disc_inp_fake = torch.cat((input_img, generated_image), 1)
            D_fake = model_D(disc_inp_fake.detach())
            D_fake_loss = adversarial_loss(D_fake, fake_target)
 
            disc_inp_real = torch.cat((input_img, target_img), 1)                         
            output = model_D(disc_inp_real)
            D_real_loss = adversarial_loss(output,  real_target)

        
            # train with fake
            #D_fake_loss.backward()
        
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            dloss_list.append(D_total_loss.cpu().item())
        
            D_total_loss.backward()
            optimizer_D.step()
            

            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Train generator with real labels
            optimizer_H.zero_grad()
            optimizer_E.zero_grad()


            fake_gen = torch.cat((input_img, generated_image), 1)
    #         print('fake_gen:', fake_gen)
            G = model_D(fake_gen)

            ######### genrator_loss 
            gen_loss = adversarial_loss(G, real_target)
            g_task = task_loss(generated_image, target_img)
            tloss_list.append(g_task.cpu().item())
            acc_task = accuracy_fn(generated_image, target_img)
            tacc_list.append(acc_task.cpu().item())

            ### watermark loss 
            ## Wloss1 = l1loss(E(H(x),k),w)
            watermark_extract = model_E(generated_image,secret_key_batch)
            w_loss1 = content_loss(watermark_extract,watermark_batch)
            ## Wloss2 = l1loss(E(x,k),wz)
            w_loss2 = content_loss(model_E(target_img,secret_key_batch),wm_zero)
            ## Wloss3 = l1loss(E(H(x),kx),wz)
            w_loss3 = content_loss(model_E(generated_image,secret_x),wm_zero)

            w_loss = w_loss1 + w_loss2 + 0.5*w_loss3
            wloss_list.append(w_loss.cpu().item())
            w_acc = calculate_watermark(watermark_batch,watermark_extract)
            wacc_list.append(w_acc.cpu().item())

            loss_task = w_loss + g_task


            G_loss = gen_loss + (100 * loss_task)
                                         
            gloss_list.append(G_loss.cpu().item())

            acc = accuracy_fn(generated_image, target_img)
            tacc_list.append(acc.cpu().item())

            G_loss.backward()
            optimizer_H.step()
            optimizer_E.step()


            train_tqdm.set_description(
                f'{train_desc} tloss:{np.mean(tloss_list):.3f} tacc:{np.mean(tacc_list):.3f} wloss:{np.mean(wloss_list):.3f} wacc:{np.mean(wacc_list):.3f} gloss:{np.mean(gloss_list):.3f} dloss:{np.mean(dloss_list):.3f}')

            if i == len(train_tqdm) - 1 and epoch%save_step==0:
                with torch.no_grad():
                    input_img = input_img[:1]
                    input_h = torch.cat([input_img,watermark_batch[:1]],dim=1)
                    target = target_img[:1]
                    output = model_H(input_h).detach()
                    ext_watermark = model_E(output,secret_key_batch[:1])
                    watermark = watermark_batch[:1]

                    save_fig(input_img,target,watermark,output,ext_watermark,save_result,'train',epock=epoch,round_wm=round_wm)

                    

    # Calculate loss and accuracy per epoch and print out what's happening
    return np.mean(tloss_list),np.mean(tacc_list),np.mean(wloss_list),np.mean(wacc_list),np.mean(gloss_list), np.mean(dloss_list)

def val_wm(epoch,
            num_epochs,
            val_dataloader,
            img_wm,
            secret_key,
            model_H: torch.nn.Module,
            model_E: torch.nn.Module,
            content_loss,
            task_loss,
            accuracy_fn,
            calculate_watermark,
            device,
            save_result,
            save_step = 10,
            round_wm=False):

    save_result = f'{save_result}/val'
    if not os.path.exists(save_result):os.mkdir(save_result)

    wloss_list,tloss_list = [],[]
    wacc_list,tacc_list = [],[]
    # put model in eval mode
    model_H.eval() 
    model_E.eval()

    # Turn on inference context manager
    with torch.inference_mode(): 

        val_desc = f'valid:[{epoch}/{num_epochs}]'
        with tqdm(val_dataloader, desc=val_desc,
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as val_tqdm:

            for i,(input_img, target_img) in enumerate(val_tqdm):
            
                watermark_batch = img_wm.unsqueeze(0).repeat(input_img.size(0), 1, 1, 1)
                secret_key_batch = secret_key.unsqueeze(0).repeat(input_img.size(0), 1, 1, 1)
                
                input_img, target_img = input_img.to(device), target_img.to(device)
                watermark_batch,secret_key_batch = watermark_batch.to(device),secret_key_batch.to(device)
                
                wm_zero = torch.rand_like(watermark_batch).to(device)
                secret_x = torch.rand_like(secret_key_batch).to(device)

               
                # 1. Forward pass
                input_h = torch.cat([input_img,watermark_batch],dim=1)
                generated_imgs = model_H(input_h)
                
                # 2. Calculate loss and accuracy
                g_task = task_loss(generated_imgs, target_img)
                tloss_list.append(g_task.cpu().item())
                acc_task = accuracy_fn(generated_imgs, target_img)
                tacc_list.append(acc_task.cpu().item())

                ## Wloss1 = l1loss(E(H(x),k),w)
                watermark_extract = model_E(generated_imgs,secret_key_batch)
                w_loss1 = content_loss(watermark_extract,watermark_batch)
                ## Wloss2 = l1loss(E(x,k),wz)
                w_loss2 = content_loss(model_E(target_img,secret_key_batch),wm_zero)
                ## Wloss3 = l1loss(E(H(x),kx),wz)
                w_loss3 = content_loss(model_E(generated_imgs,secret_x),wm_zero)

                w_loss = w_loss1 + w_loss2 + 0.5*w_loss3
                wloss_list.append(w_loss.cpu().item())
                w_acc = calculate_watermark(watermark_batch,watermark_extract)
                wacc_list.append(w_acc.cpu().item())
                
                val_tqdm.set_description(
                    f'{val_desc} tloss:{np.mean(tloss_list):.3f} tacc:{np.mean(tacc_list):.3f} wloss:{np.mean(wloss_list):.3f} wacc:{np.mean(wacc_list):.3f}')

                if i == len(val_tqdm) - 1 and epoch%save_step==0:
                    with torch.no_grad():
                        input_img = input_img[:1]
                        input_h = torch.cat([input_img,watermark_batch[:1]],dim=1)
                        target = target_img[:1]
                        output = model_H(input_h).detach()
                        ext_watermark = model_E(output,secret_key_batch[:1])
                        watermark = watermark_batch[:1]

                    save_fig(input_img,target,watermark,output,ext_watermark,save_result,'valid',epock=epoch,round_wm=round_wm)

                
    # Calculate loss and accuracy per epoch and print out what's happening
    return np.mean(tloss_list),np.mean(tacc_list),np.mean(wloss_list),np.mean(wacc_list)






def test_wm(test_dataloader,
            img_wm,
            secret_key,
            model_H: torch.nn.Module,
            model_E: torch.nn.Module,
            content_loss,
            task_loss,
            accuracy_fn,
            calculate_watermark,
            device,
            save_result,
            round_wm=False):
    
    e_path = f'{save_result}/E/best.pt'
    h_path = f'{save_result}/H/best.pt'

    checkpoint_e = torch.load(e_path)
    model_E.load_state_dict(checkpoint_e['state_dict'])

    checkpoint_h = torch.load(h_path)
    model_H.load_state_dict(checkpoint_h['state_dict'])

    save_result = f'{save_result}/test'
    if not os.path.exists(save_result):os.mkdir(save_result)


    wloss_list,tloss_list = [],[]
    wacc_list,tacc_list = [],[]
    # put model in eval mode
    model_H.eval() 
    model_E.eval()

    # Turn on inference context manager
    with torch.inference_mode(): 

        val_desc = f'test:'
        with tqdm(test_dataloader, desc=val_desc,
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as test_tqdm:

            for i,(input_img, target_img) in enumerate(test_tqdm):
            
                watermark_batch = img_wm.unsqueeze(0).repeat(input_img.size(0), 1, 1, 1)
                secret_key_batch = secret_key.unsqueeze(0).repeat(input_img.size(0), 1, 1, 1)
                
                input_img, target_img = input_img.to(device), target_img.to(device)
                watermark_batch,secret_key_batch = watermark_batch.to(device),secret_key_batch.to(device)
                
                wm_zero = torch.rand_like(watermark_batch).to(device)
                secret_x = torch.rand_like(secret_key_batch).to(device)

               
                # 1. Forward pass
                input_h = torch.cat([input_img,watermark_batch],dim=1)
                generated_imgs = model_H(input_h)
                
                # 2. Calculate loss and accuracy
                g_task = task_loss(generated_imgs, target_img)
                tloss_list.append(g_task.cpu().item())
                acc_task = accuracy_fn(generated_imgs, target_img)
                tacc_list.append(acc_task.cpu().item())

                ## Wloss1 = l1loss(E(H(x),k),w)
                watermark_extract = model_E(generated_imgs,secret_key_batch)
                w_loss1 = content_loss(watermark_extract,watermark_batch)
                ## Wloss2 = l1loss(E(x,k),wz)
                w_loss2 = content_loss(model_E(target_img,secret_key_batch),wm_zero)
                ## Wloss3 = l1loss(E(H(x),kx),wz)
                w_loss3 = content_loss(model_E(generated_imgs,secret_x),wm_zero)

                w_loss = w_loss1 + w_loss2 + 0.5*w_loss3
                wloss_list.append(w_loss.cpu().item())
                w_acc = calculate_watermark(watermark_batch,watermark_extract)
                wacc_list.append(w_acc.cpu().item())
                
                test_tqdm.set_description(
                    f'{val_desc} tloss:{np.mean(tloss_list):.3f} tacc:{np.mean(tacc_list):.3f} wloss:{np.mean(wloss_list):.3f} wacc:{np.mean(wacc_list):.3f}')

                with torch.no_grad():
                    input_img = input_img[:1]
                    input_h = torch.cat([input_img,watermark_batch[:1]],dim=1)
                    target = target_img[:1]
                    output = model_H(input_h).detach()
                    ext_watermark = model_E(output,secret_key_batch[:1])
                    watermark = watermark_batch[:1]

                save_fig(input_img,target,watermark,output,ext_watermark,save_result,'test',round_wm=round_wm)

                
    # Calculate loss and accuracy per epoch and print out what's happening
    return np.mean(tloss_list),np.mean(tacc_list),np.mean(wloss_list),np.mean(wacc_list)