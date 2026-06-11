import torch
from  tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import os 
import torchvision.utils as vutils


def generator_loss(generated_image, target_img, G, real_target,adversarial_loss,l1_loss):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    #print(gen_loss)
    return gen_total_loss

def save_fig(input_, target, output, save_result, mode, epock=''):
    

    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    input_ = np.transpose(input_[0].cpu().numpy(),(1,2,0))
    plt.imshow(input_,cmap='gray')
    plt.title('original_img')
    plt.axis('off')

    plt.subplot(1,3,2)
    target = np.transpose(target[0].cpu().numpy(),(1,2,0))
    plt.imshow(target,cmap='gray')
    plt.title('target_img')
    plt.axis('off')

    plt.subplot(1,3,3)

    output = np.transpose(output[0].cpu().numpy(),(1,2,0))
    plt.imshow(output,cmap='gray')
    plt.title('output')
    plt.axis('off')
    
    plt.savefig(f'{save_result}/{mode}_batch_ep{epock}.png')
    plt.close()


def train_step(epoch,
               num_epochs,
               train_dataloader,
               G_model: torch.nn.Module, 
               G_optimizer: torch.optim.Optimizer,
               D_model: torch.nn.Module, 
               D_optimizer: torch.optim.Optimizer,
               loss_fn,
               adversarial_loss,
               accuracy_fn,
               device,
               save_result,
               save_step = 10):
    ## create save result folder
    save_result = f'{save_result}/train'
    if not os.path.exists(save_result):os.mkdir(save_result)

    train_gloss_list, train_gacc_list,train_dloss_list = [], [],[]

    G_model.train()
    train_desc = f'train:[{epoch}/{num_epochs}]'
    with tqdm(train_dataloader, desc=train_desc,
    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as train_tqdm:
        
        for i,(input_img, target_img) in enumerate(train_tqdm):

            
            # Send data to GPU
    
            input_img, target_img = input_img.to(device), target_img.to(device)

            D_optimizer.zero_grad()

            generated_image = G_model(input_img)
        
            disc_inp_fake = torch.cat((input_img, generated_image), 1)
       
        
            
            real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
            fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))


            D_fake = D_model(disc_inp_fake.detach())
            
    #         print("D_fake:",D_fake.shape)
            D_fake_loss   =  adversarial_loss(D_fake, fake_target)
            # print(discriminator(real_images))
            #D_real_loss.backward()
        
            disc_inp_real = torch.cat((input_img, target_img), 1)
            
                                            
            output = D_model(disc_inp_real)
            D_real_loss = adversarial_loss(output,  real_target)

        
            # train with fake
            #D_fake_loss.backward()
        
            D_total_loss = (D_real_loss + D_fake_loss) / 2
            train_dloss_list.append(D_total_loss.cpu().item())
        
            D_total_loss.backward()
            D_optimizer.step()
            
            
            # Train generator with real labels
            G_optimizer.zero_grad()
            fake_gen = torch.cat((input_img, generated_image), 1)
    #         print('fake_gen:', fake_gen)
            G = D_model(fake_gen)
            G_loss = generator_loss(generated_image, target_img, G, real_target,adversarial_loss,loss_fn)                                 
            train_gloss_list.append(G_loss.cpu().item())

            acc = accuracy_fn(generated_image, target_img)
            train_gacc_list.append(acc.cpu().item())

            G_loss.backward()
            G_optimizer.step()
        
            

            train_tqdm.set_description(
                f'{train_desc} gloss:{np.mean(train_gloss_list):.3f}  dloss:{np.mean(train_dloss_list):.3f}  acc:{np.mean(train_gacc_list):.3f}')

            if i == len(train_tqdm) - 1 and epoch%save_step==0:
                with torch.no_grad():
                    input_img = input_img[:1]
                    output = G_model(input_img).detach()

                    save_fig(input_img,target_img,output,save_result,'train',epock=epoch)

                    

    # Calculate loss and accuracy per epoch and print out what's happening
    return np.mean(train_gloss_list),np.mean(train_dloss_list), np.mean(train_gacc_list)


def val_step(epoch,
            num_epochs,
            model: torch.nn.Module,
            val_dataloader,
            loss_fn,
            accuracy_fn,
            device,
            save_result,
            save_step=10):
    
    save_result = f'{save_result}/val'
    if not os.path.exists(save_result):os.mkdir(save_result)

    
    val_loss_list, val_acc_list = [], []
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 

        val_desc = f'valid:[{epoch}/{num_epochs}]'
        with tqdm(val_dataloader, desc=val_desc,
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as val_tqdm:

            for i,(input_img, target_img) in enumerate(val_tqdm):
            
                input_img, target_img = input_img.to(device), target_img.to(device)
                
                # 1. Forward pass
                test_pred = model(input_img)
                
                # 2. Calculate loss and accuracy
                v_loss = loss_fn(test_pred, target_img)
                v_acc = accuracy_fn(test_pred, target_img)

                val_loss_list.append(v_loss.cpu().item())
                val_acc_list.append(v_acc.cpu().item())
                
                val_tqdm.set_description(
                    f'{val_desc} loss:{np.mean(val_loss_list):.3f} acc{np.mean(val_acc_list):.3f}')

                if i == len(val_tqdm) - 1 and epoch%save_step==0:
                    with torch.no_grad():
                        input_img = input_img[:1]
                        output = model(input_img).detach()

                        save_fig(input_img,target_img,output,save_result,'valid',epock=epoch)
                
    return np.mean(val_loss_list),np.mean(val_acc_list)





def test_step(model: torch.nn.Module,
            test_dataloader,
            loss_fn,
            accuracy_fn,
            device,
            save_result):
    
    save_result = f'{save_result}/test'
    if not os.path.exists(save_result):os.mkdir(save_result)

    
    test_loss_list, test_acc_list = [], []
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 

        test_desc = f'test:'
        with tqdm(test_dataloader, desc=test_desc,
                  bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as test_tqdm:

            for i,(input_img, target_img) in enumerate(test_tqdm):
            
                input_img, target_img = input_img.to(device), target_img.to(device)
                

                # 1. Forward pass
                test_pred = model(input_img)
                
                # 2. Calculate loss and accuracy
                t_loss = loss_fn(test_pred, target_img)
                t_acc = accuracy_fn(test_pred, target_img)
                
                test_loss_list.append(t_loss.cpu().item())
                test_acc_list.append(t_acc.cpu().item())
                
                test_tqdm.set_description(
                    f'{test_desc} loss:{np.mean(test_loss_list):.3f} acc{np.mean(test_acc_list):.3f}')

                if i == len(test_tqdm) - 1:
                    with torch.no_grad():
                        input_img = input_img[:1]
                        output = (model(input_img).detach())
                        save_fig(input_img,target_img,output,save_result,'test')

                
    return np.mean(test_loss_list),np.mean(test_acc_list)