import torch
from src.model_unet import *
from src.DWM.data_setup import get_train_loader,get_test_loader,get_watermark_img
from src.DWM.engine import *
from src.DWM.deep_watermark import *
from torch import optim
import pandas as pd 
import os
import random

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

runs = 'runs2'
dataset_path = rf'datasets\new_noise_dataset2'

batch_size = 16
imgsize=(256, 256)
join_img = True
num_epochs = 150
patience = 10
convert = 'RGB'
model_path = rf''
resume = False
lr = 0.001
####WATERMARK #################
img_path = rf'datasets\watermark_img\Spring-flowers-clipart-free-clipart-images.png'
wm_size = (32, 32)
convert_wm = 'RGB'
wm_channel = 3
round_wm = False
save_step = 1
use_pretrained = True
freez_layer = []
watermark_acc = calculate_psnr
watermark_loss = nn.MSELoss()
pretrained_E_path = rf'runs2\watermark\flower_32_rgb\E\best.pt'
pretrained_H_path = rf'runs2\simple\new_noise_dataset2\best.pt'
result_path = f'flower_rgb_{wm_size[0]}_new'

if resume:
    split = model_path.split('/')
    save_result = split[0]+'/'+split[1]+'/'+split[2]
else:    
    save_result = os.path.join(runs,'watermark_pretrained',result_path)
    temp = save_result
    i= 1
    while True:
        if not os.path.exists(temp):
            os.makedirs(temp,exist_ok=True)
            save_result = temp
            break
        
        i +=1
        temp = save_result + str(i)


    print(f'result_save_in: {save_result}')

#### dataloader
train_path = dataset_path + rf'/train'
val_path = dataset_path + rf'/val'
test_path = dataset_path + rf'/test'
test_flag = False
train_dataloader = get_train_loader(train_path,
                                    batch_size=batch_size,
                                    imgsize=imgsize,
                                    convert=convert)

val_path = val_path if os.path.exists(val_path) else test_path
val_dataloader = get_test_loader(val_path,
                                    batch_size=batch_size,
                                    imgsize=imgsize,
                                    convert=convert)


if os.path.exists(test_path) and not (val_path == test_path):
    test_dataloader = get_test_loader(test_path,
                                    batch_size=batch_size,
                                    imgsize=imgsize,
                                    convert=convert)

    test_flag = True

print(f'train_batch: {len(train_dataloader)} val_batch: {len(val_dataloader)} test_batch: {len(test_dataloader)}')

img_wm = get_watermark_img(img_path,
                           wm_size,
                           convert=convert_wm,
                           round=round_wm)


def main():
    call_back = Callback(patience=patience)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    
    start_epock = 1
    epock_list = []
    train_gloss_plot,train_dloss_plot = [],[]
    train_wloss_plot,train_tloss_plot = [],[]
    train_tacc_plot,train_wacc_plot = [],[]

    val_wloss_plot,val_tloss_plot = [],[]
    val_tacc_plot,val_wacc_plot = [],[]

    # model_H = UNet(n_channel=1,n_class=1).to(device)
    model_H = UnetGenerator(input_nc=3,output_nc=3).to(device)
    secret_key = torch.ones((1,)+wm_size)
    model_D = Discriminator(6, 64, n_layers=3).to(device)
    model_E = Extraction(wh=wm_size,
                         input_Eb=1,
                         output_Eb=1,
                         WH=imgsize,
                         input_Ea=3,
                         output_nc=wm_channel,
                         input_inception=8).to(device)
    
    model_D.apply(init_weights)
    model_E.apply(init_weights)
    

    

    optimizer_H = optim.Adam(model_H.parameters(), lr = lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(model_D.parameters(), lr = lr, betas=(0.5, 0.999))
    optimizer_E = optim.Adam(model_E.parameters(), lr = lr, betas=(0.5, 0.999))
    
    

    adversarial_loss = nn.BCELoss()
    # content_loss = nn.L1Loss()

    task_fn_loss = nn.L1Loss()

    if use_pretrained:

        model_E,optimizer_E = use_E_weights(pretrained_E_path,
                                        model_E,
                                        optimizer_E,freez_layer)
        
        model_H,optimizer_H = use_H_weights(pretrained_H_path,
                                        model_H,
                                        optimizer_H)
        

    if resume:

        model_H,optimizer_H,start_epock = call_back.load_ckp(save_result+'/H/checkpoint.pt',
                                                            model_H,
                                                            optimizer_H)
        
        model_E,optimizer_E,start_epock = call_back.load_ckp(save_result+'/E/checkpoint.pt',
                                                            model_E,
                                                            optimizer_E)
        
        model_D,optimizer_D,start_epock = call_back.load_ckp(save_result+'/D/checkpoint.pt',
                                                            model_D,
                                                            optimizer_D)

       
        start_epock +=1
        print(f'resume training from epock={start_epock}')

        if os.path.exists(rf'{save_result}/result.csv'):

            df = pd.read_csv(rf'{save_result}/result.csv')
            epock_list = list(df['epocks'])
            train_tloss_plot = list(df['train_task_loss'])
            train_tacc_plot = list(df['train_task_acc']) 
            train_wloss_plot = list(df['train_watermark_loss'])
            train_wacc_plot = list(df['train_watermark_acc'])
            train_gloss_plot = list(df['train_gloss'])
            train_dloss_plot = list(df['train_dloss'])
            val_tloss_plot = list(df['valid_task_loss'])
            val_tacc_plot = list(df['valid_task_acc']) 
            val_wloss_plot = list(df['valid_watermark_loss'])
            val_wacc_plot = list(df['valid_watermark_acc'])

    for epoch in range(start_epock,num_epochs+1):

        train_tloss,train_tacc,train_wloss,train_wacc,train_gloss,train_dloss = train_wm(epoch,
                                                                                num_epochs,
                                                                                train_dataloader,
                                                                                img_wm,
                                                                                secret_key,
                                                                                model_H,
                                                                                optimizer_H,
                                                                                model_D,
                                                                                optimizer_D,
                                                                                model_E,
                                                                                optimizer_E,
                                                                                adversarial_loss,
                                                                                watermark_loss,
                                                                                task_fn_loss,
                                                                                calculate_psnr,
                                                                                watermark_acc,
                                                                                device,
                                                                                save_result,
                                                                                save_step,
                                                                                round_wm)
                                          
        train_wacc_plot.append(train_wacc)
        train_wloss_plot.append(train_wloss)
        train_tacc_plot.append(train_tacc)
        train_tloss_plot.append(train_tloss)
        train_gloss_plot.append(train_gloss)
        train_dloss_plot.append(train_dloss)


        val_tloss,val_tacc,val_wloss,val_wacc = val_wm(epoch,
                                                num_epochs,
                                                val_dataloader,
                                                img_wm,
                                                secret_key,
                                                model_H,
                                                model_E,
                                                watermark_loss,
                                                task_fn_loss,
                                                calculate_psnr,
                                                watermark_acc,
                                                device,
                                                save_result,
                                                save_step,
                                                round_wm)
                                          
        val_wacc_plot.append(val_wacc)
        val_wloss_plot.append(val_wloss)
        val_tacc_plot.append(val_tacc)
        val_tloss_plot.append(val_tloss)

        val_gloss = val_tloss + val_wloss


        state_H = {'epoch': epoch,
        'state_dict': model_H.state_dict(),
        'optimizer': optimizer_H.state_dict()}

        state_E = {'epoch': epoch,
        'state_dict': model_E.state_dict(),
        'optimizer': optimizer_E.state_dict()}

        state_D = {'epoch': epoch,
        'state_dict': model_D.state_dict(),
        'optimizer': optimizer_D.state_dict()}
        
        call_back.save_ckp(state_H,val_gloss,save_result+'/H')
        call_back.save_ckp(state_E,val_gloss,save_result+'/E')
        call_back.save_ckp(state_D,val_gloss,save_result+'/D')

        epock_list.append(epoch)
        df = pd.DataFrame({"epocks":epock_list,
                           'train_task_loss':train_tloss_plot,
                           'train_task_acc':train_tacc_plot,
                           'train_watermark_loss':train_wloss_plot,
                           'train_watermark_acc':train_wacc_plot,
                           'train_gloss':train_gloss_plot,
                           'train_dloss':train_dloss_plot,
                           'valid_task_loss':val_tloss_plot,
                           'valid_task_acc':val_tacc_plot,
                           'valid_watermark_loss':val_wloss_plot,
                           'valid_watermark_acc':val_wacc_plot
                           })
        df.to_csv(rf'{save_result}/result.csv',index=False)

        
        if call_back.early_stop(val_gloss):

            print(f'early stopping in [{epoch}\{num_epochs}]')
            break

   

    plt.plot(epock_list,train_tloss_plot)
    plt.plot(epock_list,train_wloss_plot)
    plt.plot(epock_list,train_gloss_plot)
    plt.plot(epock_list,train_dloss_plot)
    plt.xlabel('epocks')
    plt.ylabel('losses')
    plt.legend(('task_loss','watermark_loss','g_loss','d_loss'))
    plt.title(('Train losses'))
    plt.savefig(rf'{save_result}\train_loss.png')
    plt.close()

    plt.plot(epock_list,train_tloss_plot)
    plt.plot(epock_list,train_wloss_plot)
    plt.plot(epock_list,val_tloss_plot)
    plt.plot(epock_list,val_wloss_plot)
    plt.xlabel('epocks')
    plt.ylabel('losses')
    plt.legend(('train_task_loss','train_watermark_loss','valid_task_loss','valid_watermark_loss'))
    plt.title(('Train and valid losses'))
    plt.savefig(rf'{save_result}\losses.png')
    plt.close()


    plt.plot(epock_list,train_tacc_plot)
    plt.plot(epock_list,train_wacc_plot)
    plt.plot(epock_list,val_tacc_plot)
    plt.plot(epock_list,val_wacc_plot)
    plt.xlabel('epocks')
    plt.ylabel('accuracy')
    plt.legend(('train_task_acc','train_watermark_acc','valid_task_acc','valid_watermark_acc'))
    plt.title(('Train and valid accuracy'))
    plt.savefig(rf'{save_result}\accuracy.png')
    plt.close()

    if test_flag:

        test_tloss,test_tacc,test_wloss,test_wacc = test_wm(test_dataloader,
                                                    img_wm,
                                                    secret_key,
                                                    model_H,
                                                    model_E,
                                                    watermark_loss,
                                                    task_fn_loss,
                                                    calculate_psnr,
                                                    watermark_acc,
                                                    device,
                                                    save_result,
                                                    round_wm)
        
        with open(rf'{save_result}\result_test.txt','w') as f:

            f.write(f'test_tloss:{test_tloss}\n')
            f.write(f'test_tacc:{test_tacc}\n')
            f.write(f'test_wloss:{test_wloss}\n')
            f.write(f'test_wacc:{test_wacc}\n')



if __name__ == '__main__':
    main()
