import torch
from src.model_unet import *
from src.engine_unet import *
from src.DWM.data_setup import get_train_loader,get_test_loader
from src.DWM.engine import Callback,calculate_psnr
from torch import optim
import pandas as pd 
import os
import matplotlib.pyplot as plt
runs = 'runs3'
dataset_path = rf'datasets/draw2pic'
result_path = 'draw2pic__results'
batch_size = 32
imgsize=(256, 256)
join_img = True
num_epochs = 150
lr = 0.001
beta1 = 0.9
patience = 10
convert = 'RGB'
save_step = 1
resume = False
model_path = rf''


if resume:
    split = model_path.split('/')
    save_result = split[0]+'/'+split[1]+'/'+split[2]
else:    
    save_result = os.path.join(runs,'simple',result_path)
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


if os.path.exists(test_path) and not(val_path == test_path):
    test_dataloader = get_test_loader(test_path,
                                    batch_size=batch_size,
                                    imgsize=imgsize,
                                    convert=convert)

    test_flag = True

print(f'train_batch: {len(train_dataloader)} val_batch: {len(val_dataloader)}')

def main():
    call_back = Callback(patience=patience)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'v
    
    start_epock = 1
    epock_list = []
    train_glosses_plot,train_accuracy_plot,train_dlosses_plot = [],[],[]
    valid_losses_plot,valid_accuracy_plot = [],[]

    G_model = UnetGenerator(input_nc=3,output_nc=3).to(device)
    D_model= Discriminator(6, 64, n_layers=3).to(device)
    # unet_model = UNet(n_channel=1,n_class=1).to(device)
    criterion = torch.nn.L1Loss()  # For binary segmentation
    # G_optimizer = torch.optim.Adam(G_model.parameters(), lr=0.001)
    G_optimizer = optim.Adam(G_model.parameters(), lr = lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D_model.parameters(), lr = lr, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss() 

 
    if resume:

        G_model,G_optimizer,start_epock = call_back.load_ckp(rf"runs\chest-segmentation\checkpoint.pt",
                           model=G_model,optimizer=G_optimizer)
        start_epock +=1
        print(f'resume training from epock={start_epock}')

        if os.path.exists(rf'{save_result}/result.csv'):

            df = pd.read_csv(rf'{save_result}/result.csv')
            epock_list = list(df['epocks'])
            train_glosses_plot = list(df['train_gloss'])
            train_dlosses_plot = list(df['train_dloss'])
            train_accuracy_plot = list(df['train_acc']) 
            valid_losses_plot = list(df['valid_loss'])
            valid_accuracy_plot = list(df['valid_acc'])
    
    for epoch in range(start_epock,num_epochs+1):

        train_gloss,train_dloss,train_acc = train_step(epoch=epoch,
                                          num_epochs=num_epochs,
                                          train_dataloader=train_dataloader,
                                          G_model=G_model,    
                                          G_optimizer=G_optimizer,
                                          D_model=D_model,
                                          D_optimizer=D_optimizer,
                                          loss_fn=criterion,
                                          adversarial_loss=adversarial_loss,
                                          accuracy_fn=calculate_psnr,
                                          device=device,
                                          save_result=save_result,
                                          save_step=save_step)
        
        train_glosses_plot.append(train_gloss)
        train_dlosses_plot.append(train_dloss)
        train_accuracy_plot.append(train_acc)

        val_loss,val_acc = val_step(epoch=epoch,
                                    num_epochs=num_epochs,
                                    model=G_model,
                                    val_dataloader=val_dataloader,
                                    loss_fn=criterion,
                                    accuracy_fn=calculate_psnr,
                                    device=device,
                                    save_result=save_result,
                                    save_step=save_step)
        
        valid_losses_plot.append(val_loss)
        valid_accuracy_plot.append(val_acc)

        state = {'epoch': epoch,
        'state_dict': G_model.state_dict(),
        'optimizer': G_optimizer.state_dict()}
        
        call_back.save_ckp(state,val_loss,save_result)
        epock_list.append(epoch)
        df = pd.DataFrame({"epocks":epock_list,
                           'train_gloss':train_glosses_plot,
                           'train_dloss':train_dlosses_plot,
                           'train_acc':train_accuracy_plot,
                           'valid_loss':valid_losses_plot,
                           'valid_acc':valid_accuracy_plot })
        df.to_csv(rf'{save_result}/result.csv',index=False)

        
        if call_back.early_stop(val_loss):

            print(f'early stopping in [{epoch}\{num_epochs}]')
            break

    x = np.arange(len(valid_losses_plot))+1
    plt.plot(x,train_glosses_plot)
    plt.plot(x,train_dlosses_plot)
    plt.plot(x,valid_losses_plot)
    plt.xlabel('epocks')
    plt.ylabel('losses')
    plt.legend(('train_gloss','train_dloss','val_loss'))
    plt.savefig(rf'{save_result}\loss.png')
    plt.close()

    plt.plot(x,train_accuracy_plot)
    plt.plot(x,valid_accuracy_plot)
    plt.xlabel('epocks')
    plt.ylabel('accuracy')
    plt.legend(('train_accuracy','val_accuracy'))
    plt.savefig(rf'{save_result}\accuracy.png')
    plt.close()
     

    if test_flag:

        test_loss,test_acc = test_step(model=G_model,
                  test_dataloader=test_dataloader,
                  loss_fn=criterion,
                  accuracy_fn=calculate_psnr,
                  device=device,
                  save_result=save_result)
        
        with open(rf'{save_result}\result_test.txt','w') as f:

            f.write(f'test_loss:{test_loss}\n')
            f.write(f'test_acc:{test_acc}\n')



        
       


if __name__ == '__main__':
    main()
