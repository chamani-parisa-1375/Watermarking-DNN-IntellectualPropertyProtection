import torch
from src.model_unet import UNet,UnetGenerator
from src.engine_unet import *
from src.DWM.data_setup import get_train_loader,get_test_loader
from src.DWM.engine import Callback
from torch import optim
import pandas as pd 
import os
runs = 'runs'
dataset_path = rf'datasets\flood_area_segmnet'

batch_size = 32
imgsize=(256, 256)
join_img = True
num_epochs = 20
lr = 0.001
beta1 = 0.9
patience = 10
convert = 'L'
save_step = 1
number_of_freezlayer = 1
resume = False
model_path = rf'runs/watermark_retrain/qr_128_new'

def fine_tune_model(model,nlayer=3):

    checkpoint = torch.load(model_path+'/H/best.pt')
    model.load_state_dict(checkpoint['state_dict'])

    """
    Freezes all layers of a given module except the last n layers.
    """
    # Collect all the parameter names and freeze all of them initially
    layer_names = []
    for name, param in model.named_parameters():
        layer_names.append(name)
        param.requires_grad = False  # Freeze all layers initially

    # Track number of layers with 'weight' in their name to identify trainable layers
    i = 0
    trainable_layers = []

    # Go through the layers in reverse and unfreeze the last `n` weight layers
    for name in reversed(layer_names):
        if 'weight' in name:
            i += 1
        if i > nlayer:
            break  # Stop once we've unfrozen `n` layers
        trainable_layers.append(name)

    # Unfreeze the identified layers
    for name, param in model.named_parameters():
        if name in trainable_layers:
            param.requires_grad = True

        # Count trainable and frozen parameters for verification
    trainable_count = 0
    frozen_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_count += 1
        else:
            frozen_count += 1
    print(f"Trainable layers: {trainable_count}, Frozen layers: {frozen_count}")       
            # Update optimizer for parameters that require gradients
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    return model,optimizer

        

def accuracy_fn(y_pred,Y):
    temp = torch.round(y_pred)
    acc = 1 - torch.mean(input=temp.to(torch.int) ^ Y.to(torch.int),dtype=torch.float)
    
    return  acc
split = model_path.split('/')
if resume:
    
    save_result = split[0]+'/'+split[1]+'/'+split[2]
else:    
    save_result = os.path.join(runs,'fine_tuen',split[-1])
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
    train_losses_plot,train_accuracy_plot = [],[]
    valid_losses_plot,valid_accuracy_plot = [],[]

    unet_model = UnetGenerator(input_nc=1,output_nc=1).to(device)
    # unet_model = UNet(n_channel=1,n_class=1).to(device)
    criterion = torch.nn.BCELoss()  # For binary segmentation
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.001)


    unet_model,optimizer = fine_tune_model(unet_model,number_of_freezlayer)
 
    if resume:

        unet_model,optimizer,start_epock = call_back.load_ckp(rf"runs\chest-segmentation\checkpoint.pt",
                           model=unet_model,optimizer=optimizer)
        start_epock +=1
        print(f'resume training from epock={start_epock}')

        if os.path.exists(rf'{save_result}/result.csv'):

            df = pd.read_csv(rf'{save_result}/result.csv')
            epock_list = list(df['epocks'])
            train_losses_plot = list(df['train_loss'])
            train_accuracy_plot = list(df['train_acc']) 
            valid_losses_plot = list(df['valid_loss'])
            valid_accuracy_plot = list(df['valid_acc'])
    
    for epoch in range(start_epock,num_epochs+1):

        train_loss,train_acc = train_step(epoch=epoch,
                                          num_epochs=num_epochs,
                                          model=unet_model,
                                          train_dataloader=train_dataloader,
                                          optimizer=optimizer,
                                          loss_fn=criterion,
                                          accuracy_fn=accuracy_fn,
                                          device=device,
                                          save_result=save_result,
                                          save_step=save_step)
        
        train_losses_plot.append(train_loss)
        train_accuracy_plot.append(train_acc)

        val_loss,val_acc = val_step(epoch=epoch,
                                    num_epochs=num_epochs,
                                    model=unet_model,
                                    val_dataloader=val_dataloader,
                                    loss_fn=criterion,
                                    accuracy_fn=accuracy_fn,
                                    device=device,
                                    save_result=save_result,
                                    save_step=save_step)
        
        valid_losses_plot.append(val_loss)
        valid_accuracy_plot.append(val_acc)

        state = {'epoch': epoch,
        'state_dict': unet_model.state_dict(),
        'optimizer': optimizer.state_dict()}
        
        call_back.save_ckp(state,val_loss,save_result)
        epock_list.append(epoch)
        df = pd.DataFrame({"epocks":epock_list,'train_loss':train_losses_plot,'train_acc':train_accuracy_plot,'valid_loss':valid_losses_plot,'valid_acc':valid_accuracy_plot })
        df.to_csv(rf'{save_result}/result.csv',index=False)

        
        if call_back.early_stop(val_loss):

            print(f'early stopping in [{epoch}\{num_epochs}]')
            break

    x = np.arange(len(valid_losses_plot))+1
    plt.plot(x,train_losses_plot)
    plt.plot(x,valid_losses_plot)
    plt.xlabel('epocks')
    plt.ylabel('losses')
    plt.legend(('train_loss','val_loss'))
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

        test_loss,test_acc = test_step(model=unet_model,
                  test_dataloader=test_dataloader,
                  loss_fn=criterion,
                  accuracy_fn=accuracy_fn,
                  device=device,
                  save_result=save_result)
        
        with open(rf'{save_result}\result_test.txt','w') as f:

            f.write(f'test_loss:{test_loss}\n')
            f.write(f'test_acc:{test_acc}\n')


if __name__ == '__main__':
    main()
