import os

import torch
import torch.nn as nn
import torch.optim as optim
import easydict 

from Networks.model1 import BasicNet, MyCNNNet, TransModel
from process import train, validation, model_test
from visualization import draw_training_result
from dataload.cifa import get_cifa10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_BATCH_SIZE = 4
LR = 0.001

args = easydict.EasyDict({
       'num_epochs':15,
       'num_epochs_ae':50,
       'lr':1e-3,
       'lr_ae':1e-3,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'BATCH_SIZE':1024,
       'TEST_BATCH_SIZE':4,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':0,
       'save_path':'./weights/best_model.pth'
                })

if __name__=="__main__":
    weight_root = './weights'
    os.makedirs(weight_root, exist_ok=True)
    dataloader_train, dataloader_vaild, dataloader_test, classes = get_cifa10(args)
    epoch_start = 0
    
    # model = MyCNNNet(num_classes=len(classes)).to(device)
    model = TransModel(num_classes = len(classes)).to(device)
    if args.pretrain:
        state_dict = torch.load(args.save_path)
        model.load_state_dict(state_dict['net_dict'])
        epoch_start = state_dict['epoch']
        print('start epoch:', epoch_start)
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    
    train_epochs, val_epochs = [], []
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = [], [], [], []

    loss_score = 1000
    for epoch in range(epoch_start, args.num_epochs):

        train_loss, train_acc = train(model, criterion, optimizer, scheduler, dataloader_train, device) 
        print('Model training... Epoch: {}, Loss: {:.3f}, Accuracy: {:.3f}'.format(epoch, train_loss, train_acc))

        if (epoch+1)%5==0:
            valid_loss, val_acc, val_f1 = validation(model, criterion, dataloader_vaild, device)
            print('Model vaildation ... Loss: {:.3f}, Accuracy: {:.3f}, F1: {:.3f}'.format(valid_loss, val_acc, val_f1))
            if loss_score>valid_loss:
                loss_score = valid_loss
                model = model.cpu()
                print('---'*20)
                print('lowest valid loss: {:.3f}, best valid acc: {:.3f} And SAVE model'.format(valid_loss, val_acc))
                torch.save({'epoch': epoch,
                    'net_dict': model.state_dict()}, args.save_path)
                
                model.to(device)
                
            val_loss_list.append(valid_loss)
            val_epochs.append(epoch)
            val_acc_list.append(val_acc)
            
        train_loss_list.append(train_loss)
        train_epochs.append(epoch)
        train_acc_list.append(train_acc)
    
    draw_training_result(train_loss_list, train_acc_list, train_epochs, val_loss_list, val_acc_list, val_epochs)           
    print('Finished Training')
    print()
    print('Test')
    model_test(model, criterion, dataloader_test, device)
