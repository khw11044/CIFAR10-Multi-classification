import matplotlib.pyplot as plt 



def draw_training_result(train_loss_list, train_acc_list, train_epochs, val_loss_list, val_acc_list, vaild_epochs)  :
    plt.figure(figsize=(10,6))
    
    plt.subplot(1, 2, 1)  
    plt.plot(train_epochs, train_loss_list, label='train_err', marker = '.')
    plt.plot(vaild_epochs, val_loss_list, label='val_err', marker = '.')
    
    plt.subplot(1, 2, 2)  
    plt.plot(train_epochs, train_acc_list, label='train_accuracy', marker = '.')
    plt.plot(vaild_epochs, val_acc_list, label='valaccuracy', marker = '.')
    
    plt.grid()
    plt.legend()
    plt.savefig('./train_result')
    
    #plt.show()