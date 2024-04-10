
from tqdm import tqdm
import torch 
from sklearn.metrics import f1_score, classification_report

def train(model, criterion, optimizer, scheduler, dataloader, device):
    model.train()
    total_loss, train_correct, train_total = 0, 0, 0
    for i, (inputs, labels) in enumerate(tqdm(dataloader), 0):
        # 입력 받기 (데이터 [입력, 라벨(정답)]으로 이루어짐)
        inputs = inputs.to(device)
        labels = labels.to(device)
        #학습
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 결과 출력
        total_loss += loss.item()
        
        train_total += labels.size(0)
        train_correct += ((torch.argmax(outputs, 1)==labels)).sum().item()
        
    scheduler.step()    
    
    train_avg_loss = total_loss / len(dataloader)
    train_avg_accuracy = 100* (train_correct / train_total)
    
    return train_avg_loss, train_avg_accuracy

def validation(model, criterion, dataloader, device):
    
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total =0
        correct, f1 = 0, 0
        for i, (inputs, labels) in enumerate(tqdm(dataloader), 0):
            # 입력 받기 (데이터 [입력, 라벨(정답)]으로 이루어짐)
            x = inputs.to(device)
            labels = labels.to(device)
            
            # vaild
            outputs = model(x)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1 += f1_score(labels.cpu().numpy() , predicted.cpu().numpy(), average='macro')
        
        val_avg_loss = (total_loss / len(dataloader))
        val_avg_accuracy = 100 * (correct / total)
        val_avg_f1 = (f1 / len(dataloader))
        
    return val_avg_loss, val_avg_accuracy, val_avg_f1
    
    
def model_test(model, criterion, dataloader, device):

    model.eval()

    with torch.no_grad():

        test_loss_sum = 0
        test_correct, f1 = 0,0
        test_total = 0
        total_predicted, total_labels = [], []
        for images, labels in dataloader:

            x_test = images.to(device)
            y_test = labels.to(device)

            outputs = model(x_test)
            loss = criterion(outputs, y_test)

            test_loss_sum += loss.item()

            _, predicted = torch.max(outputs, 1)
            test_total += y_test.size(0)
            test_correct += (predicted.cpu().numpy() == labels.cpu().numpy()).sum().item()
            f1 += f1_score(labels.cpu().numpy() , predicted.cpu().numpy(), average='macro')
            total_predicted += list(predicted.cpu().numpy())
            total_labels += list(labels.cpu().numpy())

        test_avg_loss = test_loss_sum / len(dataloader)
        test_avg_accuracy = 100* (test_correct / test_total)
        test_avg_f1 = (f1 / len(dataloader))

        print('loss:', test_avg_loss)
        print('accuracy:', test_avg_accuracy)
        print('F1-score:', test_avg_f1)
        print(classification_report(total_labels, total_predicted))
        