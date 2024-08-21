"""
    PointNet
"""

import os
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.misc_utils import set_seed
from utils.data_utils import PointNetModelNet40
from utils.network_utils import EarlyStopping

class TransformNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        """
            T-Net class learns transformation matrices.
        
            input_dim: dimension of the input points (default 3 for 3D points).
            output_dim: dimension of the output transformation matrix (default 3 for 3x3).
        """
        super(TransformNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(128, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(1024, momentum=0.5)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim * output_dim)
        self.bn4 = nn.BatchNorm1d(512, momentum=0.5)
        self.bn5 = nn.BatchNorm1d(256, momentum=0.5)
        
        # Initialize the weights for the final fc layer to produce an identity matrix
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(output_dim).view(-1))
        
    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        x = x.view(-1, self.output_dim, self.output_dim)

        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=True):
        """
            PointNet feature extraction module.
        
            global_feat: whether to return only the global feature.
            feature_transform: whether to include feature transformation.
        """
        super(PointNetfeat, self).__init__()
        
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        
        self.input_transform_net = TransformNet(input_dim=3, output_dim=3)
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(64, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(128, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(1024, momentum=0.5)
        
        if self.feature_transform:
            self.feature_transform_net = TransformNet(input_dim=64, output_dim=64)
    
    def forward(self, x):
        batch_size, num_points, _ = x.size()
        
        x = x.transpose(1, 2)
        trans = self.input_transform_net(x)
        x = x.transpose(1, 2)
        x = torch.bmm(x, trans)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if self.feature_transform:
            trans_feat = self.feature_transform_net(x)
            x = x.transpose(1, 2)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(1, 2)
        else:
            trans_feat = None
        
        point_feat = x
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(batch_size, 1024, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_feat], 1), trans, trans_feat
    
    def feature_transform_regularizer(self, trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :]
        if trans.is_cuda:
            I = I.cuda()
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
        return loss

class PointNetCls(PointNetfeat):
    def __init__(self, num_classes=40, feature_transform=True):
        """
            PointNet classification network.
        
            num_classes: number of output classes.
            feature_transform: whether to include feature transformation.
        """
        super(PointNetCls, self).__init__(global_feat=True, feature_transform=feature_transform)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(p=0.3)
        self.cls_bn1 = nn.BatchNorm1d(512, momentum=0.5)
        self.cls_bn2 = nn.BatchNorm1d(256, momentum=0.5)
        self.relu = nn.ReLU()

        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x, trans, trans_feat = super(PointNetCls, self).forward(x)
        
        x = F.relu(self.cls_bn1(self.fc1(x)))
        x = F.relu(self.cls_bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        
        return x, trans, trans_feat
        # return F.log_softmax(x, dim=1), trans, trans_feat

    def compute_loss(self, outputs, labels, trans_feat):
        class_loss = self.criterion(outputs, labels)
        if self.feature_transform:
            reg_loss = self.feature_transform_regularizer(trans_feat)
            total_loss = class_loss + 0.001 * reg_loss
        else:
            total_loss = class_loss
        return total_loss

def adjust_bn_momentum(model, epoch, max_epoch, initial_momentum=0.5, max_momentum=0.99):
    momentum = initial_momentum + (max_momentum - initial_momentum) * (epoch / max_epoch)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = momentum
    print(f"BatchNorm momentum adjusted to {momentum:.4f} for epoch {epoch+1}")

def plot_loss_curve(log_file_path, save_path=None):
    epochs = []
    losses = []

    with open(log_file_path, 'r') as log_file:
        next(log_file)  # Skip the header line
        for line in log_file:
            epoch, loss, lr, bn_momentum = line.strip().split(',')
            epochs.append(int(epoch))
            losses.append(float(loss))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
    
    plt.show()

if __name__ == '__main__':
    set_seed()
    dataset_name = 'modelnet40'
    
    if dataset_name == 'modelnet40':
        categories = ['cup', 'flower_pot']
        train_dataset = PointNetModelNet40(categories=categories, show_progress=True)
        test_dataset = PointNetModelNet40(categories=categories, split='test', show_progress=True)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PointNetCls(num_classes=len(categories), feature_transform=True)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'logs/{dataset_name}/')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    loss_file_path = os.path.join(log_dir, 'loss.txt')
    with open(loss_file_path, 'w') as log_file:
        log_file.write("Epoch, Loss, LR, BN_Momentum\n")

    best_model_path = os.path.join(log_dir, 'best_model.pth')

    num_epochs = 100
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        with tqdm(train_dataloader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}") as tepoch:
            for data in tepoch:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs, trans, trans_feat = model(inputs)
                loss = model.compute_loss(outputs, labels, trans_feat)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                tepoch.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        with open(loss_file_path, 'a') as log_file:
            log_file.write(f"{epoch+1},{epoch_loss:.4f},{scheduler.get_last_lr()[0]:.6f},{model.bn1.momentum:.4f}\n")

        scheduler.step()
        
        adjust_bn_momentum(model=model, epoch=epoch, max_epoch=num_epochs)

        if epoch % 5 == 0:
            # Evaluation step
            model.eval()
            correct = 0
            total = 0
            eval_loss = 0.0

            with torch.no_grad():
                for data in test_dataloader:    # Use test set here.
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs, trans, trans_feat = model(inputs)
                    loss = model.compute_loss(outputs, labels, trans_feat)
                    eval_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            eval_loss /= len(test_dataloader)
            print(f"Validation Loss: {eval_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Save the best model based on evaluation loss
            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")

            early_stopping(eval_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    # Final model save
    model_save_path = os.path.join(log_dir, 'model_final.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    loss_curve_path = os.path.join(log_dir, 'loss_curve.png')
    plot_loss_curve(loss_file_path, save_path=loss_curve_path)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs, trans, trans_feat = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")