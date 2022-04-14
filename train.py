import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torchvision import transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import argparse
from misc import progress_bar
import os.path
import sys

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vggcfg = {
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        }
        self.features = self._make_layers(self.vggcfg['VGG16'])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

class Solver(object):
    def __init__(self, config):
        self.model = None
        self.model_name = config.model
        self.lr = config.lr
        self.epochs = config.epoch
        self.batch_size = config.batch
        self.cuda = config.cuda
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.train_loader = None
        self.val_loader = None
        self.feasible_model = ['CNN', 'GoogleNet', 'AlexNet', 'VGGNet']
        self.eval_losses = []
        self.eval_accu = []
        self.train_accu = []
        self.train_losses = []

    def load_data(self):
        transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = self.batch_size

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

        validation_portion = 0.3
        num_of_validation = int(len(trainset) * validation_portion)
        num_of_train = int(len(trainset) - num_of_validation)

        trainset, val_set = torch.utils.data.random_split(trainset, [num_of_train, num_of_validation], generator=torch.Generator().manual_seed(0))
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

        self.val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

        print("Loading Data...")
        print("Train set size: ", len(trainset))
        print("Validation set size: ", len(val_set))
        
    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        if self.model_name == "GoogleNet":
            self.model = GoogLeNet().to(self.device)
        
        if self.model_name == "CNN":
            self.model = Net().to(self.device)
        
        if self.model_name == "AlexNet":
            self.model = AlexNet().to(self.device)
        
        if self.model_name == "VGGNet":
            self.model = VGG().to(self.device)

        elif self.model_name not in self.feasible_model:
            print("Please type correct model name")
            sys.exit(0)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 7, 10, 15, 18], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        train_loss = 0
        train_correct = 0
        total = 0
        self.model.train(True)
        
        for batch_num, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)
            # train_correct incremented by one if predicted right
            train_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

            progress_bar(batch_num, len(self.train_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_num + 1), 100. * train_correct / total, train_correct, total))

        return train_loss / (batch_num + 1), train_correct / total

    def val(self):
        print("val:")
        self.model.eval()
        val_loss = 0
        val_correct = 0
        total = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                val_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.val_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (val_loss / (batch_num + 1), 100. * val_correct / total, val_correct, total))
        return val_loss / (batch_num + 1), val_correct / total

    def test(self, model_path):
        correct = 0
        total = 0
        model = torch.load(model_path)
        model.eval()
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                # calculate outputs by running images through the network
                outputs = model(data)
                # the class with the highest energy is what we choose as prediction
                prediction = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())
            
            if model_path == 'tmp.pt':
                print(f'New Validation Accuracy: {100 * correct // total} %')
            else:
                print(f'Existing Validation Accuracy: {100 * correct // total} %')
            return 100 * correct / total
    
    def save_acc_history(self):
        plt.figure()
        plt.plot(self.train_accu, '-o')
        plt.plot(self.eval_accu, '-o')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Accuracy')
        plt.xticks(np.arange(0, self.epochs, step=3))
        plt.savefig("graph/" + self.model_name + "_acc.png")

    def save_loss_history(self):
        plt.figure()
        plt.plot(self.train_losses, '-o')
        plt.plot(self.eval_losses, '-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train','Valid'])
        plt.title('Train vs Valid Losses')
        plt.xticks(np.arange(0, self.epochs, step=3))
        plt.savefig("graph/" + self.model_name + "_loss.png")

    def save(self):
        model_out_path = 'weight/' + self.model_name + '.pt'
        tmp_model_path = 'tmp.pt'
        torch.save(self.model, tmp_model_path)
        existing_acc = 0.0

        if os.path.exists(model_out_path):
            existing_acc = self.test(model_out_path)

        new_test_acc = self.test(tmp_model_path)

        if new_test_acc > existing_acc:
            torch.save(self.model, model_out_path)
            self.save_acc_history()
            self.save_loss_history()
            print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        accuracy = 0
        for epoch in range(1, self.epochs + 1):
            txt = "\n===> epoch: {e}/{total}".format(e=epoch, total=self.epochs)
            print(txt)
            train_result = self.train()
            loss, accu = train_result
            self.train_losses.append(loss)
            self.train_accu.append(accu)

            val_result = self.val()
            loss, accu = val_result
            self.eval_losses.append(loss)
            self.eval_accu.append(accu)

            self.scheduler.step(epoch)
            accuracy = max(accuracy, val_result[1])

            if epoch == self.epochs:
                print("===> BEST ACC. PERFORMANCE: %.3f%%" % (accuracy * 100))
                self.save()
    
def main():
    parser = argparse.ArgumentParser(description="Various model with cifar-10 using PyTorch")
    parser.add_argument('-l', '--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-e', '--epoch', default=20, type=int, help='number of epochs tp train for')
    parser.add_argument('-b', '--batch', default=200, type=int, help='batch size')
    parser.add_argument('-m', '--model', default="GoogleNet", type=str, help='Network Type')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.load_data()
    solver.load_model()
    solver.run() 

if __name__ == "__main__":
    main()