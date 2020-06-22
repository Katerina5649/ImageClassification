import torch
from torchvision.models import resnet34
#классификация для двух классов : логотип, свастика

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PulseContentModel(torch.nn.Module):
    
    def __init__(self):
        super(PulseContentModel, self).__init__()
        n_class = 2
        self.n = n_class
        self.model = resnet34(pretrained = True).float()
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.conv1 = torch.nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        
        
        self.L1 = torch.nn.Linear(64*32*32, 1)  
        self.L2 = torch.nn.Linear(64*32*32, 1)  
        
        
        
    def forward(self, x):   # берем свастика, логотип
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        
        y1 = self.conv1(x)
        y2 = self.conv2(x)        
                
        #print(y1.shape)
        #print(y2.shape)
        
        y1 = self.L1(y1.view(y1.shape[0], -1))
        y2 = self.L2(y2.view(y2.shape[0], -1))
        
        return y1, y2