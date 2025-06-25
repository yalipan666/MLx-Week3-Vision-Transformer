import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from prepare_dataset import Combine


# Set random seed for reproducibility
torch.manual_seed(42)
# Define the neural network architecture
class ViT(nn.Module):
    def __init__(self,img_width,img_channels,patch_size,embed_dim,num_heads,num_layers,num_classes,ff_dim,drop_rate):
        super().__init__() #call the parent class's __init__
        # get the CLS which "summerize" the information of the whole sequence
        # the use of nn.Parameter here will garantee a correct backpop and update parameters
        self.cls = nn.Parameter(torch.randn(1,1,embed_dim))
        # get the embedding layer
        self.emb = nn.Linear(img_channels*patch_size*patch_size,embed_dim) #will do the broadcast to the data tensor, so the input and output dim don't need to be matched
        # get the positional encoding [including the patch_embeddings plus 1--the cls token]
        self.pos = nn.Embedding((img_width//patch_size)*(img_width//patch_size)+1,embed_dim)
        # only register the parameters
        self.register_buffer('rng',torch.arange((img_width//patch_size)*(img_width//patch_size)+1))
        # build the encoder, first define the encoder_layer, then just stack them together num_layers times
        # the use of mudulelist here will garantee a correct backpop and update parameters
        self.enc = nn.ModuleList([EncoderLayer(embed_dim,drop_rate) for _ in range(num_layers)])
        # define the finnal output layer in the model
        self.fin = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    # define the forward module (i.e., the information flow)
    def forward(self,x):
        # flatten the patch matrix into a vector, also stack together all patches
        b, c, nh, nw, ph, pw = x.shape  #[64,1,4,4,7,7]
        # stack the patches together
        x = x.reshape(b, nh*nw, ph, pw) #[64,16,7,7]
        # flatte each patch into one vector
        x = x.reshape(b, nh*nw, ph*pw)  #[64, 16, 49]
        
        # each flatten patch will be embedded to the embed_dim
        pch = self.emb(x)

        # pre-pend the CLS ("secretory") token in front of the embedding
        cls = self.cls.expand(b,-1,-1) #CLS token for the whole batch
        hdn = torch.cat([cls, pch], dim=1)

        # add the position embeddings, which are learnable parameters
        hdn = hdn + self.pos(self.rng)

        # go into the transformer attention blocks
        for enc in self.enc: hdn = enc(hdn)

        # only select the hidden vector of the CLS token for making the prediction
        out = hdn[:,0,:]

        # go throught the finnal fc layer for the classification task
        return self.fin(out)

class EncoderLayer(nn.Module):
    def __init__(self,dim,drop_rate):
        super().__init__()
        self.att = Attention(dim,drop_rate)
        self.ffn = FFN(dim,drop_rate)
        self.ini = nn.LayerNorm(dim)
        self.fin = nn.LayerNorm(dim)

    def forward(self,src):
        # the skip connections in the residual block (residual:the diff between the input and the output-->the delta)
        out = self.att(src)
        src = src + out
        src = self.ini(src)
        out = self.ffn(src)
        src = src + out
        src = self.fin(src)
        return src


class FFN(nn.Module):
    def __init__(self,dim,drop_rate):
        super().__init__()
        self.one = nn.Linear(dim, dim)
        self.drp = nn.Dropout(drop_rate)
        self.rlu = nn.GELU()
        # self.rlu = nn.ReLU(inplace=True)
        self.two = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.one(x)
        x = self.rlu(x)
        x = self.drp(x)
        x = self.two(x)
        return x


class Attention(nn.Module):
    def __init__(self,dim,drop_rate):
        super().__init__()
        self.dim = dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.drpout = nn.Dropout(drop_rate)

    def forward(self,x):
        qry = self.q_proj(x)
        key = self.k_proj(x)
        val = self.v_proj(x)
        att = qry @ key.transpose(-2,-1) * self.dim ** -0.5
        att = torch.softmax(att, dim=-1)
        att = self.drpout(att)
        out = torch.matmul(att,val)
        return self.o_proj(out)


def patchify(batch_data, patch_size):
    """
    patchify the batch of images
    """
    b,c,h,w = batch_data.shape  #[batch_size,channels,height,width] 
    ph = patch_size
    pw = patch_size
    nh, nw = h//ph, w//pw

    batch_patches = torch.reshape(batch_data, (b,c,nh,ph,nw,pw))
    batch_patches = torch.permute(batch_patches,(0,1,2,4,3,5)) #[64,1,4,4,7,7]

    # flatten the pixels in each patch
    return batch_patches


def train_model(model, train_loader, criterion, optimizer, device, patch_size):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):

        # ### try to get one data exaple
        # data_iter = iter(train_loader)
        # data,target = next(data_iter)

        data, target = data.to(device), target.to(device)
        # data.shape = [64, 1, 28, 28] #tensor: [batch_size,channels,height,width]
        # target is a tensor of shape [64]
        
        # ### try to plot a random image to have a peek
        # img = data[0].cpu().squeeze()
        # plt.imshow(img, cmap='gray')
        # plt.show()
        
        # patchify the image into a grid (in order to implement vision transformer)
        data = patchify(data,patch_size) #[64,1,4,4,7,7]
        # ### try to check on the pathify
        # img_patches = patches[0,0]
        # fig,axes = plt.subplots(4,4,figsize=(7,7))
        # for i in range (4):
        #     for j in range(4):
        #         axes[i,j].imshow(img_patches[i,j],cmap='gray')
        #         axes[i,j].axis('off')
        # plt.show()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Batch: {batch_idx + 1}, Loss: {running_loss/100:.3f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0


def evaluate_model(model, test_loader, device, patch_size):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # patchify the image into a grid (in order to implement vision transformer)
            data = patchify(data,patch_size) #[64,1,4,4,7,7]
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    # set all hyper-parameters
    batch_size = 128
    lr = 3e-4
    num_epochs = 30
    img_width = 28
    img_channels = 1
    num_classes = 10
    patch_size = 7
    embed_dim = 64
    ff_dim = 2048
    num_heads = 8
    num_layers = 3
    weight_decay = 1e-4
    drop_rate = 0.1

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')



    # ### load in the original MNIST dataset, where each image contains one digit
    # # Load and preprocess data
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])
    # train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


    # ### load in the customed dataset, where each image contains 4 digits in 4 quarants
    train_dataset = Combine(train=True)
    test_dataset = Combine(train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


    # Initialize model, loss function, and optimizer
    model = ViT(
        img_width = img_width,
        img_channels = img_channels,
        patch_size = patch_size, 
        embed_dim = embed_dim,
        num_heads = num_heads,
        num_classes = num_classes,
        num_layers = num_layers,
        ff_dim = ff_dim,
        drop_rate = drop_rate
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    # Training loop
    print('Starting training...')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        train_model(model, train_loader, criterion, optimizer, device, patch_size)
        evaluate_model(model, test_loader, device, patch_size)

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print('Model saved to mnist_model.pth')

if __name__ == '__main__':
    main() 