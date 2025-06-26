import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from prepare_dataset import Combine
from dataclasses import dataclass


# Set random seed for reproducibility
torch.manual_seed(42)

# pre-set all the relevant parameters
@dataclass
class TrainingHyperparameters:
    batch_size: int = 128
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    drop_rate: float = 0.1

@dataclass
class ModelHyperparameters:
    img_width: int = 56
    img_channels: int = 1
    num_classes: int = 10
    patch_size: int = 14
    embed_dim: int = 64
    ff_dim: int = 2048
    num_heads: int = 8
    num_layers: int = 3
    n_digit: int = 4

# Define the encoder architecture
class ViT(nn.Module):
    def __init__(self, model_cfg: ModelHyperparameters, train_cfg: TrainingHyperparameters):
        super().__init__() #call the parent class's __init__
        ##### building the encoder
        # get the CLS which "summerize" the information of the whole sequence
        # the use of nn.Parameter here will garantee a correct backpop and update parameters
        self.cls = nn.Parameter(torch.randn(1,1,model_cfg.embed_dim))
        # get the embedding layer
        self.emb = nn.Linear(model_cfg.img_channels*model_cfg.patch_size*model_cfg.patch_size, model_cfg.embed_dim) #will do the broadcast to the data tensor, so the input and output dim don't need to be matched
        # get the positional encoding [including the patch_embeddings plus 1--the cls token]
        num_patches = (model_cfg.img_width // model_cfg.patch_size) * (model_cfg.img_width // model_cfg.patch_size)
        self.pos = nn.Embedding(num_patches + 1, model_cfg.embed_dim)
        # only register the parameters
        self.register_buffer('rng', torch.arange(num_patches + 1))
        # build the encoder, first define the encoder_layer, then just stack them together num_layers times
        # the use of mudulelist here will garantee a correct backpop and update parameters
        self.enc = nn.ModuleList([EncoderLayer(model_cfg.embed_dim, model_cfg.num_heads, train_cfg.drop_rate) for _ in range(model_cfg.num_layers)])
        # define the finnal output layer in the model
        self.fin = nn.Sequential(
            nn.LayerNorm(model_cfg.embed_dim),
            nn.Linear(model_cfg.embed_dim, model_cfg.num_classes)
        )
    
        #### building the decoder
        # +2 for start and end tokens
        self.emb_output = nn.Embedding(model_cfg.num_classes+2, model_cfg.embed_dim)
        self.pos_output = nn.Embedding(model_cfg.n_digit+2, model_cfg.embed_dim)
        self.register_buffer('rng_output', torch.arange(model_cfg.n_digit+2))
        self.dec = nn.ModuleList([DecoderLayer(model_cfg.embed_dim, model_cfg.num_heads, train_cfg.drop_rate) for _ in range(model_cfg.num_layers)])
        self.fin_output = nn.Sequential(
            nn.LayerNorm(model_cfg.embed_dim),
            nn.Linear(model_cfg.embed_dim, model_cfg.num_classes+2)
        )
        self.n_digit = model_cfg.n_digit
        self.num_classes = model_cfg.num_classes
        self.start_token = model_cfg.num_classes
        self.end_token = model_cfg.num_classes + 1

    def forward(self, x, y):
        # flatten the patch matrix into a vector, also stack together all patches
        
        ### full-transformer
        b, np, ph, pw = x.shape  #[128,16,14,14]
        # flatte each patch into one vector
        x = x.reshape(b, np, ph*pw)  #[128, 16, 196]
        
        # each flatten patch will be embedded to the embed_dim
        pch = self.emb(x)

        # pre-pend the CLS ("secretory") token in front of the embedding
        cls = self.cls.expand(b,-1,-1) #CLS token for the whole batch
        hdn = torch.cat([cls, pch], dim=1)
        
        # add the position embeddings, which are learnable parameters
        hdn = hdn + self.pos(torch.arange(hdn.size(1), device=hdn.device)) # "broadcast" the positional encoding across the batch dimension.
        # go into the transformer attention blocks --- the encoder
        for enc in self.enc: hdn = enc(hdn)  # the key and val of the encoder goes into the decoder
        # # this section is only useful for the classification task
        # # only select the hidden vector of the CLS token for making the prediction
        # out = hdn[:,0,:]
        # # go throught the finnal fc layer for the classification task
        # out = self.fin(out)
        ##### build the decoder, using y(i.e.,target), a list of 6 elements [start,digit1,digit2,digit3,digit4,end]
    
        out_emb = self.emb_output(y) #[batch, seq_len, embed_dim]
        out_emb = out_emb + self.pos_output(self.rng_output[:out_emb.size(1)])
        tgt_mask = generate_mask(out_emb.size(1)).to(x.device)
        tgt = out_emb
        for dec in self.dec: tgt = dec(tgt, memory=hdn, tgt_mask=tgt_mask)
        # memory indicates the output from the encoder layer, which will be projected by the decoder's 
        # cross-attention layer into keys and values as needed
        
        # get the finnal prediction
        logits = self.fin_output(tgt)
        out = torch.softmax(logits, dim=-1)
        return out

    # build the generation task where the model predict the token/digit one by one
    def autoregressive_inference(self, x, device, max_digits=None): 
        # patchify and encode the image
        # x: [batch, np, ph, pw]
        b, np, ph, pw = x.shape
        x = x.reshape(b, np, ph*pw)
        pch = self.emb(x)
        cls = self.cls.expand(b,-1,-1)
        hdn = torch.cat([cls, pch], dim=1)
        hdn = hdn + self.pos(self.rng)
        for enc in self.enc: hdn = enc(hdn)
        # Initialize the output sequence, start with [start_token]
        max_steps = max_digits if max_digits is not None else self.n_digit + 2
        y = torch.full((b, 1), self.start_token, dtype=torch.long, device=device) # the "have been seen" tokens
        outputs = []
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for step in range(max_steps):
            out_emb = self.emb_output(y)
            out_emb = out_emb + self.pos_output(self.rng_output[:out_emb.size(1)])
            tgt_mask = generate_mask(out_emb.size(1)).to(x.device) # the mask is auto-adjusted to how many y you have here
            tgt = out_emb
            for dec in self.dec: # go throught the loop of the decoder layers
                tgt = dec(tgt, memory=hdn, tgt_mask=tgt_mask)
            logits = self.fin_output(tgt)  # [b, cur_seq, num_classes]
            next_token_logits = logits[:, -1, :]  # [b, num_classes]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [b, 1] Returns the indices of the maximum value of all elements in the input tensor.
            # If already finished, keep predicting end_token
            next_token[finished.unsqueeze(1)] = self.end_token
            y = torch.cat([y, next_token], dim=1)
            outputs.append(next_token)
            # Update finished mask
            finished = finished | (next_token.squeeze(1) == self.end_token)
            # If all finished, break
            if finished.all():
                break
        # Stack outputs, remove start token, and trim at end_token for each sample
        outputs = torch.cat(outputs, dim=1)  # [b, <=max_steps]
        # Remove tokens after end_token for each sample
        result = []
        for i in range(b):
            out = outputs[i]
            if (out == self.end_token).any():
                idx = (out == self.end_token).nonzero(as_tuple=True)[0][0]
                result.append(out[:idx].cpu())
            else:
                result.append(out.cpu())
        # Pad to max length in batch
        maxlen = max([r.size(0) for r in result])
        result_padded = torch.full((b, maxlen), self.end_token, dtype=torch.long) #long: only integers no decimals
        for i, r in enumerate(result):
            result_padded[i, :r.size(0)] = r
        return result_padded.to(device)

class EncoderLayer(nn.Module):
    def __init__(self,dim,num_heads,drop_rate):
        super().__init__()
        self.att = Attention(dim,num_heads,drop_rate)
        self.ini = nn.LayerNorm(dim)
        self.ffn = FFN(dim,drop_rate)
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

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_rate):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, drop_rate)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self,tgt, memory, tgt_mask=None, memory_mask=None):
        # masked self-attention
        tgt2,_ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        # cross attention between encoder and decoder
        tgt2,_ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        # feed-forward
        tgt2 = self.ffn(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

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


class Attention(nn.Module): #MultiHeadAttention
    def __init__(self,dim,num_heads,drop_rate):
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.drpout = nn.Dropout(drop_rate)

    def forward(self,x):
        B, N, C = x.shape #(batch, seq_len, embed_dim)
        # project and split into heads
        qry = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2) #(B, num_heads, N, head_dim)
        key = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2) 
        val = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1,2) 
        # scaled dot-product attention
        att = (qry @ key.transpose(-2,-1) / self.head_dim ** 0.5)
        att = torch.softmax(att, dim=-1)
        att = self.drpout(att)
        out = torch.matmul(att,val) #(B, num_heads, N, head_dim)
        # concatenate heads
        out = out.transpose(1,2).reshape(B, N, C) # (B, N, embed_dim)
        out = self.o_proj(out)
        # concatenate k and v
        key = key.transpose(1,2).reshape(B, N, C)
        val = val.transpose(1,2).reshape(B, N, C)
        return out


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

def generate_mask(sz):
    # generate a (sz,sz) mask (upper triangle) with -inf above the diagonal, 0 elsewhere
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    for batch_idx, combine_data in enumerate(train_loader):
        # ### try to get one data exaple
        # data_iter = iter(train_loader)
        # data,target = next(data_iter)
        data = combine_data[1]  # [batch, np, ph, pw]
        target = combine_data[2]  # [batch, seq_len] (with start and end tokens)
        data, target = data.to(device), target.to(device)
        # data.shape = [64, 1, 28, 28] #tensor: [batch_size,channels,height,width]
        # target is a tensor of shape [64]
        input_seq = target[:,:-1]
        target_seq = target[:,1:] # shift towards right by one token
        # ### try to plot a random image to have a peek
        # img = data[0].cpu().squeeze()
        # plt.imshow(img, cmap='gray')
        # plt.show()
        
        ## patchify the image into a grid (in order to implement vision transformer)
        # data = patchify(data,patch_size) #[64,1,4,4,7,7]
        # ### try to check on the pathify
        # img_patches = patches[0,0]
        # fig,axes = plt.subplots(4,4,figsize=(7,7))
        # for i in range (4):
        #     for j in range(4):
        #         axes[i,j].imshow(img_patches[i,j],cmap='gray')
        #         axes[i,j].axis('off')
        # plt.show() 
        
        optimizer.zero_grad()
        output = model(data, input_seq)  # output: [batch, seq_len, vocab_size]
        # Reshape for loss: flatten batch and seq
        output = output.reshape(-1, output.size(-1))
        target_seq = target_seq.reshape(-1)
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # Token-level accuracy
        pred_tokens = output.argmax(dim=-1)
        correct_tokens += (pred_tokens == target_seq).sum().item()
        total_tokens += target_seq.numel()
        if batch_idx % 100 == 99:
            print(f'Batch: {batch_idx + 1}, Loss: {running_loss/100:.3f}, Token Accuracy: {100.*correct_tokens/total_tokens:.2f}%')
            running_loss = 0.0
            correct_tokens = 0
            total_tokens = 0

def evaluate_model(model, test_loader, device):
    model.eval()
    total_seqs = 0
    correct_seqs = 0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for combine_data in test_loader:
            data = combine_data[1]
            target = combine_data[2]
            data, target = data.to(device), target.to(device)
            # Remove start token for comparison
            target_seq = target[:, 1:]
            # Generate predictions
            pred_seq = model.autoregressive_inference(data, device, max_digits=target_seq.size(1))
            # Pad to same length for comparison
            min_len = min(target_seq.size(1), pred_seq.size(1))
            target_trim = target_seq[:, :min_len]
            pred_trim = pred_seq[:, :min_len]
            # Token-level accuracy
            correct_tokens += (pred_trim == target_trim).sum().item()
            total_tokens += target_trim.numel()
            # Sequence-level accuracy (all tokens must match)
            correct_seqs += ((pred_trim == target_trim).all(dim=1)).sum().item()
            total_seqs += target_seq.size(0)
    token_acc = 100. * correct_tokens / total_tokens if total_tokens>0 else 0.0
    seq_acc = 100. * correct_seqs / total_seqs if total_seqs>0 else 0.0
    print(f'Test Token Accuracy: {token_acc:.2f}%, Sequence Accuracy: {seq_acc:.2f}%')
    return seq_acc

def main():
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

    # loadin the config
    train_cfg = TrainingHyperparameters()
    model_cfg = ModelHyperparameters()

    # ### load in the customed dataset, where each image contains 4 digits in 4 quarants
    train_dataset = Combine(train=True)
    test_dataset = Combine(train=False)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # get the image dimension
    tmp = next(iter(train_dataset))
    model_cfg.img_width = tmp[0].shape[0]
    model_cfg.img_channels = 1
    model_cfg.patch_size = tmp[1].shape[1]

    # Initialize model, loss function, and optimizer
    model = ViT(model_cfg, train_cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    # Training loop
    print('Starting training...')
    for epoch in range(train_cfg.num_epochs):
        print(f'\nEpoch {epoch + 1}/{train_cfg.num_epochs}')
        train_model(model, train_loader, criterion, optimizer, device)
        evaluate_model(model, test_loader, device)

    # Save the trained model
    torch.save(model.state_dict(), 'mnist_transformer_model.pth')
    print('Model saved to mnist_transformer_model.pth')

if __name__ == '__main__':
    main() 