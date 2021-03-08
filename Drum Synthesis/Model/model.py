import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.distributions.categorical import Categorical


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        '''
        data is organized as a numpy array of dimensions
            (number of samples, length) if using pure samples
            (number of samples, frequency bin, length) if using stft
        
        =========== ARGUMENTS ===========  
            > path - path to dataset
        '''
        self.data = np.load(path)
         
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        '''
        Returns the same input and target
        '''
        return torch.from_numpy(self.data[idx])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        '''
        Defines a single residual block for ResNet architecture
        '''
        super(ResidualBlock, self).__init__()
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same 
        # as it's input, have a convolutional layer downsample the layer 
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    # Had to implement this to resolve NotImplemented error
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.shortcut(x)
        return x


        
class ResNetBigger(nn.Module):
    def __init__(self, d=32, dropout_rate=0.5, linear_layer_size=64):
        '''
        =========== ARGUMENTS ===========
            > d - dimensionality of latent space
            > dropout_rate - dropout ratae
            > linear_layer_size - size of linear layer prior to latent space
                I found this had to = 64, not sure exactly why
        =================================
        '''
        super(ResNetBigger, self).__init__()
        print(f"training with dropout={dropout_rate}")

        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        
        self.linear_layer_size=linear_layer_size
        
        # Create blocks
        #self.block1 = self._create_block(64, 64, stride=1)
        #self.block2 = self._create_block(64, 128, stride=2)
        #self.block3 = self._create_block(128, 128, stride=2)
        #self.block4 = self._create_block(128, 128, stride=3)
        self.block1 = self._create_block(64, 64, stride=1)
        self.block2 = self._create_block(64, 32, stride=2)
        self.block3 = self._create_block(32, 16, stride=2)
        self.block4 = self._create_block(16, 16, stride=2)
        self.bn2 = nn.BatchNorm1d(linear_layer_size)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(linear_layer_size, 32)
        self.linear2 = nn.Linear(32, d*2)
      
        self.dropout = nn.Dropout(dropout_rate)
        
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf
    
    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
    # Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x.unsqueeze(1))))
        print('out 1: {}'.format(out.shape))
        out = self.block1(out)
        print('out 2: {}'.format(out.shape))
        out = self.block2(out)
        print('out 3: {}'.format(out.shape))
        out = self.block3(out)
        print('out 4: {}'.format(out.shape))
        out = self.block4(out)
        print('out 5: {}'.format(out.shape))
        out = nn.AvgPool2d((4,1))(out)
        print('out 5: {}'.format(out.shape))
        out = out.view(out.size(0), -1)
        print('out 6: {}'.format(out.shape))
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out
    
    def set_device(self, device):
        for b in [self.block1, self.block2, self.block3, self.block4]:
            b.to(device)
        self.to(device)

class VAE(nn.Module):
    def __init__(self, d, src_path, batch_size, device):
        super(VAE, self).__init__()
        '''
        =========== ARGUMENTS ===========
            > d - dimensionality of latent space
            > src_path - path to source samples
            > batch_size - number of training examples in single batch
            > device - CPU or GPU in use
        =================================
        '''
        self.enc = ResNetBigger(d=d) 
        self.dec = nn.Sequential(
            nn.Linear(d, 40),
            nn.ReLU(),
            nn.Linear(40, 50),
            nn.Sigmoid()
        )
        self.batch_size = batch_size
        self.src = torch.from_numpy(np.load(src_path))
        self.src = self.src.unsqueeze(0)
        self.src = self.src.repeat(self.batch_size, 1, 1, 1)
        self.src = torch.tensor(self.src, device = device)
        self.d = d

    def reparameterise(self, mu, logvar):
        if self.training:  # If the model is training
            std = logvar.mul(0.5).exp_()                # Standard deviation
            eps = std.data.new(std.size()).normal_()    # Follows normal distributions
            return eps.mul(std).add_(mu)
        else:
            return mu

    def combine_samples(self, weights):
        '''
        Takes the weights of the output nodes and combine src samples into output
        '''
        print('src shape: {}'.format(self.src.shape))
        output = weights * self.src                   # Multiply each sample by weight
        output = torch.sum(output, dim=1)             # Sum them all together
        print('output shape: {}'.format(output.shape))
        return output.view(-1) / self.batch_size      # Flatten and normalize

    def forward(self, x):
        # mu_logvar = self.enc(x.view(sample_dim)).view(-1, 2, d)
        mu_logvar = self.enc(x).view(-1, 2, self.d)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)

        # Reshape decoded values to apply to linear combination
        weights = self.dec(z)
        print('z shape: {}'.format(z.shape))
        print('dec z shape: {}'.format(self.dec(z).shape))
        print('weights shape: {}'.format(weights.shape))
        weights = weights.unsqueeze(2).unsqueeze(3)
        weights = weights.repeat(1, 1, self.src.shape[2], self.src.shape[3])
        print('new weights shape: {}'.format(weights.shape))
        # Return the decoded latent variable along with mu and logvar to compute loss
        return self.combine_samples(weights), mu, logvar


