import copy
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import midi_stats
from tqdm import tqdm
import torch_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set number of hits allowed per timestep for each instrument
# kick: 3, snare: 7, closed_hh: 3, open_hh: 4, low_tom: 3, mid_tom: 3, hi_tom: 3, crash: 2, ride: 2
inst_lim = [3, 7, 3, 4, 3, 3, 3, 2, 2]
    
# Set dict to map duplicate instruments to their original instruments
actual_inst_nums = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:2, 11:2, 12:2, 13:3, 14:3, 15:3, 16:3, 17:4, 18:4, 19:4, 20:5, 21:5, 22:5, 23:6, 24:6, 25:6, 26:7, 27:7, 28:8, 29:8}

# Indicates first index of kick, snare, etc in 30 dim output
primary_indices = [0, 3, 10, 13, 17, 20, 23, 26, 28]

def reparam(mu, logvar):
    std = torch.exp(0.5*logvar)
    batch_sigma = torch.diag_embed(std)
    q_z = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=mu,
            covariance_matrix=batch_sigma)
    z = q_z.rsample()
    return z, q_z

def kl_loss_beta_vae(q_z, free_bits, beta_rate, global_step, max_beta):
    # shape: [1, batch_size, z_size]
    batch_size = q_z.loc.shape[1]
    z_size = q_z.loc.shape[2]
    
    # Prior distribution.
    p_z = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(batch_size, z_size).to(device),
            covariance_matrix=torch.diag_embed(torch.ones(batch_size, z_size).to(device)))

    # KL Divergence (nats) - shape: [1, batch_size]
    kl_div = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    #kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    free_nats = free_bits * np.log(2.0)
    
    kl_cost = torch.max(kl_div-free_nats, other=torch.zeros_like(kl_div))

    beta = (torch.pow(torch.tensor(beta_rate),
                     torch.tensor(global_step).float()) * max_beta).to(device)
    
    kl_loss = beta * kl_cost
    
    kl_bits = kl_div / np.log(2.0)
    
    return kl_loss, kl_bits

#def reparameterize(mu, logvar):
#    std = torch.exp(0.5*logvar)
#    eps = torch.randn_like(std).to(device)
#    z = mu + eps*std
#    return z.to(device)

############# ENCODERS AND DECODERS #############

class FFDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.z_size = hparams.z_size
        self.input_dim = hparams.input_dim
        self.output_dim = hparams.input_dim
        
        self.lstm_size = hparams.dec_rnn_size[0]
        self.lstm_layers = len(hparams.dec_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        
        self.linear1 = nn.Linear(hparams.z_size, 300, bias=True)
        self.projection = nn.Linear(300,4*self.output_dim, bias=True)
        
    def forward(self, batch_size, inputs=None, targets=None, z=None, sampling_probability=0., length=4, mode='training'):
        # z shape: [1,batch_size,z_size]
        x = nn.ReLU()(self.linear1(z))
        x = self.dropout(x)
        flat_out = self.projection(x)
        out = flat_out.view(4,batch_size,-1)
        loss = torch.nn.MSELoss()(out, targets)
        return out, (loss,loss,loss)

class LSTMEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.input_dim = hparams.input_dim
        self.lstm_size = hparams.enc_rnn_size[0]
        self.lstm_layers = len(hparams.enc_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        self.z_size = hparams.z_size

        #self.projection_size=300
        #self.input_projection = nn.Linear(self.input_dim, self.projection_size)

        self.forward_lstm = nn.LSTM(self.input_dim, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)
        #self.forward_lstm = nn.LSTM(self.projection_size, hidden_size=self.lstm_size,
        #                    num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)

        self.mu_projection = nn.Linear(self.lstm_size, self.z_size, bias=True)
        self.sigma_projection = nn.Linear(self.lstm_size, self.z_size, bias=True)

    def forward(self, src):
        #src shape: [batch size, seq_len, input_dim]
        batch_size, seq_len, input_dim = src.shape

        #src = self.input_projection(src)

        forward_lstm_output, forward_lstm_state = self.forward_lstm(src)
        forward_final_h, forward_final_c = forward_lstm_state

        encoder_out = forward_final_h[-1:,:,:]
        #encoder_out = torch.cat([forward_final_h[-1:,:,:], backward_final_h[-1:,:,:]],dim=2)

        # mu and sigma shapes: [1, batch_size, z_size]
        mu = self.mu_projection(encoder_out)
        logvar = self.sigma_projection(encoder_out) #nn.Softplus()(self.sigma_projection(encoder_out))

        z, q_z = reparam(mu, logvar)

        return mu, logvar, z, q_z
    
class BidirectionalLSTMEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        
        self.input_dim = hparams.input_dim
        self.lstm_size = hparams.enc_rnn_size[0]
        self.lstm_layers = len(hparams.enc_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        self.z_size = hparams.z_size
        
        self.projection_size=300
        
        self.input_projection = nn.Linear(self.input_dim, self.projection_size)

        #self.lstms = nn.ModuleList() #[]
        
        self.forward_lstm = nn.LSTM(self.projection_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)
        
        self.backward_lstm = nn.LSTM(self.projection_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)

        # sigma and mu, let's see how this plays out in forward()
        self.mu_projection = nn.Linear(self.lstm_size*2, self.z_size, bias=True)
        self.sigma_projection = nn.Linear(self.lstm_size*2, self.z_size, bias=True)

    def forward(self, src):
        #src shape: [batch size, seq_len, input_dim]
        batch_size, seq_len, input_dim = src.shape

        src = self.input_projection(src)
        
        reverse_src = torch.flip(src, dims=[1])
        
        
        forward_lstm_output, forward_lstm_state = self.forward_lstm(src)
        forward_final_h, forward_final_c = forward_lstm_state
        
        backward_lstm_output, backward_lstm_state = self.backward_lstm(reverse_src)
        backward_final_h, backward_final_c = backward_lstm_state
        
        
        encoder_out = torch.cat([forward_final_h[-1:,:,:], backward_final_h[-1:,:,:]],dim=2)
        
        # mu and sigma shapes: [1, batch_size, z_size]
        mu = self.mu_projection(encoder_out)
        logvar = self.sigma_projection(encoder_out) #nn.Softplus()(self.sigma_projection(encoder_out))
        
        z, q_z = reparam(mu, logvar)
        
        return mu, logvar, z, q_z
    
class BottomUpEncoder(nn.Module):
    def __init__(self, hparams, pretrained_models, trainable_embeddings=False):
        super().__init__()
        
        # First handle loading pretrained models
        self.hparams = hparams
        self.trainable_embeddings = trainable_embeddings
        self.pretrained_models = pretrained_models
        self.hierarchy_step_lengths = []
        
        # Load pretrained VAE models (only using encoders here)
        #for i in range(self.pretrained_checkpoint_dirs):
        #    checkpoint_dir = self.pretrained_checkpoint_dirs[i]
        #    vae = GrooVAE(hparams = self.pretrained_hparams_list[i])
        #    torch_utils.load_checkpoint(checkpoint_dir, vae)
        #    if trainable_flags[i] is False:  
        #        for p in vae.parameters(): ## TODO is this the right way to do it?
        #            p.requires_grad = False
        #    self.pretrained_models.append(vae)
        
        # Now handle initializing this new model's components
        self.input_dim = hparams.input_dim
        self.lstm_size = hparams.enc_rnn_size[0]
        self.lstm_layers = len(hparams.enc_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        self.z_size = hparams.z_size
        
        self.subseq_embedding_dim = self.pretrained_models[0].hparams.z_size
        
        self.projection_size=300
        
        #self.input_projection = nn.Linear(self.input_dim, self.projection_size)
        self.input_projection = nn.Linear(self.subseq_embedding_dim, self.projection_size)

        #self.lstms = nn.ModuleList() #[]
        
        self.forward_lstm = nn.LSTM(self.projection_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)
        
        self.backward_lstm = nn.LSTM(self.projection_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)

        self.mu_projection = nn.Linear(self.lstm_size*2, self.subseq_embedding_dim, bias=True)
        self.sigma_projection = nn.Linear(self.lstm_size*2, self.subseq_embedding_dim, bias=True)
        
    def forward(self, src):
        #src shape: [batch size, seq_len, input_dim]
        batch_size, seq_len, input_dim = src.shape
        
        # TODO handle more levels and diff low-level sequence lengths
        # For now assume low level sequence is length=1 timestep
        
        # split and embed all sub-sequences
        
        # Split by steps
        step_tensors = src.split(1,dim=1) # tuple of length 32, each element is (128,1,90) (batch_size, subseq_len, input_dim)
        # Encode each step with the low-level encoder
        batch_step_tensors = torch.cat(step_tensors, dim=0) # batched by sequence length first, then batch_size
        # Lookup embeddings using low-level model in a single call
        _, _, _z, _ = self.pretrained_models[0].encode(batch_step_tensors)
        # split to get back to batched sequences of embeddings
        embedding_sequences = torch.cat(_z.split(seq_len,dim=1))

        src = embedding_sequences
        src = self.input_projection(src)
        reverse_src = torch.flip(src, dims=[1])
        
        
        forward_lstm_output, forward_lstm_state = self.forward_lstm(src)
        forward_final_h, forward_final_c = forward_lstm_state
        
        backward_lstm_output, backward_lstm_state = self.backward_lstm(reverse_src)
        backward_final_h, backward_final_c = backward_lstm_state
        
        encoder_out = torch.cat([forward_final_h[-1:,:,:], backward_final_h[-1:,:,:]],dim=2)
        
        # mu and sigma shapes: [1, batch_size, z_size]
        mu = self.mu_projection(encoder_out)
        logvar = self.sigma_projection(encoder_out) #nn.Softplus()(self.sigma_projection(encoder_out))
        
        z, q_z = reparam(mu, logvar)
        
        return mu, logvar, z, q_z
    

class BottomUpDecoder(nn.Module):
    # Take in a list of pretrained VAE model paths,
    # hparams for each,
    # and Module classes for each model
    # along with a corresponding list of how many timesteps each model handles (in hparams)
    # and a flag for whether each should be trainable
    # Input to the Forward function is a single (batch) Z-vector
    # This decoder predicts the (lower-dim) Z vectors at each level to decode
    #def __init__(self, hparams, pretrained_checkpoint_dirs, pretrained_hparams_list, trainable_flags):
    def __init__(self, hparams, pretrained_models, trainable_embeddings=False):
        super().__init__()
        
        # First handle loading pretrained models
        self.hparams = hparams
        self.trainable_embeddings = trainable_embeddings
        self.pretrained_models = pretrained_models
        self.hierarchy_step_lengths = []
        
        # Load pretrained VAE models (only using decoders here)
        #for i in range(self.pretrained_checkpoint_dirs):
        #    checkpoint_dir = self.pretrained_checkpoint_dirs[i]
        #    vae = GrooVAE(hparams = self.pretrained_hparams_list[i])
        #    torch_utils.load_checkpoint(checkpoint_dir, vae)
        #    if trainable_flags[i] is False:  
        #        for p in vae.parameters(): ## TODO is this the right way to do it?
        #            p.requires_grad = False
        #    self.pretrained_models.append(vae)
            
        
        # Now handle initializing this new model's components
        self.input_dim = hparams.input_dim
        self.output_dim = hparams.output_dim
        self.lstm_size = hparams.dec_rnn_size[0]
        self.lstm_layers = len(hparams.dec_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        self.z_size = hparams.z_size
        
        self.subseq_embedding_dim = self.pretrained_models[0].hparams.z_size
        
        # Make the input projection size the same as the lstm size
        self.input_projection = nn.Linear(self.subseq_embedding_dim, self.lstm_size)
        
        self.lstm = nn.LSTM(self.lstm_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)
        
        self.input_h_projections = nn.ModuleList()
        
        for i in range(self.lstm_layers):
            self.input_h_projections.append(nn.Linear(self.z_size, self.lstm_size).to(device))        

        self.lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout)
        
        self.output_projection = nn.Linear(self.lstm_size, self.subseq_embedding_dim)
        
      
        self.to(device)
               
            
    def forward(self, z=None, targets=None, embedding_targets=None, sampling_probability=0.,
                length=32, mode='training', batch_size=1):
        # Targets here refers to the final Midi/Drumroll representation, not the target low-level embeddings
        # Training on just these targets means we are optimizing just by differentating through the hierarchy
        # TODO: implement embedding_targets to train directly on those - should be easier to optimize...

        # Given a latent vector Z (or none), which represents an entire (e.g. 2-bar) sequence
        # Decode a (batch) sequence of embeddings of shape (batch_size, length, embedding_dim) e.g. (128, 32, 64)
        if embedding_targets is None or mode!='training' or np.random.uniform(0,1) < sampling_probability:
            do_sample = True
        else:
            do_sample = False
        
        # make these instance vars so we can see them inside _sample()
        self.embedding_outputs = [] # Embeddings to be input to lower level decoders
        self.raw_outputs = [] # Logits from the lower level decoders
        self.outputs = [] # Sampled outputs of the lower level decoders
        
        # if z is given, set the initial LSTM states using the projections from z
        # z shape is be [1, batch_size, z_size]
        if z is not None:
            batch_size = z.shape[1]
        
        if z is not None:
            z.to(device)
            initial_states = []
            for i in range(self.lstm_layers):
                initial_states.append(nn.Tanh()(self.input_h_projections[i](z)))
            initial_h = torch.cat(initial_states, dim=0)
            initial_c = torch.zeros_like(initial_h)
            # [num_layers, batch_size, hidden_size]
            state = (initial_h, initial_c)
        else:
            state = None
         
        # First time step
        # input feature at time t0 is 0 vector
        lstm_input = torch.zeros(1, batch_size, self.subseq_embedding_dim).to(device)
        # Loop over timesteps
        for i in range(0, length):
            lstm_input = torch.nn.Tanh()(self.input_projection(lstm_input))
            lstm_output, state = self.lstm(lstm_input, state)
            #lstm_output = self.dropout(lstm_output)
            lstm_out = self.output_projection(lstm_output)
            self.embedding_outputs.append(lstm_out) # embeddings to be decoded lower level
            
            #out_logits, out = self.pretrained_models[0].dec(lstm_out)
            out_logits, out = self.pretrained_models[0].dec(z=lstm_out,length=self.pretrained_models[0].hparams.max_seq_len)
            
            self.raw_outputs.append(out_logits)
            self.outputs.append(out)
            #out = self._sample(lstm_out)
            
            if do_sample: # No embedding targets or loss, just sample
                lstm_input = lstm_out.detach()
            else:
                lstm_input = embedding_targets[i:i+1].detach()
        
        self.embedding_outputs = torch.cat(self.embedding_outputs,dim=0)
        self.outputs = torch.cat(self.outputs,dim=0)
        self.raw_outputs = torch.cat(self.raw_outputs,dim=0)
        
        return self.outputs, self.raw_outputs#, self.embedding_outputs
    
class GrooveLSTMDecoder(nn.Module):
    def __init__(self, hparams):
        
        super().__init__()
        
        self.input_dim = hparams.input_dim
        self.output_dim = hparams.input_dim
        self.lstm_size = hparams.dec_rnn_size[0]
        self.lstm_layers = len(hparams.dec_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        self.z_size = hparams.z_size
        
        # Make the input projection size the same as the lstm size
        self.input_projection = nn.Linear(self.input_dim, self.lstm_size)
        
        self.lstm = nn.LSTM(self.input_dim, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)
        
        self.input_h_projections = nn.ModuleList()
        
        for i in range(self.lstm_layers):
            self.input_h_projections.append(nn.Linear(self.z_size, self.lstm_size).to(device))        

        self.lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout)
        
        self.output_projection = nn.Linear(self.lstm_size, self.output_dim)
      
        self.to(device)
    
    def _sample(self, rnn_output, temperature=1.0, constrain_outputs=False, constrain_hits=False):
            # Dimension of rnn_output is [1,batch_size,90]
            h,v,o = torch.split(rnn_output, int(self.input_dim/3), dim=-1)

            hits_sampler = torch.distributions.bernoulli.Bernoulli(logits=h)
            sampled_hits = hits_sampler.sample()
            
            if constrain_hits:             
                # Don't let a 1 occur after a 0 at the same instrument and timestep - that would be invalid   
                # We're not directly affecting the gradient here because the loss is computed on the logits.
                # We're just enforcing that the sampled LSTM inputs at each step are "valid".
                # Basically trying to make the scheduled sampling work better.
                divided_hits = list(torch.split(sampled_hits,1,dim=-1))
                for i in range(len(divided_hits)):
                    if not i in primary_indices:
                        divided_hits[i] = divided_hits[i]*divided_hits[i-1]
                sampled_hits = torch.cat(divided_hits, dim=-1)
                
                # Don't apply in training right at the start because it will be too slow
                # After a few steps there won't be as many hits anyways
                if len(torch.nonzero(sampled_hits)) < 500:
                    for _, b, i in torch.nonzero(sampled_hits):
                        i = int(i); b = int(b)
                        instrument_index = actual_inst_nums[i]
                        min_time_allowed = midi_stats.min_times[instrument_index]
                        if i in primary_indices:
                            #pass
                            # Across-timesteps - don't allow less than min_time (in offset units) between successive positions
                            # look at last output for this inst at previous timestep
                            if len(self.outputs) == 0: # skip if there is no previous timestep
                                pass
                            else:
                                #pass
                                """
                                This basically never happens so we can skip it...
                                # Only look if it has the possibility of clashing w/ previous step
                                if sampled_hits[0,b,i] == 1 and (o[0,b,i] - min_time_allowed) < 1:
                                    # current primary index + current inst limit - 1
                                    #last_step_index = primary_indices[instrument_index] + inst_lim[instrument_index] - 1
                                    last_start_index = primary_indices[instrument_index]
                                    last_end_index = last_start_index+inst_lim[instrument_index]
                                    last_outputs = self.outputs[-1][0,b,last_start_index:last_end_index]
                                    last_offsets = self.outputs[-1][0,b,60+last_start_index:60+last_end_index]
                                    last_offset = torch.max(last_outputs*last_offsets)
                                    # no need to constrain if there were no hits at the last step
                                    # those hits would have been input at this step
                                    if last_offset > 0:
                                        last_offset_on_current_step_scale = last_offset-2
                                        # make current offset at least min_distance from last offset
                                        if o[0,b,i] < last_offset_on_current_step_scale + min_time_allowed:
                                            o[0,b,i] = torch.clamp(last_offset_on_current_step_scale + min_time_allowed,-1,1)
                                            print('Across-step-forcing')
                                """
                        else:
                            # TODO should be able to vectorize this, doesn't seem to be taking too long though
                            # Within-timestep - don't allow less than min_time (in offset units) between successive positions
                            # Only look at offsets of the hits that are actually sampled and get played
                            if sampled_hits[0,b,i] == 1: 
                                if o[0,b,i] < o[0,b,i-1] + min_time_allowed:
                                    o[0,b,i] = torch.clamp(o[0,b,i-1] + min_time_allowed, -1,1)
                                    #print('Within-step-forcing')  

            return torch.cat([sampled_hits,v,o],dim=-1)
        
    def forward(self, z=None, targets=None, sampling_probability=0.,
                length=32, mode='training', batch_size=1,
                constrain_outputs=False, constrain_hits=False):
        # shape  [seq_len, batch_size, input_dim]
        
        if targets is None or mode!='training' or np.random.uniform(0,1) < sampling_probability:
            do_sample = True
        else:
            do_sample = False
        
        self.outputs = [] # make these instance vars so we can see them inside _sample()
        self.raw_outputs = [] # logits
        
        # if z is given, set the initial LSTM states using the projections from z
        # z shape is be [1, batch_size, z_size]
        if z is not None:
            batch_size = z.shape[1]
        
        if z is not None:
            z.to(device)
            initial_states = []
            for i in range(self.lstm_layers):
                initial_states.append(nn.Tanh()(self.input_h_projections[i](z)))
            initial_h = torch.cat(initial_states, dim=0)
            initial_c = torch.zeros_like(initial_h)
            # [num_layers, batch_size, hidden_size]
            state = (initial_h, initial_c)
        else:
            state = None
         
        # First time step
        # input feature at time t0 is 0 vector
        lstm_input = torch.zeros(1, batch_size, self.input_dim).to(device)
        # Loop over timesteps
        for i in range(0, length):
            lstm_input = torch.nn.Tanh()(self.input_projection(lstm_input))
            lstm_output, state = self.lstm(lstm_input, state)
            #lstm_output = self.dropout(lstm_output)
            lstm_out = self.output_projection(lstm_output)
            h,v,o = torch.split(lstm_out, int(self.input_dim/3), dim=-1)
            v=v.clamp(0,1)
            o=o.clamp(-1,1)
            lstm_out=torch.cat([h,v,o],dim=-1)
            self.raw_outputs.append(lstm_out)
            out = self._sample(lstm_out, constrain_outputs=constrain_outputs, constrain_hits=constrain_hits)
            self.outputs.append(out)
            
            if do_sample: # No targets or loss, just sample
                lstm_input = out.detach()
            else:
                lstm_input = targets[i:i+1].detach()
        
        self.outputs = torch.cat(self.outputs,dim=0)
        self.raw_outputs = torch.cat(self.raw_outputs,dim=0)
        
        return self.outputs, self.raw_outputs
    
class DrumPerfLSTMDecoder(nn.Module):
    def __init__(self, hparams):
        ## TODO Init and most of Forward is copied from GrooveLSTMDecoder. Can refactor out
        super().__init__()
        
        self.input_dim = hparams.input_dim
        self.output_dim = hparams.input_dim
        self.lstm_size = hparams.dec_rnn_size[0]
        self.lstm_layers = len(hparams.dec_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        self.z_size = hparams.z_size
        
        # Make the input projection size the same as the lstm size
        self.input_projection = nn.Linear(self.input_dim, self.lstm_size)
        
        self.lstm = nn.LSTM(self.input_dim, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)
        
        self.input_h_projections = nn.ModuleList()
        
        for i in range(self.lstm_layers):
            self.input_h_projections.append(nn.Linear(self.z_size, self.lstm_size).to(device))        

        self.lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout)
        
        self.output_projection = nn.Linear(self.lstm_size, self.output_dim)
      
        self.to(device)
        
    def forward(self, z=None, targets=None, sampling_probability=0.,
                length=281, mode='training', batch_size=1,
                constrain_outputs=False, constrain_hits=False,
                seq_lens=None):
        
        if targets is None or mode!='training' or np.random.uniform(0,1) < sampling_probability:
            do_sample = True
        else:
            do_sample = False
        
        self.outputs = [] # make these instance vars so we can see them inside _sample()
        self.raw_outputs = [] # logits
        
        # if z is given, set the initial LSTM states using the projections from z
        # z shape is be [1, batch_size, z_size]
        if z is not None:
            batch_size = z.shape[1]
        
        if z is not None:
            z.to(device)
            initial_states = []
            for i in range(self.lstm_layers):
                initial_states.append(nn.Tanh()(self.input_h_projections[i](z)))
            initial_h = torch.cat(initial_states, dim=0)
            initial_c = torch.zeros_like(initial_h)
            # [num_layers, batch_size, hidden_size]
            state = (initial_h, initial_c)
        else:
            state = None
            
        if targets is not None:
            length = len(targets)
         
        # First time step
        # input feature at time t0 is 0 vector
        lstm_input = torch.zeros(1, batch_size, self.input_dim).to(device)
        # Loop over timesteps
        for i in range(0, length):
            lstm_input = torch.nn.Tanh()(self.input_projection(lstm_input))
            lstm_output, state = self.lstm(lstm_input, state)
            #lstm_output = self.dropout(lstm_output)
            lstm_out = self.output_projection(lstm_output)
            # In performance rep, everything is one-hot, no no split losses needed here
            self.raw_outputs.append(lstm_out)
            out = self._sample(lstm_out)
            self.outputs.append(out)
            
            if do_sample: # No targets or loss, just sample
                #lstm_input = out.detach()
                lstm_input = torch_utils.torch_one_hot(out.detach(), device, n_dims=self.output_dim)
            else:
                lstm_input = targets[i:i+1].detach()
        
        self.outputs = torch.cat(self.outputs,dim=0)
        self.raw_outputs = torch.cat(self.raw_outputs,dim=0)
        
        return self.outputs, self.raw_outputs
    
    def _sample(self, rnn_output, temperature=1.0):
        sampler = torch.distributions.categorical.Categorical(logits=rnn_output)
        return sampler.sample()
    

class EmbeddingEncoder(nn.Module):
    def __init__(self, hparams, subseq_embedding_dim=32):
        super().__init__()
        
        self.hparams = hparams
        self.input_dim = hparams.input_dim
        self.lstm_size = hparams.enc_rnn_size[0]
        self.lstm_layers = len(hparams.enc_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        self.z_size = hparams.z_size
        
        self.subseq_embedding_dim = subseq_embedding_dim
        
        #self.projection_size=300
        #self.input_projection = nn.Linear(self.subseq_embedding_dim, self.projection_size)
        self.projection_size=self.subseq_embedding_dim
        
        self.forward_lstm = nn.LSTM(self.projection_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)
        
        self.backward_lstm = nn.LSTM(self.projection_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout, batch_first=True)

        self.mu_projection = nn.Linear(self.lstm_size*2, self.z_size, bias=True)
        self.sigma_projection = nn.Linear(self.lstm_size*2, self.z_size, bias=True)
        
    def forward(self, src):
        #src shape: [batch size, seq_len, embedding_dim]
        batch_size, seq_len, input_dim = src.shape
        
        # split and embed all sub-sequences
        #embedding_sequences = torch_utils.encode_to_subsequence_embeddings(src, subsequence_vae, subsequence_length)
        #src = embedding_sequences
        #src = self.input_projection(src)
        reverse_src = torch.flip(src, dims=[1])
        
        forward_lstm_output, forward_lstm_state = self.forward_lstm(src)
        forward_final_h, forward_final_c = forward_lstm_state
        backward_lstm_output, backward_lstm_state = self.backward_lstm(reverse_src)
        backward_final_h, backward_final_c = backward_lstm_state
        encoder_out = torch.cat([forward_final_h[-1:,:,:], backward_final_h[-1:,:,:]],dim=2)
        
        # mu and sigma shapes: [1, batch_size, z_size]
        mu = self.mu_projection(encoder_out)
        logvar = self.sigma_projection(encoder_out) #nn.Softplus()(self.sigma_projection(encoder_out))
        z, q_z = reparam(mu, logvar)
        
        return mu, logvar, z, q_z    

class EmbeddingDecoder(nn.Module):
    def __init__(self, hparams, subseq_embedding_dim=32):
        super().__init__()
        
        self.hparams = hparams
        self.input_dim = hparams.input_dim
        self.output_dim = hparams.output_dim
        self.lstm_size = hparams.dec_rnn_size[0]
        self.lstm_layers = len(hparams.dec_rnn_size)
        self.dropout = nn.Dropout(hparams.dropout)
        self.z_size = hparams.z_size
        
        self.subseq_embedding_dim = subseq_embedding_dim
        
        # Make the input projection size the same as the lstm size
        #self.input_projection = nn.Linear(self.subseq_embedding_dim, self.lstm_size)
        

        
        self.input_h_projections = nn.ModuleList()
        
        for i in range(self.lstm_layers):
            self.input_h_projections.append(nn.Linear(self.z_size, self.lstm_size).to(device))        

        #self.lstm = nn.LSTM(input_size=self.lstm_size, hidden_size=self.lstm_size,
        #                    num_layers=self.lstm_layers, dropout=hparams.dropout)
        self.lstm = nn.LSTM(input_size=self.subseq_embedding_dim, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams.dropout)
        
        
        self.output_projection = nn.Linear(self.lstm_size, self.subseq_embedding_dim)
        
      
        self.to(device)
               
            
    def forward(self, z=None, targets=None, sampling_probability=0.,
                length=32, mode='training', batch_size=1,
                constrain_outputs=None, constrain_hits=None):
        # Targets here are the sequence of embedding vectors
        # Given a latent vector Z (or none), which represents an entire (e.g. 2-bar) sequence
        # Decode a (batch) sequence of embeddings of shape (batch_size, length, embedding_dim) e.g. (128, 32, 64)
        if targets is None or mode!='training' or np.random.uniform(0,1) < sampling_probability:
            do_sample = True
        else:
            do_sample = False
        
        # make these instance vars so we can see them inside _sample()
        self.embedding_outputs = [] # Embeddings to be input to lower level decoders
        self.raw_outputs = [] # Logits from the lower level decoders
        self.outputs = [] # Sampled outputs of the lower level decoders
        
        # if z is given, set the initial LSTM states using the projections from z
        # z shape is be [1, batch_size, z_size]
        if z is not None:
            batch_size = z.shape[1]
        
        if z is not None:
            z.to(device)
            initial_states = []
            for i in range(self.lstm_layers):
                initial_states.append(nn.Tanh()(self.input_h_projections[i](z)))
            initial_h = torch.cat(initial_states, dim=0)
            initial_c = torch.zeros_like(initial_h)
            # [num_layers, batch_size, hidden_size]
            state = (initial_h, initial_c)
        else:
            state = None
         
        # First time step
        # input feature at time t0 is 0 vector
        lstm_input = torch.zeros(1, batch_size, self.subseq_embedding_dim).to(device)
        # Loop over timesteps
        for i in range(0, length):
            #lstm_input = torch.nn.Tanh()(self.input_projection(lstm_input))
            lstm_output, state = self.lstm(lstm_input, state)
            lstm_out = self.output_projection(lstm_output)
            self.embedding_outputs.append(lstm_out) # embeddings to be decoded at lower level
            
            #out_logits, out = self.pretrained_models[0].dec(lstm_out)
            #out_logits, out = self.pretrained_models[0].dec(z=lstm_out,length=self.pretrained_models[0].hparams.max_seq_len)
            #self.raw_outputs.append(out_logits)
            #self.outputs.append(out)
            #out = self._sample(lstm_out)
            
            if do_sample: # No embedding targets or loss, just sample
                lstm_input = lstm_out.detach()
            else:
                lstm_input = targets[i:i+1].detach()
        
        self.embedding_outputs = torch.cat(self.embedding_outputs,dim=0)
        #self.outputs = torch.cat(self.outputs,dim=0)
        #self.raw_outputs = torch.cat(self.raw_outputs,dim=0)

        return self.embedding_outputs, self.embedding_outputs # TODO: repeats, just to be consistent
        #return self.outputs, self.raw_outputs#, self.embedding_outputs
    


############# END ENCODERS AND DECODERS #############
    
############# MAIN VAE MODELS #############

    
class GrooveScore(nn.Module):
    def __init__(self, score_hparams, groove_hparams, sampling_probability=0.,
                 aux_decoders=False):
        super().__init__()
        self.global_step = 0
        self.best_val_loss = np.inf
        self.sampling_probability = sampling_probability
        self.aux_decoders = aux_decoders
        self.constrain_outputs = score_hparams.get('constrain_outputs') is True
        self.constrain_hits = score_hparams.get('constrain_hits') is True
        self.score_hparams = score_hparams
        self.groove_hparams = groove_hparams
        self.score_encoder = BidirectionalLSTMEncoder(score_hparams).to(device)
        self.groove_encoder = BidirectionalLSTMEncoder(groove_hparams).to(device)
        decoder_hparams = copy.deepcopy(score_hparams)
        decoder_hparams.z_size = score_hparams.z_size + groove_hparams.z_size
        self.dec = GrooveLSTMDecoder(decoder_hparams).to(device)
        if aux_decoders:
            self.score_decoder = GrooveLSTMDecoder(score_hparams).to(device)
            self.groove_decoder = GrooveLSTMDecoder(groove_hparams).to(device)
        self.to(device)
        
    def forward(self, src_score, src_groove, targets=None,sampling_probability=None,
                score_targets=None, groove_targets=None):
        #src = src.float().to(device)
        src_score = src_score.float().to(device)
        src_groove = src_groove.float().to(device)
        
        # score encoder
        mu_score, logvar_score, z_score, q_z_score = self.score_encoder(src_score)
        
        # groove encoder
        mu_groove, logvar_groove, z_groove, q_z_groove = self.groove_encoder(src_groove)

        p = sampling_probability if sampling_probability is not None else self.sampling_probability #todo clean this
        
        z = torch.cat([z_score, z_groove], dim=-1)
        
        # Main Decoder
        outputs, raw_outputs = self.dec(z=z,targets=targets,sampling_probability=p,
                                        constrain_outputs=self.constrain_outputs,
                                        constrain_hits=self.constrain_hits)
        
        if self.aux_decoders:
            score_outputs, score_raw_outputs = self.score_decoder(z=z_score,targets=score_targets,
                sampling_probability=p, constrain_outputs=self.constrain_outputs, constrain_hits=self.constrain_hits)
            
            groove_outputs, groove_raw_outputs = self.groove_decoder(z=z_groove,targets=groove_targets,
                sampling_probability=p, constrain_outputs=self.constrain_outputs, constrain_hits=self.constrain_hits)
            
            return q_z_score, q_z_groove, outputs, raw_outputs, score_outputs, score_raw_outputs, groove_outputs, groove_raw_outputs
        else:       
            return q_z_score, q_z_groove, outputs, raw_outputs
    
    def encode_score(self, src_score):
        return self.score_encoder(src_score)

    def encode_groove(self, src_groove):
        return self.groove_encoder(src_groove)    
    
    def decode(self, z):
        outputs, raw_outputs = self.dec(z=z,constrain_outputs=self.constrain_outputs, constrain_hits=self.constrain_hits)
        return outputs, raw_outputs
        
    def sample(self, batch_size=1, length=32):
        z_size_score = self.score_hparams.z_size
        z_size_groove = self.groove_hparams.z_size
        p_z_score = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(1, batch_size, z_size_score).to(device),
            covariance_matrix=torch.diag_embed(torch.ones(1, batch_size, z_size_score).to(device)))
        z_score = p_z_score.sample()
        p_z_groove = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(1, batch_size, z_size_groove).to(device),
            covariance_matrix=torch.diag_embed(torch.ones(1, batch_size, z_size_groove).to(device)))
        z_groove = p_z_groove.sample()
        
        z = torch.cat([z_score, z_groove], dim=-1)
      
        outputs, raw_outputs = self.dec(z=z)
        return outputs, raw_outputs, z

class GrooVAE(nn.Module):
    def __init__(self, hparams, sampling_probability=0.,
                 encoder_class=BidirectionalLSTMEncoder,decoder_class=GrooveLSTMDecoder):
        super().__init__()
        self.global_step = 0
        self.best_val_loss = np.inf
        self.max_seq_len=hparams.max_seq_len
        self.sampling_probability = sampling_probability
        self.constrain_outputs = hparams.get('constrain_outputs') is True
        self.constrain_hits = hparams.get('constrain_hits') is True
        self.hparams = hparams
        self.enc = encoder_class(hparams).to(device)
        self.dec = decoder_class(hparams).to(device)
        self.to(device)
        
    def forward(self, src, targets=None,sampling_probability=None, length=None):
        src = src.float().to(device)
        mu, logvar, z, q_z = self.enc(src)
        p = sampling_probability if sampling_probability is not None else self.sampling_probability #todo clean this
        max_seq_len = length if length is not None else self.max_seq_len
        outputs, raw_outputs = self.dec(z=z,targets=targets,sampling_probability=p,
                                        length=max_seq_len,
                                        constrain_outputs=self.constrain_outputs,
                                        constrain_hits=self.constrain_hits)
        return q_z, outputs, raw_outputs
    
    def encode(self, src):
        src = src.float().to(device)
        mu, logvar, z, q_z = self.enc(src)
        return mu, logvar, z, q_z
    
    def decode(self, z):
        outputs, raw_outputs = self.dec(z=z,constrain_outputs=self.constrain_outputs, constrain_hits=self.constrain_hits)
        return outputs, raw_outputs
        
    def sample(self, batch_size=1, length=32):
        z_size = self.hparams.z_size
        p_z = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(1, batch_size, z_size).to(device),
            covariance_matrix=torch.diag_embed(torch.ones(1, batch_size, z_size).to(device)))
        z = p_z.sample()
        outputs, raw_outputs = self.dec(z=z,length=length)
        return outputs, raw_outputs, z
    
class MultiLevelVAE(nn.Module):
    def __init__(self, embedding_hparams, subsequence_hparams, sampling_probability=0.8):
        super().__init__()
        self.global_step = 0
        self.best_val_loss = np.inf
        self.max_seq_len=embedding_hparams.max_seq_len
        self.sampling_probability = sampling_probability
        self.constrain_outputs = subsequence_hparams.get('constrain_outputs') is True
        self.constrain_hits = subsequence_hparams.get('constrain_hits') is True
        self.embedding_hparams = embedding_hparams
        self.subsequence_hparams = subsequence_hparams
        
        self.embedding_encoder = EmbeddingEncoder(embedding_hparams, subseq_embedding_dim=32).to(device)
        self.subsequence_encoder = BidirectionalLSTMEncoder(subsequence_hparams).to(device)
        
        self.embedding_decoder = EmbeddingDecoder(embedding_hparams, subseq_embedding_dim=32).to(device)
        self.subsequence_decoder = GrooveLSTMDecoder(subsequence_hparams).to(device)
        
    def forward(self, src, targets=None, sampling_probability=None, length=None):
        p = sampling_probability if sampling_probability is not None else self.sampling_probability #todo clean this
        max_seq_len = length if length is not None else self.max_seq_len
        
        # Encode tensors to sequence of embeddings with Subsequence Encdoer
        x_embeddings, q_zs = torch_utils.encode_to_subsequence_embeddings(
            src, self.subsequence_encoder, sequence_length=32,return_kl_params=True,return_mu=False)
        x_embeddings = x_embeddings.float().to(device)
        if targets is not None:
            # If targets aren't passed in, we need to compute this later, outside the model, to get the embedding loss
            with torch.no_grad():
                target_embeddings = torch_utils.encode_to_subsequence_embeddings(
                        targets, self.subsequence_encoder, sequence_length=32).detach().float().to(device)#.permute(1,0,2)
        else:
            target_embeddings = None
        
        # Run Embedding Encoder to get Z
        mu, logvar, z, q_z = self.embedding_encoder(x_embeddings)
        # Run Embedding Decoder
        output_embeddings, _ = self.embedding_decoder(
            z=z, targets=target_embeddings, sampling_probability=p, length=max_seq_len)
        
        # Run Subsequence decoder on embedding outputs
        output_tensors, raw_output_tensors = torch_utils.decode_from_subsequence_embeddings(
            output_embeddings, self.subsequence_decoder)
        
        h = {}
        h['q_z_embedding'] = q_z
        h['q_z_subsequences'] = q_zs
        h['output_tensors'] = output_tensors
        h['raw_output_tensors'] = raw_output_tensors
        h['output_embeddings'] = output_embeddings
        h['target_embeddings'] = target_embeddings
        
        return h
    
    def sample(self, batch_size=1, length=32):
        z_size = self.embedding_hparams.z_size
        p_z = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(1, batch_size, z_size).to(device),
            covariance_matrix=torch.diag_embed(torch.ones(1, batch_size, z_size).to(device)))
        z = p_z.sample()
        
        # Run Embedding Decoder
        output_embeddings, _ = self.embedding_decoder(
            z=z,length=length)
        # Run Subsequence decoder on embedding outputs
        outputs, raw_outputs = torch_utils.decode_from_subsequence_embeddings(
            output_embeddings, self.subsequence_decoder)
        return outputs, raw_outputs, z

        
        
        

"""
class LevelVAE(nn.Module):
    def __init__(self, hparams, sampling_probability=0,
                 pretrained_checkpoint_dir=None,
                 pretrained_hparams=None,
                 trainable_embeddings=False):
        super().__init__()
        
        self.hparams = hparams
        self.pretrained_checkpoint_dir = pretrained_checkpoint_dir
        self.pretrained_hparams = pretrained_hparams
        self.trainable_embeddings = trainable_embeddings
        self.pretrained_models = nn.ModuleList()
        self.hierarchy_step_lengths = []
        
        # Load pretrained VAE model
        pretrained_vae = GrooVAE(hparams = self.pretrained_hparams)
        torch_utils.load_checkpoint(self.pretrained_checkpoint_dir, pretrained_vae)
        
        #if trainable_embeddings is False:  
        #    for p in pretrained_vae.parameters():
        #        p.requires_grad = False
        #self.pretrained_vae = pretrained_vae
        
        self.pretrained_models.append(pretrained_vae)
        
        
        self.global_step = 0
        self.best_val_loss = np.inf
        self.max_seq_len=hparams.max_seq_len
        self.sampling_probability = sampling_probability
        #self.constrain_outputs = hparams.get('constrain_outputs') is True
        #self.constrain_hits = hparams.get('constrain_hits') is True
        self.hparams = hparams
        
        self.enc = BottomUpEncoder(hparams=self.pretrained_hparams,
                                   pretrained_models=self.pretrained_models,
                                   trainable_embeddings=self.trainable_embeddings).to(device)
        self.dec = BottomUpDecoder(hparams=self.pretrained_hparams,
                                   pretrained_models=self.pretrained_models,
                                   trainable_embeddings=self.trainable_embeddings).to(device)
        
        self.to(device)
        
    def forward(self, src, targets=None,sampling_probability=None, length=None):
        src = src.float().to(device)
        mu, logvar, z, q_z = self.enc(src)
        p = sampling_probability if sampling_probability is not None else self.sampling_probability #todo clean this
        max_seq_len = length if length is not None else self.max_seq_len
        #outputs, raw_outputs, embedding_outputs = self.dec(
        outputs, raw_outputs = self.dec(
            z=z,targets=targets,sampling_probability=p,length=max_seq_len)
        return q_z, outputs, raw_outputs#, embedding_outputs


class EmbeddingVAE(nn.Module):
    def __init__(self, hparams, sampling_probability=0):
        super().__init__()
        
        self.hparams = hparams
        
        # Load pretrained VAE model
        #pretrained_vae = GrooVAE(hparams = self.pretrained_hparams)
        #torch_utils.load_checkpoint(self.pretrained_checkpoint_dir, pretrained_vae)
        
        self.global_step = 0
        self.best_val_loss = np.inf
        self.max_seq_len=hparams.max_seq_len
        self.sampling_probability = sampling_probability
        #self.constrain_outputs = hparams.get('constrain_outputs') is True
        #self.constrain_hits = hparams.get('constrain_hits') is True
        self.hparams = hparams
        
        self.enc = EmbeddingEncoder(hparams=hparams).to(device)
        self.dec = EmbeddingDecoder(hparams=hparams).to(device)
        
        self.to(device)
        
    def forward(self, src, targets=None,sampling_probability=None, length=None):
        src = src.float().to(device)
        mu, logvar, z, q_z = self.enc(src)
        p = sampling_probability if sampling_probability is not None else self.sampling_probability #todo clean this
        max_seq_len = length if length is not None else self.max_seq_len
        #outputs, raw_outputs, embedding_outputs = self.dec(
        outputs, raw_outputs = self.dec(
            z=z,targets=targets,sampling_probability=p,length=max_seq_len)
        return q_z, outputs, raw_outputs#, embedding_outputs
"""
        
        
############# END MAIN VAE MODELS #############



##### CLASSIFICATION MODELS #####

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, device, hid_dim=100, dropout=0.5):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.output_layer = nn.Linear(hid_dim, output_dim)

    def forward(self, x):
        # input dim: [batch size, input_dim], e.g. [128, 864]
        x = self.linear1(x)
        x = self.dropout(x)
        preds = self.output_layer(x)
        return preds
     
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # input dim: [batch size, input_dim], e.g. [128, 864]
        preds = self.output_layer(x)
        return preds
    
    
##### END CLASSIFICATION MODELS #####