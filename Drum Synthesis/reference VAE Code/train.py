Skip to content
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@jrgillick 
jrgillick
/
groove
Private
1
0
0
Code
Issues
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
groove/train.py /
@jrgillick
jrgillick Check in code for gesture training
Latest commit e056c06 29 days ago
 History
 1 contributor
1087 lines (893 sloc)  57.8 KB
  
from paths import *
from midi_utils import *
from models import *
import torch_utils
import numpy as np, IPython, importlib, librosa
import argparse, os, time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from magenta.models.music_vae import data as music_vae_data

import sys
sys.path.append(audio_feature_learning_path)
import audio_utils
from functools import partial

standard_drumroll_converter = configs.CONFIG_MAP['pt_drumroll_2bar'].data_converter
standard_drumperf_converter = configs.CONFIG_MAP['pt_perf_2bar_drums'].data_converter

#python train.py --config=pt_drumroll_2bar --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --checkpoint_dir=/data/jrgillick/checkpoints/groove/drumroll_2bar_l2

# python train.py --config=pt_drumroll_2bar_constrain_outputs --checkpoint_dir=/data/jrgillick/checkpoints/groove/drumroll_2bar_mask_hieraug_samp08_drop05_constrain_all_tst --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True

# python train.py --config=pt_drumroll_1step_smaller_mb02_fb12 --checkpoint_dir=/data/jrgillick/checkpoints/groove/pt_drumroll_1step_smaller_mb02_fb12 --augment_fn=None --sampling_probability=0.8 --mask_hits=True --window_size=1

# python train.py --config=pt_perf_2bar_drums --checkpoint_dir=/data/jrgillick/checkpoints/groove/perf_2bar_drums_hieraug_samp0_drop05 --augment_fn=hierarchical_augment --sampling_probability=0.0 --training_type=perf

# python train.py --config=pt_groove_2bar_32nds --training_type=groove --checkpoint_dir=/data/jrgillick/checkpoints/groove/pt_groove_32nds_mb002 --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True

# python train.py --config=pt_groove_2bar_64ths --training_type=groove --checkpoint_dir=/data/jrgillick/checkpoints/groove/pt_groove_64ths_mb002 --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --training_steps=150000

# python train.py --config=pt_embedding_drumroll_2bar_1step_z32 --training_type=embedding --checkpoint_dir=/data/jrgillick/checkpoints/groove/embedding_1_32 --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True

#python train.py --config=pt_embedding_drumroll_2bar_1step_z32 --training_type=embedding --checkpoint_dir=/data/jrgillick/checkpoints/groove/embeddings_noise_test_1_32 --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --embeddings_test=True

# python train.py --config=pt_embedding_drumroll_2bar_1step_z32 --training_type=multilevel --checkpoint_dir=/data/jrgillick/checkpoints/groove/multi_1_32_end_to_end_mb002 --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --max_beta=0.002 --training_steps=150000 

# python train.py --config=pt_embedding_drumroll_2bar_1step_z32 --training_type=multilevel --checkpoint_dir=/data/jrgillick/checkpoints/groove/multi_1_32_pretrained_frozen_mb002 --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --max_beta=0.002 --training_steps=150000 --use_pretrained_subsequence_vae=True --freeze_subsequence_vae=True

# python train.py --config=pt_embedding_drumroll_2bar_1step_z32 --training_type=multilevel --checkpoint_dir=/data/jrgillick/checkpoints/groove/multi_1_32_pretrained_trainable_mb002 --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --max_beta=0.002 --training_steps=150000 --use_pretrained_subsequence_vae=True

# python train.py --config=pt_embedding_drumroll_2bar_1step_z32 --training_type=multilevel --checkpoint_dir=/data/jrgillick/checkpoints/groove/multi_1_32_pretrained_finetune_lr01_mb002 --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --max_beta=0.002 --training_steps=150000 --use_pretrained_subsequence_vae=True --pretrained_small_lr=True

#python train.py --config=pt_drumroll_2bar --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --max_beta=0.0002 --regression_loss_type=l1 --checkpoint_dir=/data/jrgillick/checkpoints/groove/drumroll_2bar_L1_mb0002

#python train.py --config=pt_drumroll_2bar --augment_fn=hierarchical_augment --sampling_probability=0.8 --mask_hits=True --max_beta=0.00002 --free_bits=48  --checkpoint_dir=/data/jrgillick/checkpoints/groove/drumroll_2bar_mask_hieraug_samp08_drop05_mb00002_fb48_alpha02_mlr1e3 --training_steps=50000 --min_learning_rate=1e-3 --alpha=0.2

parser = argparse.ArgumentParser()
parser.add_argument('--log_path', type=str, default=None)
parser.add_argument('--config', type=str)
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--augment_fn', type=str, default=None)
parser.add_argument('--sampling_probability', type=str, default='0.0')
parser.add_argument('--mask_hits', type=str, default=None)
parser.add_argument('--transfer', type=str, default=None)
parser.add_argument('--fixed_velocities', type=str, default=None)
parser.add_argument('--aux_decoders', type=str, default=None)
parser.add_argument('--training_type', type=str, default='groove')
parser.add_argument('--window_size', type=str, default=None)
parser.add_argument('--embeddings_test', type=str, default=None)
parser.add_argument('--free_bits', type=str, default=None)
parser.add_argument('--max_beta', type=str, default=None)
parser.add_argument('--min_learning_rate', type=str, default=None)
parser.add_argument('--regression_loss_type', type=str, default=None)

parser.add_argument('--alpha', type=str, default=None) # param for augment_fn

parser.add_argument('--training_steps', type=str, default='100000') 


parser.add_argument('--use_pretrained_subsequence_vae', type=str, default=None)
parser.add_argument('--freeze_subsequence_vae', type=str, default=None)
parser.add_argument('--include_embedding_loss', type=str, default=None)


parser.add_argument('--pretrained_small_lr', type=str, default=None)
parser.add_argument('--small_lr_weighting', type=str, default=0.01)

parser.add_argument('--skipping_logging_at_step_1', type=str, default=None)


# Add global argument to train humanize or drumify models
# Rather than implementing directly in the data converters, like we probably should for consistency
# Use the 16th note config for quantizing inputs for humanization
#   more args/options could be added later
parser.add_argument('--train_humanize', type=str, default=None)
parser.add_argument('--train_drumify', type=str, default=None)

args = parser.parse_args()
training_type = args.training_type
augment_fn = get_augment_fn(args.augment_fn)
checkpoint_dir = args.checkpoint_dir
sampling_probability = float(args.sampling_probability)
mask_hits = False if args.mask_hits is None else True
window_size = None if args.window_size is None else int(args.window_size)
transfer = False if args.transfer is None else True
fixed_velocities = False if args.fixed_velocities is None else True
skipping_logging_at_step_1 = False if args.skipping_logging_at_step_1 is None else True
aux_decoders = False if args.aux_decoders is None else True
config = configs.CONFIG_MAP[args.config]
data_converter = config.data_converter
hparams = config.hparams

embeddings_test = False if args.embeddings_test is None else True
training_steps = int(args.training_steps)

drumroll_training = config.hparams.get('drumroll_style') is True

use_pretrained_subsequence_vae = False if args.use_pretrained_subsequence_vae is None else True
freeze_subsequence_vae = False if args.freeze_subsequence_vae is None else True
include_embedding_loss = False if args.include_embedding_loss is None else True
pretrained_small_lr = False if args.pretrained_small_lr is None else True
small_lr_weighting = args.small_lr_weighting

train_humanize = False if args.train_humanize is None else True
train_drumify = False if args.train_drumify is None else True

if args.alpha is not None:
    augment_fn=partial(augment_fn, alpha=float(args.alpha))
    print(augment_fn)


if args.free_bits is not None:
    hparams.free_bits = float(args.free_bits)
if args.max_beta is not None:
    hparams.max_beta = float(args.max_beta)
if args.min_learning_rate is not None:
    hparams.min_learning_rate = float(args.min_learning_rate)
    

bce_criterion = torch.nn.BCEWithLogitsLoss()    
mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
cross_entropy_criterion = torch.nn.CrossEntropyLoss(reduction='none')

if args.regression_loss_type is not None:
    if args.regression_loss_type == 'l1':
        regression_criterion = l1_criterion
        print("Training with L1 Loss")
else:
    regression_criterion = mse_criterion
    print("Training with L2 Loss")

if train_humanize:
    score_converter=music_vae_data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, humanize=True)
    tap_converter=None

elif train_drumify:
    score_converter=None
    if drumroll_training:
        tap_converter = data_converter
    elif training_type == 'groove' or training_type == 'perf':
        tap_converter = music_vae_data.GrooveConverter(
            split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
            max_tensors_per_notesequence=20, tapify=True, fixed_velocities=False) #fixed_velocities=True
    
elif transfer:
    humanize_config = configs.CONFIG_MAP['pt_groove_2bar_humanize']
    if fixed_velocities:
        tap_config = configs.CONFIG_MAP['pt_groove_2bar_drumify_fixed_velocities']
    else:
        tap_config = configs.CONFIG_MAP['pt_groove_2bar_drumify']
    score_hparams = humanize_config.hparams
    groove_hparams = tap_config.hparams
    score_converter=music_vae_data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, humanize=True)
    tap_converter = music_vae_data.GrooveConverter(
        split_bars=2, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20, tapify=True, fixed_velocities=True)
    
else:
    score_converter = None
    tap_converter = None

def perf_collate_fn(seq_label_tuples):
    sequences, targets = audio_utils.unpack_list_of_tuples(seq_label_tuples)
    sequence_lengths = np.array([len(s) for s in sequences])
    target_lengths = np.array([len(s) for s in targets])
    
    max_seq_len = max(sequence_lengths)
    max_trg_len = max(target_lengths)
    max_len = max(max_seq_len, max_trg_len)

    # make them not one-hots for padding
    sequences = [seq.argmax(axis=1) for seq in sequences]
    targets = [target.argmax(axis=1) for target in targets]

    kwargs = {'constant_values': standard_drumperf_converter._depth-1}
    
    padded_sequences = [librosa.util.fix_length(
        seq, max_len, axis=0, **kwargs) for seq in sequences]
    
    padded_targets = [librosa.util.fix_length(
        target, max_len, axis=0, **kwargs) for target in targets]
    
    onehot_seqs = np.array([torch_utils.np_onehot(
        seq, depth=standard_drumperf_converter._depth, dtype=np.int32) for seq in padded_sequences])
    
    onehot_targets = np.array([torch_utils.np_onehot(
        target, depth=standard_drumperf_converter._depth, dtype=np.int32) for target in padded_targets])

    return onehot_seqs, onehot_targets, sequence_lengths, target_lengths
    
    
    
if training_type == 'perf':
    collate_fn = perf_collate_fn
elif training_type == 'groove':
    if window_size:
        collate_fn = torch_utils.collate_windows
    else:
        collate_fn = None
elif training_type == 'embedding':
    collate_fn = None
elif training_type =='multilevel':
    collate_fn = None
else:
    raise Exception(f"Invalid value `{training_type}` for `training_type`")

######### LOAD DATA ###########
# These are midi sequences pre-split into 2 bar windows with 1 bar hop-size
# TODO maybe dev sequences should use 2-bar hop size to be consistent with assumptions in data aug
train_sequences = load_gmd_from_tfds(train_data_path)
dev_sequences = load_gmd_from_tfds(dev_data_path)

# Fix a couple unsupported pitches and remove all control changes
for s in dev_sequences+train_sequences:
    start_notes_at_0(s)
    for n in s.notes:
            if n.pitch==22:
                n.pitch = 42
            if n.pitch==26:
                n.pitch=46
    while(len(s.control_changes)>0):
        s.control_changes.pop()

if transfer:
    dev_dataset = GrooveDataset(dev_sequences, data_converter, augment_fn=None,
                               transfer=True,score_converter=score_converter, tap_converter=tap_converter)
else:
    dev_dataset = GrooveDataset(dev_sequences, data_converter, augment_fn=None, window_size=window_size,
                               score_converter=score_converter, tap_converter=tap_converter,
                               train_humanize=train_humanize, train_drumify=train_drumify)

if args.augment_fn == 'hierarchical_augment':
    # Load presplit sequences for augmentation
    pre_split_train_seqs = load_pre_split_sequences_from_tf_records(train_data_split_path)
    if transfer:
        train_dataset = GrooveSplitDataset(sequence_lists=pre_split_train_seqs, data_converter=data_converter,
                                          transfer=True,score_converter=score_converter, tap_converter=tap_converter)
    else:
        train_dataset = GrooveSplitDataset(sequence_lists=pre_split_train_seqs, data_converter=data_converter,
                                          score_converter=score_converter, tap_converter=tap_converter,
                                          train_humanize=train_humanize, train_drumify=train_drumify)
else:
    if transfer:
        train_dataset = GrooveDataset(train_sequences, data_converter, augment_fn=augment_fn,
                                     transfer=True,score_converter=score_converter, tap_converter=tap_converter)
    else:
        train_dataset = GrooveDataset(train_sequences, data_converter, augment_fn=augment_fn, window_size=window_size,
                                     score_converter=score_converter, tap_converter=tap_converter,
                                     train_humanize=train_humanize, train_drumify=train_drumify)
    
######### Convert data to tensors for training #########
if collate_fn is not None:
    training_generator = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=hparams.batch_size, shuffle=True, collate_fn=collate_fn)
    training_generator_for_eval = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=hparams.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_generator = torch.utils.data.DataLoader(dev_dataset, num_workers=0, batch_size=hparams.batch_size, shuffle=True, collate_fn=collate_fn)
else:
    training_generator = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=hparams.batch_size, shuffle=True)
    training_generator_for_eval = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=hparams.batch_size, shuffle=True)
    dev_generator = torch.utils.data.DataLoader(dev_dataset, num_workers=0, batch_size=hparams.batch_size, shuffle=True)

if training_type=='perf':
    vae = GrooVAE(hparams,
              encoder_class=BidirectionalLSTMEncoder,
              decoder_class=DrumPerfLSTMDecoder,
              sampling_probability=sampling_probability)
    
elif training_type=='embedding':
    # TODO make these args
    #Drumroll_1step_checkpoint_dir = '/data/jrgillick/checkpoints/groove/pt_drumroll_1step_smaller_mb02_fb24'
    config_drumroll_1step = configs.CONFIG_MAP['pt_drumroll_1step_smaller_mb02_fb24'] 
    drumroll_1step_vae = GrooVAE(hparams=config_drumroll_1step.hparams)
    torch_utils.load_checkpoint(drumroll_1step_checkpoint_dir, drumroll_1step_vae, name='best.pth.tar')
    drumroll_1step_vae.to(device)
    drumroll_1step_vae.eval()
    vae = GrooVAE(hparams, encoder_class=EmbeddingEncoder, decoder_class=EmbeddingDecoder)
    #vae = LevelVAE(hparams=config.hparams,
    #                 pretrained_checkpoint_dir=drumroll_1step_checkpoint_dir,
    #                 pretrained_hparams=config_drumroll_1step.hparams,
    #                 trainable_embeddings=True)

elif training_type=='groove':
    if transfer:
        vae = GrooveScore(score_hparams, groove_hparams, sampling_probability=sampling_probability, aux_decoders=aux_decoders)
    else:
        vae = GrooVAE(hparams, sampling_probability=sampling_probability)
        
elif training_type=='multilevel':
    config_drumroll_1step = configs.CONFIG_MAP['pt_drumroll_1step_smaller_mb02_fb24'] 
    config_embedding = configs.CONFIG_MAP['pt_embedding_drumroll_2bar_1step_z32']
    # TODO make it more clear that the `embedding_hparams` takes in the hparams arg here
    vae = MultiLevelVAE(embedding_hparams=hparams, subsequence_hparams=config_drumroll_1step.hparams)
    
    if use_pretrained_subsequence_vae:
        drumroll_1step_vae = GrooVAE(hparams=config_drumroll_1step.hparams)
        torch_utils.load_checkpoint(drumroll_1step_checkpoint_dir, drumroll_1step_vae, name='best.pth.tar')
        drumroll_1step_vae.to(device)
        vae.subsequence_encoder.load_state_dict(drumroll_1step_vae.enc.state_dict())
        vae.subsequence_decoder.load_state_dict(drumroll_1step_vae.dec.state_dict())

optimizer = optim.Adam(vae.parameters())

#vae.apply(torch_utils.init_weights)
torch_utils.count_parameters(vae)

if os.path.exists(checkpoint_dir):
    torch_utils.load_checkpoint(checkpoint_dir, vae, optimizer)
else:
    print("Saving checkpoints to ", checkpoint_dir)

writer = SummaryWriter(checkpoint_dir) # Create writer after creating checkpoint dir



def run_evals(model, generator, batch_limit=None, mode='train', calc_drumroll_metrics=False,
              training_type='groove', calc_perf_loss=True, embeddings_test=False, squared_err=0.):
    model.eval()
    sampling_prob = None if mode is 'train' else 1. # set sampling probability to 1 in 'eval' mode. i.e. running dev evals, not train evals
    with torch.no_grad():
        all_losses = []
        all_r_losses = []
        all_kl_losses = []
        all_kl_bits = []
        
        all_h_losses = []; all_v_losses = []; all_o_losses = []
        primary_h_losses = []; primary_v_losses = []; primary_o_losses = []
        secondary_h_losses = []; secondary_v_losses = []; secondary_o_losses = []
        
        predicted_hits = []; true_hits = []
        primary_predicted_hits = []; secondary_predicted_hits = []
        primary_true_hits = []; secondary_true_hits = []
        
        all_aux_losses = []; all_s_losses = []; all_g_v_losses = []; all_g_o_losses = []
        
        all_perf_losses = []
        
        all_embedding_losses = []
        all_subsequence_kl_losses = []
        all_subsequence_kl_bits = []
        
        for j, batch in enumerate(generator):
            if batch_limit is not None and j > batch_limit:
                break
                
            if training_type=='perf':
                x, targets, seq_lens, trg_lens = batch
                x = torch.from_numpy(x).float().to(device)            
                targets = torch.from_numpy(targets).float().to(device).permute(1,0,2)
                optimizer.zero_grad()
                q_z, outputs, raw_outputs = vae(src=x, targets=targets)
                kl_loss, kl_bits = kl_loss_beta_vae(q_z,hparams.free_bits,hparams.beta_rate,
                                                        1,hparams.max_beta)
                flat_raw_outputs = raw_outputs.permute(1,0,2).contiguous().view(-1,hparams.output_dim)
                flat_targets = targets.permute(1,0,2).contiguous().view(-1,hparams.output_dim).argmax(dim=1)
                flat_mask = (flat_targets!=hparams.output_dim-1).float() # dim-1 is the <PAD> index
                perf_loss = cross_entropy_criterion(flat_raw_outputs, flat_targets)
                perf_loss = (perf_loss*flat_mask).sum()/(flat_mask.sum()) # avg the loss over the nonpadded inputs
                perf_loss = perf_loss.cpu().numpy()
                
                all_perf_losses.append(perf_loss)
                
                
                # Convert back to NoteSequences and then into groove representation to compute the groove losses
                true_trgs = targets.permute(1,0,2).detach().cpu().numpy()
                true_notesequences = [standard_drumperf_converter._to_notesequences([true_trgs[index]])[0] for index in range(true_trgs.shape[0])]
                true_drumroll_tensors = np.array([standard_drumroll_converter._to_tensors(s).outputs[0] for s in true_notesequences])
                #true_seq_lens = [len(t) for t in true_drumroll_tensors]
                #import pdb; pdb.set_trace()
                
                #perf_outs = outputs.permute(1,0,2).detach().cpu().numpy()
                perf_outs = torch_utils.torch_one_hot(outputs.permute(1,0).contiguous(), device).cpu().numpy()
                
                pred_notesequences = [standard_drumperf_converter._to_notesequences([perf_outs[index]])[0] for index in range(perf_outs.shape[0])]
                # try to convert back to tensors.. if there are no notes, then just add a blank tensor
                pred_drumroll_tensors = []
                for s in pred_notesequences:
                    if len(s.notes)>0:
                        pred_drumroll_tensors.append(standard_drumroll_converter._to_tensors(s).outputs[0])
                    else:
                        pred_drumroll_tensors.append(np.zeros_like(true_drumroll_tensors[0]))
                pred_drumroll_tensors = np.array(pred_drumroll_tensors)
                    
                # Now put all the vars into the state they'd be if they came out from Groove training, so we can run evals
                outputs = torch.from_numpy(pred_drumroll_tensors)
                raw_outputs = outputs
                targets = torch.from_numpy(true_drumroll_tensors)
                t_hits, t_vels, t_offs = torch.split(targets, int(targets.shape[-1]/3), dim=-1)

            if training_type in ['groove','embedding','multilevel']:
                # Run model forward
                if transfer:
                    x_score, x_groove, targets = batch
                else:
                    x, targets = batch
                    x = x.float().to(device)

                targets = targets.float().to(device).permute(1,0,2)
                t_hits, t_vels, t_offs = torch.split(targets, int(targets.shape[-1]/3), dim=-1)

                if transfer:
                    if aux_decoders:
                        s_targets = x_score.float().to(device).permute(1,0,2)
                        g_targets = x_groove.float().to(device).permute(1,0,2)
                        q_z_score, q_z_groove, outputs, raw_outputs, \
                        score_outputs, score_raw_outputs, groove_outputs, groove_raw_outputs = vae(
                            src_score=x_score, src_groove=x_groove, targets=targets,
                            score_targets=s_targets, groove_targets=g_targets)
                        s_hits, s_vels, s_offs = torch.split(score_raw_outputs, int(score_outputs.shape[-1]/3), dim=-1)
                        g_hits, g_vels, g_offs = torch.split(groove_raw_outputs, int(groove_outputs.shape[-1]/3), dim=-1)
                        score_t_hits, score_t_vels, score_t_offs = torch.split(s_targets, int(x_score.shape[-1]/3), dim=-1)
                        groove_t_hits, groove_t_vels, groove_t_offs = torch.split(g_targets, int(x_groove.shape[-1]/3), dim=-1)
                        s_loss = bce_criterion(s_hits, score_t_hits)
                        g_v_loss = regression_criterion(g_vels*groove_t_hits, groove_t_vels)
                        g_o_loss = regression_criterion(g_offs*groove_t_hits, groove_t_offs)
                        aux_loss = s_loss + g_v_loss + g_o_loss
                        all_aux_losses.append(float(aux_loss.detach()))
                        all_s_losses.append(float(s_loss.detach()))
                        all_g_v_losses.append(float(g_v_loss.detach()))
                        all_g_o_losses.append(float(g_o_loss.detach()))
                    else:
                        q_z_score, q_z_groove, outputs, raw_outputs = model(
                            src_score=x_score, src_groove=x_groove, targets=targets, sampling_probability=sampling_prob)
                else:
                    if training_type == 'embedding':
                        # Run embeddings model to get predict subsequence embeddings
                        x_embeddings = torch_utils.encode_to_subsequence_embeddings(
                            x, drumroll_1step_vae.enc, sequence_length=32).detach().float().to(device)
                        target_embeddings = torch_utils.encode_to_subsequence_embeddings( # TODO avoid permuting targets back and forth
                            targets.permute(1,0,2), drumroll_1step_vae.enc, sequence_length=32).detach().float().to(device).permute(1,0,2)

                        q_z, embedding_outputs, _ = model(src=x_embeddings, targets=target_embeddings)
                        
                        if embeddings_test:
                            noise = torch.from_numpy(np.random.normal(0,np.sqrt(squared_err),x_embeddings.shape)).float().to(device)
                            q_z, embedding_outputs, _ = model(src=x_embeddings, targets=target_embeddings)
                            embedding_outputs = (x_embeddings + noise).permute(1,0,2)
                        
                        kl_loss, kl_bits = kl_loss_beta_vae(q_z,hparams.free_bits,hparams.beta_rate,
                                                                1,hparams.max_beta)
                        embedding_loss = regression_criterion(embedding_outputs, target_embeddings)
                        all_embedding_losses.append(float(embedding_loss.detach()))
                        # Run subsequence model on predicted embeddings to get final reconstruction
                        outputs, raw_outputs = torch_utils.decode_from_subsequence_embeddings(
                            embedding_outputs, drumroll_1step_vae.dec)
                    elif training_type == 'multilevel':
                        h = model(src=x, targets=targets) #q_z, outputs, raw_outputs 
                        raw_outputs = h['raw_output_tensors']
                        outputs = h['output_tensors']
                        q_z = h['q_z_embedding']
                    else:
                        q_z, outputs, raw_outputs = model(src=x, sampling_probability=sampling_prob)                    
                    
                if calc_perf_loss:
                    # Convert back to NoteSequences and then into performance representation to compute the performance-LM-loss
                    outs = outputs.permute(1,0,2).detach().cpu().numpy()#[:,0:32,0:90]
                    pred_notesequences = [standard_drumroll_converter._to_notesequences([outs[index]])[0] for index in range(outs.shape[0])] 
                    pred_drumperf_tensors = [standard_drumperf_converter._to_tensors(
                        s).outputs[0] for s in pred_notesequences]
                    true_trgs = targets.permute(1,0,2).detach().cpu().numpy()
                    true_notesequences = [standard_drumroll_converter._to_notesequences(
                        [true_trgs[index]])[0] for index in range(true_trgs.shape[0])]
                    true_drumperf_tensors = [standard_drumperf_converter._to_tensors(
                        s).outputs[0] for s in true_notesequences]
                    pred_target_tups = [(pred_drumperf_tensors[i],
                                         true_drumperf_tensors[i]) for i in range(len(pred_drumperf_tensors))]

                    padded_preds, padded_trgs, perf_pred_lens, perf_trg_lens = perf_collate_fn(pred_target_tups)
                    padded_preds = torch.from_numpy(padded_preds).float().to(device)
                    padded_trgs = torch.from_numpy(padded_trgs).float().to(device).permute(1,0,2)
                    flat_perf_outputs = padded_preds.permute(1,0,2).contiguous().view(
                        -1,standard_drumperf_converter._depth)
                    flat_perf_targets = padded_trgs.permute(1,0,2).contiguous().view(
                        -1,standard_drumperf_converter._depth).argmax(dim=1)
                    flat_perf_mask = (flat_perf_targets!=standard_drumperf_converter._depth-1).float() # dim-1 is the <PAD> index
                    perf_loss = cross_entropy_criterion(flat_perf_outputs, flat_perf_targets)
                    perf_loss = (perf_loss*flat_perf_mask).sum()/(flat_perf_mask.sum()) # avg the loss over the nonpadded inputs
                    perf_loss = perf_loss.detach().cpu().numpy()

                    all_perf_losses.append(perf_loss)

                if transfer:
                    kl_loss_score, kl_bits_score = kl_loss_beta_vae(q_z_score,score_hparams.free_bits/2,score_hparams.beta_rate,
                                                        1,score_hparams.max_beta)
                    kl_loss_groove, kl_bits_groove = kl_loss_beta_vae(q_z_groove,groove_hparams.free_bits/2,groove_hparams.beta_rate,
                                                        1,groove_hparams.max_beta)
                    kl_loss = kl_loss_score + kl_loss_groove
                    kl_bits = kl_bits_score + kl_bits_groove
                else:
                    kl_loss, kl_bits = kl_loss_beta_vae(q_z,hparams.free_bits,hparams.beta_rate,
                                                        1,hparams.max_beta)                

            if training_type == 'multilevel':
                raw_hits, _, _ =  torch.split(raw_outputs, int(outputs.shape[-1]/3), dim=-1)
                hits, vels, offs = torch.split(outputs, int(outputs.shape[-1]/3), dim=-1)
                        
                # Loss (1)
                kl_loss, kl_bits = kl_loss_beta_vae(q_z,hparams.free_bits,hparams.beta_rate,
                                                    1,hparams.max_beta)
                all_kl_losses.append(float(kl_loss.detach()))
                all_kl_bits.append(float(kl_bits.detach()))

                # Loss (2)
                true_hits.append(t_hits.detach().int())
                predicted_hits.append(hits.detach().int())

                # Get loss for H,V,O
                h_loss = bce_criterion(raw_hits, t_hits)
                v_loss = regression_criterion(vels*t_hits, t_vels) # TODO: we always mask hits during eval. Want or not?
                o_loss = regression_criterion(offs*t_hits, t_offs) # TODO: we always mask hits during eval. Want or not?
                r_loss = h_loss + v_loss + o_loss
                all_h_losses.append(float(h_loss.detach()))
                all_v_losses.append(float(v_loss.detach()))
                all_o_losses.append(float(o_loss.detach()))
                all_r_losses.append(float(r_loss.detach()))

                # Loss (3)
                embedding_loss = regression_criterion(h['output_embeddings'], h['target_embeddings'])
                all_embedding_losses.append(float(embedding_loss.detach()))

                # Loss (4)
                subsequence_kl_losses = []
                subsequence_kl_bits = []
                for q_z_subsequence in  h['q_z_subsequences']:
                    sub_kl_loss, sub_kl_bits = kl_loss_beta_vae(
                    q_z_subsequence,vae.subsequence_hparams.free_bits,
                    vae.subsequence_hparams.beta_rate, 1,
                    vae.subsequence_hparams.max_beta)
                    subsequence_kl_losses.append(sub_kl_loss)
                    subsequence_kl_bits.append(sub_kl_bits)
                subsequence_kl_loss = torch.mean(torch.stack(subsequence_kl_losses))
                subsequence_kl_bits = torch.mean(torch.stack(subsequence_kl_bits))
                
                all_subsequence_kl_losses.append(float(subsequence_kl_loss.detach()))
                all_subsequence_kl_bits.append(float(subsequence_kl_bits.detach()))
                    
                loss = kl_loss + subsequence_kl_loss + r_loss
                if include_embedding_loss: loss += embedding_loss 

                all_losses.append(float(loss.detach()))
            
            else:
                raw_hits, _, _ =  torch.split(raw_outputs, int(outputs.shape[-1]/3), dim=-1)
                hits, vels, offs = torch.split(outputs, int(outputs.shape[-1]/3), dim=-1)

                true_hits.append(t_hits.detach().int())
                predicted_hits.append(hits.detach().int())

                # Get loss for H,V,O
                h_loss = bce_criterion(raw_hits, t_hits)
                v_loss = regression_criterion(vels*t_hits, t_vels) # TODO: we always mask hits during eval. Want or not?
                o_loss = regression_criterion(offs*t_hits, t_offs) # TODO: we always mask hits during eval. Want or not?
                all_h_losses.append(float(h_loss.detach()))
                all_v_losses.append(float(v_loss.detach()))
                all_o_losses.append(float(o_loss.detach()))

                # Get loss for reconstruction, KL, and total 
                r_loss = h_loss + v_loss + o_loss
                if training_type == 'perf':
                    loss = perf_loss + kl_loss.cpu().numpy()
                    all_losses.append(float(loss))
                else:
                    loss = r_loss + kl_loss
                if transfer and aux_decoders:
                    loss = loss + aux_loss
                all_r_losses.append(float(r_loss.detach()))
                all_kl_losses.append(float(kl_loss.detach()))
                all_kl_bits.append(float(kl_bits.detach()))
                if training_type != 'perf':
                    all_losses.append(float(loss.detach()))
            
            # If relevant, split off primary (main)  h,v,o from secondary(drumroll) and calc separate metrics
            if calc_drumroll_metrics:
                inst_indices = [sum(standard_drumroll_converter.inst_lim[0:i]) for i in range(1, len(standard_drumroll_converter.inst_lim))]
                
                raw_h_primary, raw_h_secondary = torch_utils.get_primary_and_secondary_matrix(raw_hits.cpu(), inst_indices)
                
                pred_h_primary, pred_h_secondary = torch_utils.get_primary_and_secondary_matrix(hits.cpu(), inst_indices)
                trg_h_primary, trg_h_secondary = torch_utils.get_primary_and_secondary_matrix(t_hits.cpu(), inst_indices)
                
                primary_true_hits.append(trg_h_primary.detach().int())
                primary_predicted_hits.append(pred_h_primary.detach().int())
                
                secondary_true_hits.append(trg_h_secondary.detach().int())
                secondary_predicted_hits.append(pred_h_secondary.detach().int())
                
                pred_v_primary, pred_v_secondary = torch_utils.get_primary_and_secondary_matrix(vels.cpu(), inst_indices)
                trg_v_primary, trg_v_secondary = torch_utils.get_primary_and_secondary_matrix(t_vels.cpu(), inst_indices)
                
                pred_o_primary, pred_o_secondary = torch_utils.get_primary_and_secondary_matrix(offs.cpu(), inst_indices)
                trg_o_primary, trg_o_secondary = torch_utils.get_primary_and_secondary_matrix(t_offs.cpu(), inst_indices)
                
                primary_h_loss = bce_criterion(raw_h_primary, trg_h_primary)
                secondary_h_loss = bce_criterion(raw_h_secondary, trg_h_secondary)
                
                primary_v_loss = regression_criterion(pred_v_primary*trg_h_primary, trg_v_primary)
                secondary_v_loss = regression_criterion(pred_v_secondary*trg_h_secondary, trg_v_secondary)
                
                primary_o_loss = regression_criterion(pred_o_primary*trg_h_primary, trg_o_primary)
                secondary_o_loss = regression_criterion(pred_o_secondary*trg_h_secondary, trg_o_secondary)
                
                primary_h_losses.append(float(primary_h_loss.detach()))
                secondary_h_losses.append(float(secondary_h_loss.detach()))
                primary_v_losses.append(float(primary_v_loss.detach()))
                secondary_v_losses.append(float(secondary_v_loss.detach()))
                primary_o_losses.append(float(primary_o_loss.detach()))
                secondary_o_losses.append(float(secondary_o_loss.detach()))
            
         
        r_loss = np.mean(all_r_losses)
        kl_loss = np.mean(all_kl_losses)
        kl_bits = np.mean(all_kl_bits)
        loss = np.mean(all_losses)
        h_loss = np.mean(all_h_losses)
        v_loss = np.mean(all_v_losses)
        o_loss = np.mean(all_o_losses)
        
        if transfer and aux_decoders:
            aux_loss = np.mean(all_aux_losses)
            s_loss = np.mean(all_s_losses)
            g_v_loss = np.mean(all_g_v_losses)
            g_o_loss = np.mean(all_g_o_losses)
        
        if calc_drumroll_metrics:
            primary_h_loss = np.mean(primary_h_losses)
            secondary_h_loss = np.mean(secondary_h_losses)
            primary_v_loss = np.mean(primary_v_losses)
            secondary_v_loss = np.mean(secondary_v_losses)
            primary_o_loss = np.mean(primary_o_losses)
            secondary_o_loss = np.mean(secondary_o_losses)
        
        hits_accuracy, hits_precision, hits_recall, hits_f1 = torch_utils.calc_f1_metrics(
            predicted_hits, true_hits)
        
        if calc_drumroll_metrics:
            primary_accuracy, primary_precision, primary_recall, primary_f1 = torch_utils.calc_f1_metrics(
                primary_predicted_hits, primary_true_hits)
            secondary_accuracy, secondary_precision, secondary_recall, secondary_f1 = torch_utils.calc_f1_metrics(
                secondary_predicted_hits, secondary_true_hits) 
        
        if mode == 'eval':
            current_loss = loss
            is_best = current_loss < model.best_val_loss
            if is_best:
                model.best_val_loss = current_loss
            state = torch_utils.make_state_dict(model, #optimizer=optimizer, 
                global_step=model.global_step, best_val_loss=model.best_val_loss)
            torch_utils.save_checkpoint(state, is_best=is_best, checkpoint_dir=checkpoint_dir)

        print(f"{mode.upper()} RESULTS:")
        print(f"Step: {model.global_step}")
        print(f"Loss: {loss} | R_Loss: {r_loss}")
        print(f"KL_Loss: {kl_loss}")
        print(f"KL_Bits: {kl_bits}")
        
        print(f"Hits Accuracy: {hits_accuracy}")
        if calc_drumroll_metrics:
            print(f"Primary Hits Accuracy: {primary_accuracy}")
            print(f"Secondary Hits Accuracy: {secondary_accuracy}")
            
        print(f"Precision: {hits_precision}")
        if calc_drumroll_metrics:
            print(f"Primary Hits Precision: {primary_precision}")
            print(f"Secondary Hits Precision: {secondary_precision}")
        
        print(f"Recall: {hits_recall}")
        if calc_drumroll_metrics:
            print(f"Primary Hits Recall: {primary_recall}")
            print(f"Secondary Hits Recall: {secondary_recall}")
            
        print(f"F1: {hits_f1}")
        if calc_drumroll_metrics:
            print(f"Primary F1: {primary_f1}")
            print(f"Secondary F1: {secondary_f1}")
        
        print(f"Hits Loss: {h_loss}")
        if calc_drumroll_metrics:
            print(f"Primary h_loss: #{primary_h_loss}")
            print(f"Secondary h_loss: #{secondary_h_loss}")
            
        print(f"Velocities Loss: {v_loss}")
        if calc_drumroll_metrics:
            print(f"Primary Velocities Loss: #{primary_v_loss}")
            print(f"Secondary Velocities Loss: #{secondary_v_loss}")
            
        print(f"Offsets Loss: {o_loss} ")        
        if calc_drumroll_metrics:
            print(f"Primary Offsets Loss: #{primary_o_loss}")
            print(f"Secondary Offsets Loss: #{secondary_o_loss}")
        
        
        if not skipping_logging_at_step_1 or model.global_step > 1:
            writer.add_scalar(f"loss/{mode}", loss, model.global_step)
            writer.add_scalar(f"r_loss/{mode}", r_loss, model.global_step)
            writer.add_scalar(f"h_loss/{mode}", h_loss, model.global_step)
            writer.add_scalar(f"v_loss/{mode}", v_loss, model.global_step)
            writer.add_scalar(f"o_loss/{mode}", o_loss, model.global_step)
            writer.add_scalar(f"hits_accuracy/{mode}", hits_accuracy, model.global_step)
            writer.add_scalar(f"hits_precision/{mode}", hits_precision, model.global_step)
            writer.add_scalar(f"hits_recall/{mode}", hits_recall, model.global_step)
            writer.add_scalar(f"hits_f1/{mode}", hits_f1, model.global_step)
            writer.add_scalar(f"kl_loss/{mode}", kl_loss, model.global_step)
            writer.add_scalar(f"kl_bits/{mode}", kl_bits, model.global_step)

            try:
                writer.add_scalar("learning_rate", vae.lr, model.global_step)
            except:
                pass

            if transfer and aux_decoders:
                writer.add_scalar(f"aux_loss/{mode}", aux_loss, model.global_step)
                writer.add_scalar(f"aux_hits_loss/{mode}", s_loss, model.global_step)
                writer.add_scalar(f"aux_velocities_loss/{mode}", g_v_loss, model.global_step)
                writer.add_scalar(f"aux_offsets_loss/{mode}", g_o_loss, model.global_step)

            if calc_drumroll_metrics:
                writer.add_scalar(f"primary_h_loss/{mode}", primary_h_loss, model.global_step)
                writer.add_scalar(f"secondary_h_loss/{mode}", secondary_h_loss, model.global_step)
                writer.add_scalar(f"primary_v_loss/{mode}", primary_v_loss, model.global_step)
                writer.add_scalar(f"secondary_v_loss/{mode}", secondary_v_loss, model.global_step)
                writer.add_scalar(f"primary_o_loss/{mode}", primary_o_loss, model.global_step)
                writer.add_scalar(f"secondary_o_loss/{mode}", secondary_o_loss, model.global_step)
                writer.add_scalar(f"primary_accuracy/{mode}", primary_accuracy, model.global_step)
                writer.add_scalar(f"primary_precision/{mode}", primary_precision, model.global_step)
                writer.add_scalar(f"primary_recall/{mode}", primary_recall, model.global_step)
                writer.add_scalar(f"primary_f1/{mode}", primary_f1, model.global_step)
                writer.add_scalar(f"secondary_accuracy/{mode}", secondary_accuracy, model.global_step)
                writer.add_scalar(f"secondary_precision/{mode}", secondary_precision, model.global_step)
                writer.add_scalar(f"secondary_recall/{mode}", secondary_recall, model.global_step)
                writer.add_scalar(f"secondary_f1/{mode}", secondary_f1, model.global_step)

            if calc_perf_loss:
                perf_loss = np.mean(all_perf_losses)
                print(f"Perf Loss: {perf_loss} ")  
                writer.add_scalar(f"perf_loss/{mode}", perf_loss, model.global_step)

            if training_type in ['embedding', 'multilevel']:
                embedding_loss = np.mean(all_embedding_losses)
                print(f"Embedding Loss: {embedding_loss}")
                writer.add_scalar(f"embedding_loss/{mode}", embedding_loss, model.global_step)

            if training_type == 'multilevel':
                subsequence_kl_loss = np.mean(all_subsequence_kl_losses)
                print(f"Subsequence KL Loss: {subsequence_kl_loss}")
                writer.add_scalar(f"subsequence_kl_loss/{mode}", subsequence_kl_loss, model.global_step)
                subsequence_kl_bits = np.mean(all_subsequence_kl_bits)
                print(f"Subsequence KL Bits: {subsequence_kl_bits}")
                writer.add_scalar(f"subsequence_kl_bits/{mode}", subsequence_kl_bits, model.global_step)

        print('\n')
            

######### RUN TRAINING LOOP ######

#def run_batch(vae, batch, device, optimizer, transfer=False):

def run_groove_training_loop():
    while vae.global_step < training_steps:
        for batch in training_generator:
            vae.train()
            torch.cuda.empty_cache()
            t0 = time.time()
            i = vae.global_step

            # Update learning rate every 20 steps
            if i % 20 == 0:
                lr = ((hparams.learning_rate - hparams.min_learning_rate) *
                  np.power(hparams.decay_rate, vae.global_step) + hparams.min_learning_rate)
                optimizer.lr = lr
                vae.lr=lr

            if transfer:
                x_score, x_groove, targets = batch
            else:
                x, targets = batch
                x = x.float().to(device)

            targets = targets.float().to(device).permute(1,0,2)
            t_hits, t_vels, t_offs = torch.split(targets, int(targets.shape[-1]/3), dim=-1)
            optimizer.zero_grad()

            if transfer:
                if aux_decoders:
                    s_targets = x_score.float().to(device).permute(1,0,2)
                    g_targets = x_groove.float().to(device).permute(1,0,2)
                    q_z_score, q_z_groove, outputs, raw_outputs, \
                    score_outputs, score_raw_outputs, groove_outputs, groove_raw_outputs = vae(
                        src_score=x_score, src_groove=x_groove, targets=targets,
                        score_targets=s_targets, groove_targets=g_targets)
                    s_hits, s_vels, s_offs = torch.split(score_raw_outputs, int(score_outputs.shape[-1]/3), dim=-1)
                    g_hits, g_vels, g_offs = torch.split(groove_raw_outputs, int(groove_outputs.shape[-1]/3), dim=-1)
                    score_t_hits, score_t_vels, score_t_offs = torch.split(s_targets, int(x_score.shape[-1]/3), dim=-1)
                    groove_t_hits, groove_t_vels, groove_t_offs = torch.split(g_targets, int(x_groove.shape[-1]/3), dim=-1)
                    s_loss = bce_criterion(s_hits, score_t_hits)
                    g_v_loss = regression_criterion(g_vels*groove_t_hits, groove_t_vels)
                    g_o_loss = regression_criterion(g_offs*groove_t_hits, groove_t_offs)
                    aux_loss = s_loss + g_v_loss + g_o_loss
                else:
                    q_z_score, q_z_groove, outputs, raw_outputs = vae(src_score=x_score, src_groove=x_groove, targets=targets)
            else:
                q_z, outputs, raw_outputs = vae(src=x, targets=targets)


            hits, vels, offs = torch.split(raw_outputs, int(outputs.shape[-1]/3), dim=-1)
            #pred = torch.cat([nn.Sigmoid()(hits), vels, offs], dim=-1).permute(1,0,2)

            if transfer:
                kl_loss_score, kl_bits_score = kl_loss_beta_vae(q_z_score,score_hparams.free_bits/2,score_hparams.beta_rate,
                                                    1,score_hparams.max_beta)
                kl_loss_groove, kl_bits_groove = kl_loss_beta_vae(q_z_groove,groove_hparams.free_bits/2,groove_hparams.beta_rate,
                                                    1,groove_hparams.max_beta)
                kl_loss = kl_loss_score + kl_loss_groove
                #kl_bits = kl_bits_score + kl_bits_groove
            else:
                kl_loss, kl_bits = kl_loss_beta_vae(q_z,hparams.free_bits,hparams.beta_rate,
                                                    1,hparams.max_beta)

            h_loss = bce_criterion(hits, t_hits)

            if mask_hits:
                v_loss = regression_criterion(vels*t_hits, t_vels)
                o_loss = regression_criterion(offs*t_hits, t_offs)                
            else:
                v_loss = regression_criterion(vels, t_vels)
                o_loss = regression_criterion(offs, t_offs)

            r_loss = h_loss + v_loss + o_loss
            loss = r_loss + kl_loss
            if transfer and aux_decoders:
                loss = loss + aux_loss
            loss.backward()
            print(vae.global_step)
            #torch_utils.check_zero_grads(vae)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.)
            optimizer.step(); vae.global_step += 1
            print(time.time()-t0)

            # Run evaluations every 100 steps
            if i % 100 == 0:
                run_evals(vae, training_generator_for_eval, batch_limit=3, mode='train', 
                          calc_drumroll_metrics=drumroll_training)
                run_evals(vae, dev_generator, batch_limit=5, mode='eval',
                          calc_drumroll_metrics=drumroll_training)

def run_multilevel_training_loop():
    print("Got there...")
    if freeze_subsequence_vae:
        optimizer = optim.Adam([p for p in vae.embedding_encoder.parameters()] + [p for p in vae.embedding_decoder.parameters()])
    else:
       optimizer = optim.Adam(vae.parameters())
    lr = ((hparams.learning_rate - hparams.min_learning_rate) *
         np.power(hparams.decay_rate, vae.global_step) + hparams.min_learning_rate)
    if pretrained_small_lr:
        optimizer = optim.Adam([{'params': vae.embedding_encoder.parameters(), 'lr': lr},
            {'params': vae.embedding_decoder.parameters(), 'lr': lr},
            {'params':  vae.subsequence_encoder.parameters(), 'lr': lr*small_lr_weighting},
            {'params':  vae.subsequence_decoder.parameters(), 'lr': lr*small_lr_weighting}
        ])
    else:
        optimizer.lr = lr
        vae.lr=lr

    while vae.global_step < training_steps:
        for batch in training_generator:
            vae.train()
            torch.cuda.empty_cache()
            t0 = time.time()
            i = vae.global_step

            # Update learning rate every 20 steps
            if i % 20 == 0:
                lr = ((hparams.learning_rate - hparams.min_learning_rate) *
                  np.power(hparams.decay_rate, vae.global_step) + hparams.min_learning_rate)
                if pretrained_small_lr:
                    optimizer = optim.Adam([{'params': vae.embedding_encoder.parameters(), 'lr': lr},
                                      {'params': vae.embedding_decoder.parameters(), 'lr': lr},
                                      {'params':  vae.subsequence_encoder.parameters(), 'lr': lr*small_lr_weighting},
                                      {'params':  vae.subsequence_decoder.parameters(), 'lr': lr*small_lr_weighting}
            ])
                else:
                    optimizer.lr = lr
                    vae.lr=lr

            x, targets = batch
            x = x.float().to(device)
            targets = targets.float().to(device).permute(1,0,2)
            t_hits, t_vels, t_offs = torch.split(targets, int(targets.shape[-1]/3), dim=-1)
            optimizer.zero_grad()

            # Losses: 
            # 1. kl_loss
            # 2. r_loss
            # 3. embedding_loss
            # 4. subsequence_kl_loss
            
            h = vae(src=x, targets=targets) #q_z, outputs, raw_outputs 
            raw_outputs = h['raw_output_tensors']
            outputs = h['output_tensors']
            
            
            hits, vels, offs = torch.split(raw_outputs, int(outputs.shape[-1]/3), dim=-1)
            pred = torch.cat([nn.Sigmoid()(hits), vels, offs], dim=-1).permute(1,0,2)
            q_z = h['q_z_embedding']

            # Loss (1)
            kl_loss, kl_bits = kl_loss_beta_vae(q_z,hparams.free_bits,hparams.beta_rate,
                                                    1,hparams.max_beta)

            # Loss (2)
            h_loss = bce_criterion(hits, t_hits)

            if mask_hits:
                v_loss = regression_criterion(vels*t_hits, t_vels)
                o_loss = regression_criterion(offs*t_hits, t_offs)                
            else:
                v_loss = regression_criterion(vels, t_vels)
                o_loss = regression_criterion(offs, t_offs)

            r_loss = h_loss + v_loss + o_loss
            
            # Loss (3)
            embedding_loss = regression_criterion(h['output_embeddings'], h['target_embeddings'])
            
            # Loss (4)
            subsequence_kl_losses = []
            subsequence_kl_bits = []
            for q_z_subsequence in  h['q_z_subsequences']:
                sub_kl_loss, sub_kl_bits = kl_loss_beta_vae(
                q_z_subsequence,vae.subsequence_hparams.free_bits,
                vae.subsequence_hparams.beta_rate, 1,
                vae.subsequence_hparams.max_beta)
                subsequence_kl_losses.append(sub_kl_loss)
                subsequence_kl_bits.append(sub_kl_bits)
            subsequence_kl_loss = torch.mean(torch.stack(subsequence_kl_losses))
            subsequence_kl_bits = torch.mean(torch.stack(subsequence_kl_bits))
            
            loss = kl_loss + subsequence_kl_loss + r_loss
            if include_embedding_loss: loss += embedding_loss 


            loss.backward()
            print(vae.global_step)
            #torch_utils.check_zero_grads(vae)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.)
            optimizer.step(); vae.global_step += 1
            print(time.time()-t0)

            # Run evaluations every 100 steps
            if i % 100 == 0:
                run_evals(vae, training_generator_for_eval, batch_limit=3, mode='train', 
                          calc_drumroll_metrics=drumroll_training,
                          training_type='multilevel')
                run_evals(vae, dev_generator, batch_limit=5, mode='eval',
                          calc_drumroll_metrics=drumroll_training,
                          training_type='multilevel')
                
                
def run_drumperf_training_loop():
    while vae.global_step < training_steps:
        for batch in training_generator:
            vae.train()
            torch.cuda.empty_cache()
            t0 = time.time()
            i = vae.global_step

            # Update learning rate every 20 steps
            if i % 20 == 0:
                lr = ((hparams.learning_rate - hparams.min_learning_rate) *
                  np.power(hparams.decay_rate, vae.global_step) + hparams.min_learning_rate)
                optimizer.lr = lr
                vae.lr=lr

            x, targets, seq_lens, trg_lens = batch
            
            x = torch.from_numpy(x).float().to(device)            
            targets = torch.from_numpy(targets).float().to(device).permute(1,0,2)
            #targets = torch.from_numpy(targets).float().to(device) # don't permute for cross-entropy loss- nvmd
            
            optimizer.zero_grad()

            q_z, outputs, raw_outputs = vae(src=x, targets=targets)
            
            kl_loss, kl_bits = kl_loss_beta_vae(q_z,hparams.free_bits,hparams.beta_rate,
                                                    1,hparams.max_beta)
            
            #import pdb; pdb.set_trace()  #TODO: Flatten, mask, and compute cross-entropy loss

            flat_raw_outputs = raw_outputs.permute(1,0,2).contiguous().view(-1,hparams.output_dim)
            flat_targets = targets.permute(1,0,2).contiguous().view(-1,hparams.output_dim).argmax(dim=1)
            flat_mask = (flat_targets!=hparams.output_dim-1).float() # dim-1 is the <PAD> index
            r_loss = cross_entropy_criterion(flat_raw_outputs, flat_targets)
            r_loss = (r_loss*flat_mask).sum()/(flat_mask.sum()) # avg the loss over the nonpadded inputs
            
            #r_loss = torch_utils.masked_bce_sequence_loss(raw_outputs, targets, seq_lens)
            
            loss = r_loss + kl_loss
            loss.backward()
            print(vae.global_step)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.)
            optimizer.step(); vae.global_step += 1
            print(time.time()-t0)
            # Run evaluations every 100 steps
            if i % 100 == 0:
                run_evals(vae, training_generator_for_eval, batch_limit=3, mode='train', 
                          calc_drumroll_metrics=True,#drumroll_training,
                          training_type='perf')
                run_evals(vae, dev_generator, batch_limit=5, mode='eval',
                          calc_drumroll_metrics=True,#drumroll_training)
                          training_type='perf')
                
def run_embedding_training_loop():
    while vae.global_step < training_steps:
        for batch in training_generator:
            vae.train()
            torch.cuda.empty_cache()
            t0 = time.time()
            i = vae.global_step

            # Update learning rate every 20 steps
            if i % 20 == 0:
                lr = ((hparams.learning_rate - hparams.min_learning_rate) *
                  np.power(hparams.decay_rate, vae.global_step) + hparams.min_learning_rate)
                optimizer.lr = lr
                vae.lr=lr

            x, targets = batch
            x_embeddings = torch_utils.encode_to_subsequence_embeddings(x, drumroll_1step_vae.enc, sequence_length=32).detach().float().to(device)
            target_embeddings = torch_utils.encode_to_subsequence_embeddings(
                targets, drumroll_1step_vae.enc, sequence_length=32).detach().float().to(device).permute(1,0,2)

            optimizer.zero_grad()
            q_z, outputs, raw_outputs = vae(src=x_embeddings, targets=target_embeddings)
            kl_loss, kl_bits = kl_loss_beta_vae(q_z,hparams.free_bits,hparams.beta_rate,
                                                    1,hparams.max_beta)
            embedding_loss = regression_criterion(outputs, target_embeddings)
            
            r_loss = embedding_loss
            loss = r_loss + kl_loss
            loss.backward()
            print(vae.global_step)
            print(f"R loss: {r_loss}")
            print(f"KL loss: {kl_loss}")
                  
            #torch_utils.check_zero_grads(vae)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.)
            optimizer.step(); vae.global_step += 1
            print(time.time()-t0)
   
            # Run evaluations every 100 steps
            if i % 100 == 0:
                run_evals(vae, training_generator_for_eval, batch_limit=3, mode='train', 
                          calc_drumroll_metrics=drumroll_training,
                          training_type='embedding')
                run_evals(vae, dev_generator, batch_limit=5, mode='eval',
                          calc_drumroll_metrics=drumroll_training,
                          training_type='embedding')
                
if __name__ == '__main__':
    if embeddings_test:
        squared_errs = np.arange(0,1,step=0.01)
        for squared_err in tqdm(squared_errs):
            print(f"Squared err: {squared_err}")
            vae.global_step = squared_err*100
            run_evals(vae, dev_generator, batch_limit=5, mode='eval',
                              calc_drumroll_metrics=drumroll_training,
                              training_type='embedding', embeddings_test=True, squared_err=squared_err)
    if training_type == 'groove':
        run_groove_training_loop()
    elif training_type == 'perf':
        run_drumperf_training_loop()
    elif training_type == 'embedding':
        run_embedding_training_loop()
    elif training_type == 'multilevel':
        run_multilevel_training_loop()
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
