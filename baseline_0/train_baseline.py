import numpy as np
import pandas as pd
import pretty_midi
import collections
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn


class PianoRoll(Dataset):
    
    def __init__(self, df_meta: pd.DataFrame, seq_length: int = 25, 
                 batch_size: int = 20, batch_per_file: int = 1):
        self.df_meta = df_meta
        
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.batch_per_file = batch_per_file
        self.idx_per_file = self.batch_size*self.batch_per_file
        
        self.roll_cache = {}
        
    def __len__(self):
        return self.df_meta.shape[0]*self.idx_per_file
    
    def __getitem__(self, idx):
        file_idx = idx // self.idx_per_file
        window_idx = idx % self.idx_per_file
        
        
        seq, label = self.get_rolls(file_idx, window_idx)
        
        seq = torch.from_numpy(seq).float()
        label = torch.from_numpy(label).float()
        return seq, label
    
    def midi_to_pianoroll(self, file, sample_dist=0.02):
        pm = pretty_midi.PrettyMIDI(file)
        
        sampling_rate = 1/sample_dist
        piano_roll = pm.get_piano_roll(fs=sampling_rate)
        return piano_roll
    
    def get_rolls(self, file_idx, window_idx):
        file_path = self.df_meta.iloc[file_idx]['file']
        
        if file_idx in self.roll_cache:
            roll = self.roll_cache[file_idx]
        else:
            note_dist = df_meta.iloc[file_idx]['16th_note_duration']
            roll = self.midi_to_pianoroll(file_path, sample_dist=note_dist)
            self.roll_cache[file_idx] = roll
            
        roll[roll != 0] = 1
        roll = roll.T
        
        seq = roll[:-1]
        label = roll[-1]

        if seq.shape[0] != 100:
            print(file_idx, window_idx)

        return seq, label

class PianoRollLSTM(nn.Module):
    def __init__(self, separate=True):
        super(PianoRollLSTM, self).__init__()
        
        self.separate = separate
        
        if separate:
            input_size = 256
        else:
            input_size = 128
            
        hidden_size = input_size // 2
            
        self.lstm = nn.LSTM(input_size=input_size, batch_first=True, num_layers=1, hidden_size=hidden_size)
        
        self.left_pitch_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        
        if self.separate:
            self.right_pitch_layer = nn.Sequential(
                nn.Linear(hidden_size, input_size),
                nn.Sigmoid()
            )
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
         
        left_output = self.left_pitch_layer(h_n)
        
        if not self.separate:
            return left_output
        else:
            right_output = self.right_pitch_layer(h_n)
            return left_output, right_output

if __name__ == "__main__":

    # set filepaths
    print('i am running!', flush=True)
    _reporoot = Path('/home/ian/projects/music-rnn')
    # _reporoot = Path('/net/dali/home/mscbio/icd3/music-rnn')
    _datadir = _reporoot / 'data' / 'classical'
    _metadata_file = _datadir / 'metadata.csv'
    output_dir = Path('./')
    if not output_dir.exists():
        output_dir.mkdir()
    metrics_file = output_dir / 'metrics.csv'

    # fix filepaths in metadata according to given filepaths
    df_meta = pd.read_csv(_metadata_file)
    def process_path(row, _datadir=_datadir):
        fp = Path(row['file'])
        new_fp = _datadir / fp.name
        return str(new_fp)
    df_meta['file'] = df_meta.apply(process_path, axis=1)

    batch_per_file = 625
    seq_length = 100
    learning_rate = 1e-3
    batch_size = 8
    num_workers = 0

    df_chpn = df_meta[df_meta['composer'] == 'chpn']
    rng = np.random.default_rng(12345)
    idx = np.arange(df_chpn.shape[0])
    n_train = int(0.8*idx.shape[0])
    train_idx = rng.choice(idx, size=n_train, replace=False)
    test_idx = idx[~np.in1d(idx, train_idx)]
    df_train = df_chpn.iloc[train_idx]
    df_test = df_chpn.iloc[test_idx]

    dset_train = PianoRoll(df_meta=df_train, 
                        batch_size=batch_size, 
                        batch_per_file=batch_per_file,
                        seq_length=seq_length)


    dset_test = PianoRoll(df_meta=df_test, 
                        batch_size=batch_size, 
                        batch_per_file=20,
                        seq_length=seq_length)


    train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PianoRollLSTM(separate=False)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    metrics = collections.defaultdict(list)

    iter_idx = -1
    train_iterator = iter(train_dataloader)

    train_losses = []

    while iter_idx < 1000:
        iter_idx += 1
        print(f'iter_idx = {iter_idx}', flush=True)
        
        try:
            features, labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            features, labels = next(train_iterator)
        

        # compute prediction and loss
        pred = model(features)[0, :, :]
        loss = loss_fn(pred, labels)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())


        # compute metrics every 10 iterations
        if iter_idx > 0 and iter_idx % 50 == 0:

            metrics['iter'].append(iter_idx)

            # compute train loss
            train_loss = np.mean(np.asarray(train_losses))
            metrics['train_loss'].append(train_loss)
            train_losses = []


            # test loop
            test_loss_fn = nn.BCELoss()
            test_loss = 0
            frac_notes_correct = 0
            frac_frames_correct = 0
            num_batches = len(test_dataloader)
            for features, labels in test_dataloader:
                pred = model(features)[0, :, :]
                test_loss += test_loss_fn(pred, labels).item()

                notes = (pred > 0.5).type(torch.float)
                equal = torch.eq(notes, labels)
                frac_notes_correct += torch.mean(torch.sum(equal, axis=1) / 128)
                frac_frames_correct += torch.sum(torch.all(equal, axis=1)) / batch_size

            frac_notes_correct /= num_batches
            frac_frames_correct = frac_frames_correct / num_batches
            test_loss /= num_batches

            metrics['test_loss'].append(test_loss)
            metrics['frac_notes_correct'].append(frac_notes_correct)
            metrics['frac_frames_correct'].append(frac_frames_correct)
            
            # save metrics
            df_metrics = pd.DataFrame({ key: np.asarray(val) for key, val in metrics.items() })
            df_metrics.to_csv(metrics_file)

            # save model
            torch.save(model.state_dict(), f'model_weights_iter{iter_idx}.pth')
