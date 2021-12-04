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
                 batch_size: int = 20, batch_per_file=None):
        self.df_meta = df_meta.copy()

        self.df_meta['n_batches'] = (self.df_meta['roll_length'] - seq_length )*0.95 // batch_size
        self.df_meta['n_batches'] = self.df_meta['n_batches'].astype(int)
        file_idx_ends = []
        n_batches = self.df_meta['n_batches'].values
        file_idx_ends = [n_batches[0]*batch_size - 1]
        for batches_in_file in n_batches[1:]:
            file_idx_ends.append(batches_in_file*batch_size + file_idx_ends[-1] )

        self.df_meta['file_idx_ends'] = file_idx_ends

        
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.batch_per_file = batch_per_file

        if self.batch_per_file is not None:
            self.idx_per_file = self.batch_size*self.batch_per_file
        else:
            self.idx_per_file = None
        
        self.roll_cache = {}
        
    def __len__(self):
        if self.batch_per_file is None:
            return int(self.df_meta['n_batches'].sum()*self.batch_size)
        else:
            return self.df_meta.shape[0]*self.idx_per_file
    
    def __getitem__(self, idx):
        if self.batch_per_file is None:
            file_idx_ends = self.df_meta['file_idx_ends'].values
            for i in range(len(file_idx_ends)):
                if idx <= file_idx_ends[i]:
                    file_idx = i
                    break
            if file_idx == 0:
                idx_start = 0
            else:
                idx_start = file_idx_ends[file_idx - 1]
            window_idx = int(idx - idx_start)
        else:
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
            note_dist = self.df_meta.iloc[file_idx]['16th_note_duration']
            roll = self.midi_to_pianoroll(file_path, sample_dist=note_dist)
            self.roll_cache[file_idx] = roll
            
        roll[roll != 0] = 1
        roll = roll.T
        roll_window = roll[window_idx:window_idx+self.seq_length+1, :]
        
        seq = roll_window[:-1]
        label = roll_window[-1]
        return seq, label

class PianoRollLSTM(nn.Module):
    def __init__(self, hidden_size=64):
        super(PianoRollLSTM, self).__init__()
        
        input_size=128
        self.hidden_size = hidden_size
            
        self.lstm = nn.LSTM(input_size=input_size, batch_first=True, num_layers=1, hidden_size=hidden_size)
        
        self.norm = nn.BatchNorm1d(num_features=hidden_size)
        
        self.pitch_layer = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        linear_input = output[:, -1, :]
        normed_linear_input = self.norm(linear_input)
        left_output = self.pitch_layer(normed_linear_input)
        return left_output

if __name__ == "__main__":
    torch.manual_seed(0)
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

    seq_length = 30
    learning_rate = 3e-4
    batch_size = 8
    num_workers = 0
    n_iters = 10000
    out_interval = 100
    hidden_size = 60

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
                        batch_per_file=None,
                        seq_length=seq_length)


    dset_test = PianoRoll(df_meta=df_test, 
                        batch_size=batch_size, 
                        batch_per_file=20,
                        seq_length=seq_length)


    train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PianoRollLSTM(hidden_size=hidden_size)
    # set model bias so it initially predicts all notes as 0
    model.pitch_layer[0].bias = nn.Parameter(model.pitch_layer[0].bias - 0.24)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    metrics = collections.defaultdict(list)

    iter_idx = -1
    train_iterator = iter(train_dataloader)

    train_losses = []

    while iter_idx < 5000:
        iter_idx += 1
        # print(f'iter_idx = {iter_idx}', flush=True)
        
        try:
            features, labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            features, labels = next(train_iterator)
        

        # compute prediction and loss
        pred = model(features)
        loss = loss_fn(pred, labels)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())


        # compute metrics every 10 iterations
        if iter_idx > 0 and iter_idx % 100 == 0:

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
            with torch.no_grad():
                for features, labels in test_dataloader:
                    pred = model(features)
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

            for key, val in metrics.items():
                print(f'iter_idx={iter_idx}, {key}={val[-1]}')

            # save model
            state_file = output_dir / f'model_weights_iter{iter_idx}.pth'
            torch.save(model.state_dict(), state_file)
