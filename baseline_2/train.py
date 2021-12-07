import numpy as np
import pandas as pd
import pretty_midi
import collections
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import sys
import json

def get_df_meta(datadir: str):
    files = collections.defaultdict(list)
    for filepath in datadir.glob('*.mid'):
        files['file'].append(str(filepath))
        composer = filepath.stem.split('_')[0]
        files['composer'].append(composer)

        pm = pretty_midi.PrettyMIDI(str(filepath))
        files['end_time'].append(pm.get_end_time())


        tempos, probabilities = pm.estimate_tempi()
        assert np.isclose(sum(probabilities), 1)
        tempo_bpm = np.dot(tempos, probabilities) # expected tempo in beats/min
        seconds_per_beat = (1/tempo_bpm)*60
        time_sig_denom = pm.time_signature_changes[0].denominator
        note_dist = seconds_per_beat / (16 / time_sig_denom)
        note_dist *= 2

        files['expected_tempo'].append(tempo_bpm)
        files['sampling_note_duration'].append(note_dist)
        roll = pm.get_piano_roll(fs=1/note_dist)
        files['roll_length'].append(roll.shape[1])

    df_meta = pd.DataFrame({ key: np.asarray(val) for key, val in files.items() })
    return df_meta


#
# dataset
#
class PianoRoll(Dataset):
    
    def __init__(self, df_meta: pd.DataFrame, seq_length: int = 25, max_windows=None):
        self.df_meta = df_meta.copy()

        
        n_windows = self.df_meta['roll_length'].values - seq_length - 1
        if max_windows is not None:
            n_windows[n_windows > max_windows] = max_windows
            
        self.df_meta['n_windows'] = n_windows
        self.df_meta['n_windows'] = self.df_meta['n_windows'].astype(int)
        
        file_idx_ends = []
        n_windows = self.df_meta['n_windows'].values
        file_idx_ends = [n_windows[0] - 1]
        for windows_in_file in n_windows[1:]:
            file_idx_ends.append(windows_in_file + file_idx_ends[-1] )

        self.df_meta['file_idx_ends'] = file_idx_ends
        self.seq_length = seq_length
        
        self.roll_cache = {}
        
    def __len__(self):
        return self.df_meta['n_windows'].sum()
    
    def __getitem__(self, idx):
        file_idx = self.get_file_idx(idx)
        window_idx = self.get_window_idx(idx)
        
        seq, label = self.get_rolls(file_idx, window_idx)
        
        seq = torch.from_numpy(seq).float()
        label = torch.from_numpy(label).float()
        return seq, label
    
    def get_file_idx(self, idx):
        file_idx = None
        file_idx_ends = self.df_meta['file_idx_ends'].values
        for i in range(len(file_idx_ends)):
            if idx <= file_idx_ends[i]:
                file_idx = i
                break
        if file_idx is None:
            raise ValueError(f'file_idx could not be found for {idx=}')
        return file_idx
    
    def get_window_idx(self, idx):
        file_idx = self.get_file_idx(idx)
        file_idx_ends = self.df_meta['file_idx_ends'].values
        if file_idx == 0:
            idx_start = 0
        else:
            idx_start = file_idx_ends[file_idx - 1]
            
        window_idx = int(idx - idx_start)
        return window_idx
    
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
            note_dist = self.df_meta.iloc[file_idx]['sampling_note_duration']
            roll = self.midi_to_pianoroll(file_path, sample_dist=note_dist)
            self.roll_cache[file_idx] = roll
            
        roll[roll != 0] = 1
        roll = roll.T
        roll_window = roll[window_idx:window_idx+self.seq_length+1, :]
        
        seq = roll_window[:-1]
        # seq[seq == 0] = -1
        label = roll_window[-1]
        return seq, label


#
# model
#
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
        # xnorm = self.norm(x)
        output, (h_n, c_n) = self.lstm(x)
        linear_input = output[:, -1, :]
        normed_linear_input = self.norm(linear_input)
        left_output = self.pitch_layer(normed_linear_input)
        return left_output

if __name__ == "__main__":
    print('i am running!', flush=True)

    config_file_path = sys.argv[1]
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    
    torch.manual_seed(0)

    # set filepaths
    
    _datadir = config['DATA_DIR']
    _datadir = Path(_datadir)
    df_meta = get_df_meta(_datadir)

    output_dir = Path(config['OUTPUT_DIR'])
    if not output_dir.exists():
        output_dir.mkdir()
    metrics_file = output_dir / 'metrics.csv'

    seq_length = config['SEQUENCE_LENGTH']
    learning_rate = config['LEARN_RATE']
    batch_size = config['BATCH_SIZE']
    num_workers = 0
    num_epochs = config['NUM_EPOCHS']
    out_interval = config['OUT_INTERVAL']
    hidden_size = config['HIDDEN_SIZE']

    df_chpn = df_meta[df_meta['composer'] == 'chpn']
    rng = np.random.default_rng(12345)
    idx = np.arange(df_chpn.shape[0])
    n_train = int(0.8*idx.shape[0])
    train_idx = rng.choice(idx, size=n_train, replace=False)
    test_idx = idx[~np.in1d(idx, train_idx)]
    df_train = df_chpn.iloc[train_idx]
    df_test = df_chpn.iloc[test_idx]

    dset_train = PianoRoll(df_meta=df_train, 
                        seq_length=seq_length)
    
    try:
        tmp=config['N_ITERS']
        if tmp is not None:
            n_iters=tmp
        else:
            n_iters = len(dset_train)*num_epochs
    except KeyError:
        n_iters = len(dset_train)*num_epochs
    


    dset_test = PianoRoll(df_meta=df_test, 
                        seq_length=seq_length,
                        max_windows=60)


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

    while iter_idx < n_iters:
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
        if iter_idx % out_interval == 0:

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
                print(f'iter_idx={iter_idx}, {key}={val[-1]}', flush=True)

            # save model
            state_file = output_dir / f'model_weights_iter{iter_idx}.pth'
            torch.save(model.state_dict(), state_file)