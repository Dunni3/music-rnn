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

def get_preprocessed_files(processed_data_dir):

    preprocessed_data = collections.defaultdict(list)
    for midi_dir in processed_data_dir.glob('*/'):
        if not midi_dir.is_dir():
            continue
        piece_name = midi_dir.name
        
        chord_file_name = f'{piece_name}_pianoRoll.csv'
        data_file = midi_dir / chord_file_name
        if not data_file.exists():
            raise Exception('shit is fucked!!')

        # record piece name and the path of the file containing the full chord data
        preprocessed_data['file'].append(str(data_file))
        preprocessed_data['piece_name'].append(piece_name)

        # compute the size of the file after 
        piano_roll = pd.read_csv(data_file, index_col=0).values
        preprocessed_data['roll_length'].append(piano_roll.shape[0])

    df_preprocess = pd.DataFrame({ key: np.asarray(val) for key, val in preprocessed_data.items() })
    return df_preprocess


class PreprocessedPianoRoll(Dataset):
    
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
        
        seq, label = self.get_rolls(file_idx, window_idx, idx)
        
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
    
    def get_rolls(self, file_idx, window_idx, idx):
        file_path = self.df_meta.iloc[file_idx]['file']
        
        if file_idx in self.roll_cache:
            roll = self.roll_cache[file_idx]
        else:
            roll = pd.read_csv(file_path, index_col=0).values
            roll[roll != 0] = 1
            self.roll_cache[file_idx] = roll
            
        roll_window = roll[window_idx:window_idx+self.seq_length+1, :]
        if roll_window.shape[0] != self.seq_length + 1:
            print(f'{roll_window.shape[0]=}')
            raise Exception(f'fuck {idx=}')
        
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
        # xnorm = self.norm(x)
        output, (h_n, c_n) = self.lstm(x)
        linear_input = output[:, -1, :]
        normed_linear_input = self.norm(linear_input)
        left_output = self.pitch_layer(normed_linear_input)
        return left_output

if __name__ == "__main__":
    print('i am running', flush=True)
    config_file_path = sys.argv[1]
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    output_dir = Path(config['OUTPUT_DIR'])
    if not output_dir.exists():
        output_dir.mkdir()
    metrics_file = output_dir / 'metrics.csv'

    _datadir = config['PREPROCESSED_DATA_DIR']
    _datadir = Path(_datadir)
    df_meta = get_preprocessed_files(_datadir)
    print('data fetched', flush=True)

    g4_mask = df_meta['piece_name'].str.contains('G4')
    df_train = df_meta[~g4_mask]
    df_test = df_meta[g4_mask]

    # seq_length = 12
    # learning_rate = 3e-4
    # batch_size = 8
    # num_workers = 0
    # n_iters = 135*50
    # out_interval = 2000
    # hidden_size = 60
    seq_length = config['SEQUENCE_LENGTH']
    learning_rate = config['LEARN_RATE']
    batch_size = config['BATCH_SIZE']
    num_workers = 0
    num_epochs = config['NUM_EPOCHS']
    out_interval = config['OUT_INTERVAL']
    hidden_size = config['HIDDEN_SIZE']

    dset_train = PreprocessedPianoRoll(df_meta=df_train, 
                        seq_length=seq_length,
                        max_windows=None)

    print(f'training dataset size = {len(dset_train)}')

    try:
        tmp=config['N_ITERS']
        if tmp != "":
            n_iters=tmp
        else:
            n_iters = len(dset_train)*num_epochs
    except KeyError:
        n_iters = len(dset_train)*num_epochs


    dset_test = PreprocessedPianoRoll(df_meta=df_test, 
                        seq_length=seq_length,
                        max_windows=20)


    train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    model = PianoRollLSTM(hidden_size=hidden_size)
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
            
        # features = torch.zeros(features.shape)


        # compute prediction and loss
        # pred = model(features)[0, :, :]
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
                    # pred = model(features)[0, :, :]
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
