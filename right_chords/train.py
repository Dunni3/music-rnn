import numpy as np
import pandas as pd
import pretty_midi
import collections
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

def get_preprocessed_files(processed_data_dir):

    preprocessed_data = collections.defaultdict(list)
    for midi_dir in processed_data_dir.glob('*/'):
        if not midi_dir.is_dir():
            continue
        piece_name = midi_dir.name
        chord_file_name = f'{piece_name}_right_C_full.npy'
        data_file = midi_dir / chord_file_name
        if not data_file.exists():
            print(data_file)
            raise Exception('shit is fucked!!')

        # record piece name and the path of the file containing the full chord data
        preprocessed_data['right_full_chord_file'].append(str(data_file))
        preprocessed_data['piece_name'].append(piece_name)

        # compute the size of the file after 
        chord_roll = np.load(data_file)
        preprocessed_data['chord_roll_size'].append(chord_roll.shape[0])

    df_preprocess = pd.DataFrame({ key: np.asarray(val) for key, val in preprocessed_data.items() })
    return df_preprocess

class FullChord(Dataset):
    
    def __init__(self, df_meta: pd.DataFrame, seq_length: int = 25, 
                 batch_size: int = 20, batch_per_file=None):
        self.df_meta = df_meta.copy()

        self.df_meta['n_batches'] = (self.df_meta['chord_roll_size'] - seq_length )*0.95 // batch_size
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
    
    def get_rolls(self, file_idx, window_idx):
        file_path = self.df_meta.iloc[file_idx]['right_full_chord_file']
        
        roll = np.load(file_path)
        roll_window = roll[window_idx:window_idx+self.seq_length+1, :]
        
        seq = roll_window[:-1]
        label = roll_window[-1]
        return seq, label
    
    
class ChordLSTM(nn.Module):
    def __init__(self, vocab_size=1848):
        super(ChordLSTM, self).__init__()
            
        hidden_size = vocab_size // 8
            
        self.lstm = nn.LSTM(input_size=vocab_size, batch_first=True, num_layers=1, hidden_size=hidden_size)
        
        self.predict_layer = nn.Sequential(
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=2)
        )
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
         
        output = self.predict_layer(h_n)
        
        return output

if __name__ == "__main__":
    torch.manual_seed(0)

    # set filepaths
    _reporoot = Path('/home/ian/projects/music-rnn')
    processed_data_dir = _reporoot / 'data/chopin_processed_bin'
    output_dir = Path('./')
    if not output_dir.exists():
        output_dir.mkdir()
    metrics_file = output_dir / 'metrics.csv'

    # hyperparameters
    seq_length = 100
    learning_rate = 1e-3
    batch_size = 8
    num_workers = 0
    n_iters = 1000
    output_interval = 20

    df_preprocess = get_preprocessed_files(processed_data_dir)

    rng = np.random.default_rng(12345)
    idx = np.arange(df_preprocess.shape[0])
    n_train = int(0.8*idx.shape[0])
    train_idx = rng.choice(idx, size=n_train, replace=False)
    test_idx = idx[~np.in1d(idx, train_idx)]
    df_train = df_preprocess.iloc[train_idx]
    df_test = df_preprocess.iloc[test_idx]

    dset_train = FullChord(df_meta=df_train, 
                        batch_size=batch_size,
                        batch_per_file=None,
                        seq_length=seq_length)


    dset_test = FullChord(df_meta=df_test, 
                        batch_size=batch_size, 
                        batch_per_file=20,
                        seq_length=seq_length)


    train_dataloader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ChordLSTM()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    metrics = collections.defaultdict(list)

    iter_idx = -1
    train_iterator = iter(train_dataloader)

    train_losses = []

    while iter_idx < n_iters:
        iter_idx += 1
        # print(f'{iter_idx=}', flush=True)

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
        if iter_idx > 0 and iter_idx % output_interval == 0:

            metrics['iter'].append(iter_idx)

            # compute train loss
            train_loss = np.mean(np.asarray(train_losses))
            metrics['train_loss'].append(train_loss)
            train_losses = []


            # test loop
            test_loss_fn = nn.CrossEntropyLoss()
            test_loss = 0
            frames_correct = 0
            num_batches = len(test_dataloader)
            model.eval()
            with torch.no_grad():
                for features, labels in test_dataloader:
                    pred = model(features)[0, :, :]
                    test_loss += test_loss_fn(pred, labels).item()

                    pred_chords = pred.argmax(axis=1)
                    label_chords = labels.argmax(axis=1) 
                    equal = torch.eq(pred_chords, label_chords)
                    frames_correct += torch.sum(equal)
            model.train()

            frac_frames_correct = frames_correct / (num_batches*batch_size)
            avg_test_loss = test_loss / num_batches

            metrics['test_loss'].append(avg_test_loss)
            metrics['frac_frames_correct'].append(frac_frames_correct)

            # save metrics
            df_metrics = pd.DataFrame({ key: np.asarray(val) for key, val in metrics.items() })
            df_metrics.to_csv(metrics_file)

            for key, val in metrics.items():
                print(f'iter_idx={iter_idx}, {key}={val[-1]}')

            # save model
            state_file = output_dir / f'model_weights_iter{iter_idx}.pth'
            torch.save(model.state_dict(), state_file)