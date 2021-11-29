#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
accept a midi file as input
output annotated format with chords
"""
import numpy as np
import pandas as pd
import pretty_midi as pm 
import pychord 
import matplotlib.pyplot as plt
import os 
import sys
#GLOBALS
TIME_STEP=.02
"""
load raw midi file as pretty MIDI object
"""
def load_midi(path=None):
    if path is None:
        path='../raw/chopin/chp_op18.mid'
    raw= pm.PrettyMIDI(path)
    return raw 
def midi_to_formatted(raw):
    #get smallest duration
    smallest= get_smallest(raw)
    #get piano roll split by this duration 
    raw_piano_roll= raw.get_piano_roll(fs=1/smallest)
    return raw_piano_roll
    #annotate piano roll by listing ties and chords 
    #relabel notes as chord positions with octaves 
    #output labeled 
    pass
"""
function to build dictionary of chords where chords
are a sorted (low to high) set of chromatic note positions (0 to 11)
inputs:
    -chroma: list of all chroma to be used for training, all 
    transposed to C major or C minor and sampled at the minimum note duration 
outputs:
    -chord_dict: dictionary of chords with string representing chord as value
    for chord 034--> 100110000000
    -max_vertical_notes: (int) maximum number of vertical notes at any time point 
    -max_chord_length: (int) maximum number of unique chromatic positions in any chord 
"""
def build_chord_set_and_1h(roll,chord_set=set()):
   
    #replace all nonzeros with ones 
    roll=roll.astype(int)
    roll[roll!=0]=1
    
    roll['chords']= roll.apply(lambda r: encode_chord(r),1)
    roll_1h=pd.get_dummies(roll['chords'])
    roll_chord_set=set(roll['chords'])
    chord_set=chord_set | roll_chord_set
        
        
    return chord_set, roll_1h
def encode_chord(row):
# =============================================================================
#     if sum(row)==0:
#         print(row)
#         sys.exit()
# =============================================================================
    if len(row)!=128 and len(row)!=12:
        print('something is fucked 1')
        
    #row=list(map(int,row))
    if len(row)!=128 and len(row)!=12:
        print('something is fucked 2')
        
    #row[row!=0]=1
    if len(row)!=128 and len(row)!=12:
        print('something is fucked 3')
        
   
    s=str(list(row))
    #print(s)
    s=s[1:-1]
    s=s.replace(', ','')
    #print(s)
    if len(s)!=128 and len(s)!=12:
        print(s)
        print('something is fucked 4')
        sys.exit()
# =============================================================================
#     if s==128*'0':
#         print('rest')
# =============================================================================
    return s
        
"""
function to output piano roll as formatted data where format is:
    pandas dataframe with features as columns where features are:
        - one-hot encoding of all chords in chord_dict 
        -<max_notes>x<max_chord_length> vectors with each containing the octave number 
        of the note with that chord position 
        -<max_notes>x<max_chord_length>*4 vectors to indicate if a note is starting, continuing,
        stopping, or not tied 
"""
def piano_roll_to_formatted(piano_roll_list, chord_dict, max_notes,max_chord_length):
    formatted_data=[]
    for pr in piano_roll_list:
        #convert all notes in each column to 
        #TODO
        pass
    return formatted_data
"""take formatted data (output from RNN) and convert to MIDI"""
def formatted_to_midi(formatted):
    midi=''
    #TODO
    pass
"""transpose a midi file to C and sample at minimum duration
inputs:
    -raw_midi: raw midi file 
outputs:
    -transposed_and_sampled
    
"""
def sample_and_transpose_to_C(raw_midi,timestep):
    #timestep=np.infty
    changes=raw_midi.key_signature_changes
    for ks,i in zip(changes,range(len(changes))):
        #if key is C, continue
        if ks.key_number==0:
            continue
        else:
            diff_from_C= min(0-ks.key_number%12,12-ks.key_number%12)
            assert(ks.key_number+diff_from_C==0)
            for instrument in raw_midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        if note.end < ks.time:
                            continue
                        elif (i<len(changes)-1 and note.end >= changes[i+1].time):
                            break
                        else:
                            note.pitch+=diff_from_C
                           # length=note.get_duration()
                            #if 0<length<timestep:
                            #    timestep=length
                            
        
    
        ks.key_number+=diff_from_C
    transposed_and_sampled=raw_midi
    transposed_and_sampled.write('test.mid')
    timestep=get_smallest(raw_midi)
    piano_roll= transposed_and_sampled.get_piano_roll(fs=1/timestep)
    chroma= transposed_and_sampled.get_chroma(fs=1/timestep)
    return piano_roll,chroma

"""find the smallest subdivision for midi with same time signature"""
def get_smallest(raw):
    smallest=np.infty
    for instrument in raw.instruments:
        for note in instrument.notes:
            length=note.get_duration()
            if 0<length<smallest:
                smallest=length
   # print(smallest)
    if smallest==np.infty:
        smallest=100
    return smallest 
def split_by_timesig(raw,subdiv=None):
    time_sigs=raw.time_signature_changes
    rolls=[]
    for ts in time_sigs:
       rolls.append(raw.get_piano_roll())
        
    pass
def choose_sample_rate(pathToDirectory):
    smallest_list=[]
    tempo_list=[]
    
    for file in os.listdir(pathToDirectory):
        #get midi
        raw= load_midi(f'{pathToDirectory}/{file}')
        smallest_list.append(get_smallest(raw))
        tempo_list.append(raw.estimate_tempo())
    return smallest_list,tempo_list
def estimate_timestep(name=None, path=None):
    #name='Chopin'
    #path='../raw/chopin'
    small,temps=choose_sample_rate(path)
    small.sort()
    time_step=np.quantile(small, 0.1)
    plt.title(f'{name} shortest note durations')
    plt.xlabel('time(s)')
    plt.ylabel('number of pieces')
    plt.hist(small,bins=100)
    plt.axvline(time_step,color='r')
    plt.legend(['chosen timestep','data'])
    plt.savefig(f'{name}_note_durations.png')
    plt.show()
    plt.title(f'{name} tempi')
    plt.xlabel('tempo (bpm)')
    plt.ylabel('number of pieces')
    plt.hist(temps,bins=100)
    plt.show()
    return time_step
def transpose_all_pianoRolls(path,timestep):
    folder_name= path.split('/')[-1]+'_C_pianoRoll'
    abspath= os.path.abspath(path+'/../')
    folder=os.path.join(abspath,folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for file in os.listdir(path):
        raw= load_midi(f'{path}/{file}')
        transposed=sample_and_transpose_to_C(raw,timestep)[0]
        transposed_file= file.split('.')[-2]+'_C.csv'
        f=open(f'{folder}/{transposed_file}','w')
        transposed=pd.DataFrame(transposed.T)
        np.savetxt(f,transposed,delimiter=',')
        
"""
#split into lh and rh 
#for each hand:
    #sample all chorma and prs and transpose to C 
    #save files as csvs (checkpoint 1)
    #get chord sets for all files and 1h encoded time series 
    #insert a zero column for each missing chord 
    #save files in folder called <filename> 
"""        
def preprocess(rawDataPath):
    
    #for each file in rawDataPath:
    folder_name= rawDataPath.split('/')[-1]+'_processed'
    abspath= os.path.abspath(rawDataPath+'/../')
    output_folder=os.path.join(abspath,folder_name)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    full_chord_set_left=set()
    pitch_chord_set_left=set()
    full_chord_set_right=set()
    pitch_chord_set_right=set()
    onehot_left={}
    onehot_right={}
    for file in os.listdir(rawDataPath):
        raw_lh,raw_rh= load_midi(f'{rawDataPath}/{file}'),load_midi(f'{rawDataPath}/{file}')
       
        ins_lh= raw_lh.instruments; ins_rh=raw_rh.instruments
        for ins in ins_lh:
            if ins.name=='Piano left':
                raw_lh.instruments=[ins]
                break
            continue
        for ins in ins_rh:
            if ins.name=='Piano right':
                raw_rh.instruments=[ins]
                break
            continue
        file_folder= file.split('.')[0]
        
        for raw,tag in zip([raw_lh,raw_rh],['left','right']):
            #transpose
            
            trans_pr,trans_ch= sample_and_transpose_to_C(raw,TIME_STEP)
            trans_pr=pd.DataFrame(trans_pr.T); trans_ch=pd.DataFrame(trans_ch.T)
            
            pr_file= file.split('.')[-2]+f'_{tag}_C_pianoRoll.csv'
            ch_file= file.split('.')[-2]+f'_{tag}_C_chroma.csv'
            
            
            #checkpoint
            file_folder_path=os.path.join(output_folder,file_folder)
            
            if not os.path.exists(file_folder_path):
                os.makedirs(file_folder_path)
            
            
            
            trans_pr.to_csv(f'{output_folder}/{file_folder}/{pr_file}')
            trans_ch.to_csv(f'{output_folder}/{file_folder}/{ch_file}')
            #update chord sets and get onehot encoding 
            if tag=='left':
                full_chord_set_left, full_chord_onehot =build_chord_set_and_1h(trans_pr,full_chord_set_left)
                pitch_chord_set_left, pitch_chord_onehot = build_chord_set_and_1h(trans_ch,pitch_chord_set_left)
                onehot_left[file_folder]= {'full':full_chord_onehot,'pitch':pitch_chord_onehot}
            
            else:
                full_chord_set_right, full_chord_onehot =build_chord_set_and_1h(trans_pr,full_chord_set_right)
                pitch_chord_set_right, pitch_chord_onehot= build_chord_set_and_1h(trans_ch,pitch_chord_set_right)
                onehot_right[file_folder]= {'full':full_chord_onehot,'pitch':pitch_chord_onehot}
        
   # leftfull=[]
    #leftpitch=[]
   # rightfull=[]
   # rightpitch=[]
    for f in onehot_left.keys():
            for chord_type in onehot_left[f].keys():
                lef= onehot_left[f][chord_type]
                rig= onehot_right[f][chord_type]
                name= f
                if chord_type=='full':
                    print(f'inserting missing in full chords {f}')
                    l= insert_missing_columns(full_chord_set_left,lef)
                    r = insert_missing_columns(full_chord_set_right,rig)
                    #write to files
                    
                    l.to_csv(f'{output_folder}/{name}/{name}_left_C_full.csv')
                    r.to_csv(f'{output_folder}/{name}/{name}_right_C_full.csv')
                   # leftfull.append(list(l.columns))
                   # rightfull.append(list(r.columns))
                elif chord_type=='pitch':
                    print(f'inserting missing in pitch chords {f}')
                    l = insert_missing_columns(pitch_chord_set_left,lef)
                    r = insert_missing_columns(pitch_chord_set_right,rig)
                    l.to_csv(f'{output_folder}/{name}/{name}_left_C_pitch.csv')
                    r.to_csv(f'{output_folder}/{name}/{name}_right_C_pitch.csv')
                   # leftpitch.append(list(l.columns))
                   # rightpitch.append(list(r.columns))
                else:
                    print('error')
                    print(chord_type)
                    sys.exit()
        
def preprocess_handless(rawDataPath):
    folder_name= rawDataPath.split('/')[-1]+'_processed_handless'
    abspath= os.path.abspath(rawDataPath+'/../')
    output_folder=os.path.join(abspath,folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    full_chord_set=set()
    pitch_chord_set=set()
    onehot={}
    for file in os.listdir(rawDataPath):
        raw= load_midi(f'{rawDataPath}/{file}')
       
        file_folder= file.split('.')[0]
        
      
        
        trans_pr,trans_ch= sample_and_transpose_to_C(raw,TIME_STEP)
        trans_pr=pd.DataFrame(trans_pr.T); trans_ch=pd.DataFrame(trans_ch.T)
            
        pr_file= file.split('.')[-2]+f'_C_pianoRoll.csv'
        ch_file= file.split('.')[-2]+f'_C_chroma.csv'
            
            
            #checkpoint
        file_folder_path=os.path.join(output_folder,file_folder)
            
        if not os.path.exists(file_folder_path):
            os.makedirs(file_folder_path)
                
                
                
        trans_pr.to_csv(f'{output_folder}/{file_folder}/{pr_file}')
        trans_ch.to_csv(f'{output_folder}/{file_folder}/{ch_file}')
        #update chord sets and get onehot encoding 
                
        full_chord_set, full_chord_onehot = build_chord_set_and_1h(trans_pr,full_chord_set)
        pitch_chord_set, pitch_chord_onehot = build_chord_set_and_1h(trans_ch,pitch_chord_set)
        onehot[file_folder]= {'full':full_chord_onehot,'pitch':pitch_chord_onehot}
                    
    for f in onehot.keys():
        for chord_type in onehot[f].keys():
                    incomplete= onehot[f][chord_type]

                    name= f
                    if chord_type=='full':
                        continue
# =============================================================================
#                         print(f'inserting missing in full chords {f}')
#                         complete= insert_missing_columns(full_chord_set,incomplete)                        
#                         complete.to_csv(f'{output_folder}/{name}/{name}_C_full.csv')
# =============================================================================
                    elif chord_type=='pitch':
                        print(f'inserting missing in pitch chords {f}')
                        complete = insert_missing_columns(pitch_chord_set,incomplete)                       
                        complete.to_csv(f'{output_folder}/{name}/{name}_C_pitch.csv')                                           
                    else:
                        print('error')
                        print(chord_type)
                        sys.exit()
    
    
# =============================================================================
#     pass
#     print('checking files...')
#     for col_list in [leftfull, leftpitch, rightfull, rightpitch]:
#         check2(col_list)
# =============================================================================
    
def insert_missing_columns(chordset, data):
    
    try: 
        assert(len(chordset)>=data.shape[1])
    except:
        print(f'chord set: {len(chordset)}')
        print(f'data : {data.shape}')
    missing = list(chordset.difference(set(data.columns)))
    empties = np.zeros((len(data),len(missing)))
    empties = pd.DataFrame(empties, columns=missing)
    out = pd.concat((data,empties),axis=1)
    out = out.reindex(sorted(out.columns), axis=1)
    return out 
"""function to check if output of preprocess is sound
things to check:
    dimensions of all files should match within type 
    
"""
def check_preprocess(pathToProcessed,filetype):
    
    directory= os.listdir(pathToProcessed)
    cols=[]
    
    for subdir in directory:
        files=os.listdir(pathToProcessed+'/'+subdir)
        s=pathToProcessed+'/'+subdir+'/'
        
        for file in files:
            t=file.split('_')[-3:]
            t=f'{t[0]}_{t[1]}_{t[2]}'
            #print(file)
            if t==filetype:
                print(file)
                current=pd.read_csv(s+file).columns
                cols.append(list(current))
   
    try:
        #assert(len(cols)>1)  
        check2(cols)
        
        print(f'{filetype} passed check')
    except:
       
        print(f'{filetype} failed!')
        #print(col_set)
        #sys.exit()
def check2(col_list):
    try:
        assert(len(col_list)>1)
        first= col_list[0]
        for c in col_list[1:]:
            print('checking columns...')
            assert(np.array_equal(first,c))
        print('passed')
    except:
        print('failed!')
        sys.exit()
    
           
"""
given a piano roll (pd array), create one hot encoding of which notes are tied at each timepoint. 
onehot columns are 'no_tie', 'start','stop','continue'
output should be Nx2 for a Nx128 input matrix
"""
def encode_ties(pianoRoll):
    out=''
    return out 
def add_tie_feature(path):
    for file_folder in os.listdir(path):
        for file in os.listdir(f'{path}/{file_folder}'):
            if file.split('.')[-2].split('_')[-1]=='pianoRoll':
                tie_feature= encode_ties(pd.read_csv(f'{path}/{file_folder}/{file}'))
                filename= file.split('pianoRoll')[-2]+'ties.csv' 
                tie_feature.to_csv(f'{path}/{file_folder}/{filename}')
                print(f'{path}/{file_folder}/{filename}')
        
        
if __name__=='__main__':
   #TIME_STEP=0.0226#estimate_timestep()
   datapath= '../raw/mozart'
   #datapath= '../raw/test'
   preprocess_handless(datapath)
   #add_tie_feature(datapath)
# =============================================================================
#    ins=['left','right']
#    chord=['full','pitch']
#    for i in ins:
#        for c in chord:
#            ft=f'{i}_C_{c}.csv'
#            check_preprocess(datapath+'_processed',ft)
#        
#    
# =============================================================================
     
    
       
   
  
    
# =============================================================================
#     raw= load_midi() 
#     print(raw.key_signature_changes)
#     #raw.instruments=[raw.instruments[1]]
#     
#     pr,chroma=sample_and_transpose_to_C(raw)
#     pr_df=pd.DataFrame(pr.T)
#     pr_df[pr_df!=0]=1
#     fmted= build_chord_dict([chroma])
#     
#     
#     print(pr_df.drop_duplicates().shape)
#     unique_chroma=pd.DataFrame(chroma.T)
#     unique_chroma[unique_chroma != 0]=1
#     print(unique_chroma.drop_duplicates().shape)
#     row=pr[:,0]
#     
#     encode_chord(row)
# =============================================================================
