#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to take output format and convert to midi 

"""

import pretty_midi as pm 
import pandas as pd
import numpy as np 
import os
import sys 
import matplotlib.pyplot as plt 
import process_midi as pro
#GLOBALS
TIME_STEP=pro.TIME_STEP
ins_dict= {'left':"Piano left",'right':"Piano right"}


"""
convert piano roll (pd dataframe) into midi with velocity 50 
piano roll specs:
    Nx128 np array 

"""
def pianoRoll_to_midi(pianoRoll,timestep=TIME_STEP, ins_name="Piano"):
    
    
    midi=pm.PrettyMIDI()
    #confirm specs
    try:
        assert(pianoRoll.shape[1]==128)
    except:
        print(f'column shape {pianoRoll.shape[1]} should be 128')
        print('exiting...')
        sys.exit()
    pianoRoll=pianoRoll.astype(int)
    pianoRoll[pianoRoll!=0]=1
    
    pno = pm.Instrument(program=0)
    pno.name = ins_name
    for notenum in pianoRoll.columns:
        
            consecutives=pianoRoll[notenum].diff().ne(0).cumsum()
            #add notes 
            grouped=pianoRoll[notenum].groupby(consecutives).agg(['mean',len])
           
            grouped['step']=grouped['len'].cumsum()
            means=grouped['mean']
            means[means==0]=pd.NA
            grouped['mean']=means
            grouped=grouped.dropna()
            grouped['start']=(grouped['step']-grouped['len'])*timestep
            grouped['stop']=(grouped['step'])*timestep
            #grouped['note']=pm.Note(velocity=50,pitch=notenum,start=grouped['start'],end=grouped['stop'])
            for r, p in zip(grouped['start'],grouped['stop']):
                pno.notes.append(pm.Note(velocity=50,pitch=int(notenum),start=r, end=p))
            
    midi.instruments.append(pno)
    return midi

"""
helper function that converts binary string to vector 
"""
def chordstring_to_timestep(s):
    length=len(s)
    try:
        #assert length==128
        assert type(s)==str
    except:
        print('check input')
        print('exiting...')
        sys.exit()
    out=np.array(s.replace('','.').split('.')[1:-1],dtype=int)
    
    return out
def full_chord_to_pianoRoll(full):
    #undo onehot encoding 
    #convert each chord string to a vector 
    
    full=full.idxmax(axis=1)
    roll=full.apply(func=chordstring_to_timestep)
    roll=roll.explode().to_numpy().reshape(len(full),128)
    return pd.DataFrame(roll)
def test():
    s1='101'
    s2='000000000'
    out1=np.array([1,0,1])
    out2=np.array([0,0,0,0,0,0,0,0,0])
    assert(np.array_equal(chordstring_to_timestep(s1),out1))
    assert(np.array_equal(chordstring_to_timestep(s2),out2))
    print('fullchord_to_timestep passed')
    test_midi=pro.load_midi()
    test_pr=test_midi.get_piano_roll(1/TIME_STEP)
    test_pr=pd.DataFrame(test_pr).astype(int)
    test_pr[test_pr!=0]=50
    test_pr=pd.DataFrame(test_pr.T)
    
    try:
        out=pianoRoll_to_midi(test_pr)
        out.write('pr_test.mid')
        print('test_pr.mid created!')
    except Exception as e:
        print(e)
        print('pianoRoll_to_midi failed')
        sys.exit()
    full_test = pd.read_csv('../raw/chopin_processed/chp_op18/chp_op18_left_C_full.csv',index_col=0)
    full_test_r = pd.read_csv('../raw/chopin_processed/chp_op18/chp_op18_right_C_full.csv',index_col=0)
    r=full_test.shape[0]/full_test_r.shape[0]
    
   
    try:
        full_test_out=full_chord_to_pianoRoll(full_test)
        full_test_out_r=full_chord_to_pianoRoll(full_test_r)
        pr_full=pm.PrettyMIDI()
        
        pr_full= pianoRoll_to_midi(full_test_out,midi=pr_full,timestep=TIME_STEP/r)
        pr_full = pianoRoll_to_midi(full_test_out_r,midi=pr_full,ins_name='Piano right')
        pr_full.write('full_test.mid')
        
    except Exception as e:
        print('full test failed')
        print(e)
        sys.exit()
   
    handless_test = pd.read_csv('../raw/chopin_processed_handless/chp_op18/chp_op18_C_pianoRoll.csv', index_col=0)
    handless_out = pianoRoll_to_midi(handless_test,midi=pm.PrettyMIDI())
    handless_out.write('handless_test.mid')
    print('all tests passed')
    
if __name__=='__main__':
    #test()
    #pianoRoll_to_midi(pd.read_csv('big_chord.csv')).write('big_chord.mid')
    path='../../raw/chopin_processed_bin/header_right_full.csv'
    head= pd.read_csv(path)
    print(head)
# =============================================================================
#     file='../../raw/mozart_processed/mz_311_1/mz_311_1_right_C_full.csv'
#     chords=pd.read_csv(file,index_col=0)
#     chords['ints']=chords.idxmax(axis=1).apply(lambda r: int(r,2))
#     counts=chords['ints'].value_counts()
#     plt.plot(counts.values)
#     chp_looped='/Users/trachman/Documents/ML/final_project/raw/classical_C/chp_op18_C_.mid'
#     pr=pm.PrettyMIDI(chp_looped).get_piano_roll(fs=1/.02)
#     #get first 15s
#     pr_loop=pd.DataFrame(pr[:,:int(16/.02)].T)
#     pr_loop= pd.concat([pr_loop]*10,ignore_index=True)
#     pianoRoll_to_midi(pr_loop).write('chopin_loop.mid')
# =============================================================================
# =============================================================================
#     piano_test=pm.PrettyMIDI()
#     pno_program=pm.instrument_name_to_program('Acoustic Grand Piano')
#     pno= pm.Instrument(program=pno_program)
#     pno.name='Piano left'
#     note1=pm.Note(velocity=50,pitch=50,start=0,end=0.1)
#     note2=pm.Note(velocity=50,pitch=50,start=0.1,end=1)
#     note3=pm.Note(velocity=50,pitch=50,start=1.01,end=2)
#     for note in [note1,note2,note3]:
#         pno.notes.append(note)
#     piano_test.instruments.append(pno)
#     piano_test.write('piano_test.mid')
#     
#     
# =============================================================================
   
          
    

    
    