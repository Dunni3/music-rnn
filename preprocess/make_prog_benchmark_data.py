#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make chord benchmark data 
"""
import numpy as np 
import pretty_midi as pm
triad = [2,2] #numbers to add to index of root of scale
sev = [2,2,2]

majscale= [2,2,1,2,2,2,1]
proglist=[[1,4,5,1],[1,4,5,4,1],[1,5,4,5,1],[1,4,5,4,1],[1,6,5,4,1],[1,5,3,4,1],[1,4,2,5,1],[1,3,4,5,1],
          [1,3,6,4,1],[1,2,3,4,5,1],[4,5,1,4,5,1]]
def make_chord_prog(prog,timestep,start_pitch,scale,chord_type='triad'):
    midi_out = pm.PrettyMIDI()
    ins = pm.Instrument(program=0)
    cum_scale=np.cumsum(scale)
    num_notes = 3 if chord_type=='triad' else 4
    time=0
    start_pitch=pm.note_name_to_number(start_pitch)
    for p in prog:
        notes= make_chord(p-1,start_pitch,scale,num_notes)
        #encode as midi 
        for the_note in notes:
            ins.notes.append(pm.Note(velocity=50,pitch=the_note,start=time,end=time+timestep))
        time+=timestep
        #start_pitch+=scale[p-1]
    midi_out.instruments.append(ins)
    return midi_out
"""
return list of the pitch numbers in the chord
"""
def make_chord(start_pos,start_pitch_num,scale, n_notes):
    
    out=[]
    
    for n in range(n_notes):
        out.append(start_pitch_num+sum(scale[0:start_pos]))
        start_pos=(start_pos+2)%7
        
       
    print(out)
    return out

mido = make_chord_prog(proglist[0],1,'C4',majscale)

for prog in proglist:
    for note in ['C4','D4','E4','F4','G4','A4','B4']:
        for num in ['triad','sev']:
            mido = make_chord_prog(prog,1,note,majscale,num)
            p=str(prog)[1:-1].replace(',','-').replace(' ','')
            mido.write(f'prog_{p}_{note}_{num}.mid')