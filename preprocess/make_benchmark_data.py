#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create midi scale
"""
import pretty_midi as pm 
import numpy as np
import os

def makescale(intervals,notedur=1,n_octaves=3,n_scales=1,startnote='C4'):
    midi=pm.PrettyMIDI()
    if type(startnote)==str:
        startnum=pm.note_name_to_number(startnote)
    else:
        startnum=startnote
    ins=pm.Instrument(program=0)
    intervals=list(intervals)
    all_intervals= intervals*n_octaves +list(-1*np.array(intervals[::-1]))*n_octaves
    timestep=notedur
    cur_time=0
    cur_pitch=startnum
    n=pm.Note(velocity=50,pitch=cur_pitch,start=cur_time,end=cur_time+timestep)
    ins.notes.append(n)
    for scales in range(n_scales):
        for diff in all_intervals:
                cur_pitch+=diff
                cur_time+=timestep
                n=pm.Note(velocity=50,pitch=cur_pitch,start=cur_time,end=cur_time+timestep)
                ins.notes.append(n)
    midi.instruments.append(ins)
    return midi
 
major = [2,2,1,2,2,2,1]
minor = [2,1,2,2,1,2,2]

chromatic = np.ones(12).astype(int)
maj_arpeg = [4,3,5]
min_arpeg = [3,4,5]
makescale(maj_arpeg,n_scales=3).write('maj_arpeg.mid')
makescale(min_arpeg,n_scales=4,notedur=.1).write('min_arpeg.mid')
makescale(chromatic,n_scales=4,notedur=.1).write('chromatic.mid')

timerange=[.1]
pitchrange=[0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11]
octave_range=[1,2,3,4]
scale_range=[2]
scale_types=['maj','min','majArp','minArp','chromatic']
output_folder='benchmark'
if not os.path.exists(output_folder):
     os.makedirs(output_folder)
for t in timerange:
    for p in pitchrange:
        for octa in octave_range:
            for s in scale_range:
                
                maj_ = makescale(major, notedur=t,n_octaves=octa,startnote=72+p,n_scales=s)
                min_ = makescale(minor, notedur=t,n_octaves=octa,startnote=72+p,n_scales=s)
                maj_arp = makescale(maj_arpeg, notedur=t,n_octaves=octa,startnote=72+p,n_scales=s)
                min_arp = makescale(min_arpeg, notedur=t,n_octaves=octa,startnote=72+p,n_scales=s)
                chromat= makescale(chromatic, notedur=t,n_octaves=octa,startnote=72+p,n_scales=s)
                for st,midi in zip(scale_types,[maj_,min_,maj_arp,min_arp,chromat]):
                    midi.write(f'{output_folder}/{st}_time={int(t*10)}_pitch={pm.note_number_to_name(72+p)}_octvs={octa}_loops={s}.mid')
# =============================================================================
# maj=makescale(major,startnote='D5')
# maj.write('scale_maj_test.mid')
# =============================================================================
#mino= makescale(minor)
# =============================================================================
# mino.write('scale_min.mid')
# chromatic = [1,1,1,1,1,1,1]
# =============================================================================

