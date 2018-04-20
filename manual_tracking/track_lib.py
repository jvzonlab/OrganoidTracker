# File originally written by Guizela Huelsz Prince

import os
import pickle as pickle
import re

import numpy as np


class Track:
    def __init__(self,x,t):
        self.x=np.array([x])
        self.t=np.array([t])
        
    def get_pos(self,t):
        q=np.where(self.t==t)[0]
        if len(q)>0:
            ind=q[0]
            return self.x[ind]
        else:
            return np.array([])

    def add_point(self,x,t):
        # check if time point t is already there
        q=np.where(self.t==t)[0]
        if len(q)==0:
            # if not, then add point
            self.x=np.vstack( (self.x,x) )         
            self.t=np.append( self.t,t )
            # then sort t and x so that all points are sorted in time
            ind_sort=np.argsort(self.t)
            self.t=self.t[ind_sort]
            self.x=self.x[ind_sort]
        else:
            # replace point
            ind=q[0]
            self.x[ind,:]=x
            self.t[ind]=t
            
    def delete_point(self,t):
        # find index of point with time t
        q=[i for i,j in enumerate(self.t) if j==t]
        # if it exists
        if q:           
            ind=q[0]
            # remove time
            self.t=np.delete(self.t,ind)
            # and position
            self.x=np.delete(self.x,ind,axis=0)


def load_track_list(data_dir):
    # intialize lists
    track_list=[]
    track_lbl_list=[]  
    # find all saved tracks
    tmp = os.listdir(data_dir)
    print(data_dir)
    for t in tmp:
        # check if file <t> is a track file
        match=re.search('track_(\d\d\d\d\d)', t)
        if match:
            # if so, find track label
            lbl=int(match.group(1))
            track_lbl_list.append(lbl)
            # load track data
            track_list.append( pickle.load( open(data_dir+t, "rb" ), encoding='latin1' ))
    return (track_list, track_lbl_list)


def add_track_to_tracklist(x,t,track_list,track_lbl_list):
    # find first unused label for new track
    lbl=0
    cont=True
    while cont:
        # for each suggestested track label
        if lbl not in track_lbl_list:
            # if it is not in the label list, use this one and stop looking
            cont=False
        else:
            # if not, try the next one in line
            lbl+=1    
    track_list.append(Track(x,t))
    track_lbl_list.append(lbl)

    return (track_list,track_lbl_list,lbl)


def save_track_list(track_list,track_lbl_list,data_dir):
    # for each track in track list
    for n in range(0,len(track_list)):
        outfile = data_dir + "track_%05d.p" % track_lbl_list[n]
        if len(track_list[n].t) > 0:
            # save in data_dir with file name determined by track label
            pickle.dump( track_list[n], open(outfile, "wb" ) )
        elif os.path.exists(outfile):
            os.remove(outfile)
            print('\n----- Warning: Removed file for empty track #', track_lbl_list[n])
    print('\nSaved tracks:', track_lbl_list)
