# File originally written by Guizela Huelsz Prince

import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import os
import sys
from track_lib import Track, load_track_list, save_track_list, add_track_to_tracklist
import time
import matplotlib.patheffects as path_effects
from skimage import exposure


def make_RGB_time_image(im_array,mode, contr):
    RGB_mode=[[True,True],[False,True],[True,False]]
    im_RGB=np.zeros( (im_array[0].shape[0],im_array[0].shape[1],3), dtype=float)
    for i in range(0,2):
        if RGB_mode[mode][i]:

            im = np.array(im_array[i],dtype=float)
            im/=im.max()
            imm = exposure.rescale_intensity(im,(np.min(im),np.max(im) - contr))

            im_RGB[:,:,i]=imm
    return im_RGB

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

inp = 1 #reveive input from keybord in console

# load list of tracks
(track_list, track_lbl_list) = load_track_list(save_dir)

# determine whether new image data must be loaded, i.e. when the list im_data is 
# either not defined or empty
load_im_data=False
if ('im_data' in locals())==False:
    load_im_data=True
elif len(im_data)==0:
    load_im_data=True

# load image data
if load_im_data:    
    tt = time.clock()
    # empty image data list im_data and time point list T
    im_data=[]
    T=[]
    print ("loading files:")
    t=t0
    cont=True
    while cont:
        try:
            name = 't%0' + str(t_zeros) + 'dc1.tif'
            infile= pref + name % t
            print ("\t"+infile)
            with tif.TiffFile(data_dir+infile) as f:
                tmp=f.asarray()
            im_data.append(tmp)
            T.append(t)
            t=t+1
            if t == tf:
                cont = False
        except IOError:
            cont=False
        except MemoryError:
            cont=False
            tmp=[]
            print ("out of memory: loaded %d timepoints, %d-%d" % (len(im_data),T[0],T[-1]))
#    print (time.clock()-tt, "s loading")

    # save dimensions and number of time_points
#    NZ=im_data[0].shape[0]
#    NX=im_data[0].shape[1]
#    NY=im_data[0].shape[2]
    NF=len(im_data)

# get and print info on tracks
t0_list=[]
t0_track_list=[]
t1_list=[]
t1_track_list=[]

for i in range(0,len(track_list)):
#    print("%d" % i)
    tr=track_list[i]
    if tr.t[0] in t0_list:
        ind=[s for s,q in enumerate(t0_list) if q==tr.t[0]][0]
        t0_track_list[ind].append(track_lbl_list[i])
    else:
        t0_list.append(tr.t[0])
        t0_track_list.append([track_lbl_list[i]])

    if tr.t[-1] in t1_list:
        ind=[s for s,q in enumerate(t1_list) if q==tr.t[-1]][0]
        t1_track_list[ind].append(track_lbl_list[i])
    else:
        t1_list.append(tr.t[-1])
        t1_track_list.append([track_lbl_list[i]])
        
#print ("---------------------------------tracks starting at:")
#for i in range(0,len(t0_list)):
#    print ("\tt0=%d (" % (t0_list[i]),)
#    for t in t0_track_list[i]:
#        print ("%d" % t,)
##    print (")")
#print ("---------------------------------tracks ending at:")
#for i in range(0,len(t1_list)):
#    print ("\tt1=%d (" % (t1_list[i]),)
#    for t in t1_track_list[i]:
#        print ("%d" % t,)
##    print (")")
         
class annotate_track_dialog_window:
    def __init__(self, fig, track_list, track_lbl_list, im_data, T, curr_tr):
        # keys to move in z
        self.key_z=['q','w','e','r']
        self.dz=[-1,1,-5,5]
        #change image levels (max)
        self.key_contrast = ['a','s']
        self.dcontrast = [0.1, -0.1]
        # keys to change channel
        self.key_RGB_mode=['z','x','c']
        # keys for moving through time
        self.key_t=['1','2','3','4']
        # rewind-forward keys
        self.key_rwff=['-','=']
        
        self.im_data=im_data
        self.track_list=track_list
        self.track_lbl_list=track_lbl_list
        
        if curr_tr in track_lbl_list:
            self.current_track = track_lbl_list.index(curr_tr)
        else:
            self.current_track = curr_tr
        
        self.fig = fig
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_mouse = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        
        self.T=T
        self.time_point=0
        
        # initialize RGB_mode (0-RG, 1-R, 2-G)
        self.RGB_mode=0

        self.contrast = 0        
        
        self.NZ=im_data[0].shape[0]
        self.NX=im_data[0].shape[1]
        self.NY=im_data[0].shape[2]        
        
        # start at center of stack, if tracks exist
        if self.track_list:
            self.z=self.track_list[self.current_track].x[0,2]
        else:
            self.z=self.NZ/2
        
        self.im_peek=[]
        self.z_peek=-1
        self.time_point_peek=-1
        
        # calculate image for current z and RGB mode
        self.calc_im_RGB()
        # show image
        self.show_image([self.NY,self.NX])

    def on_button_press(self, event):
        if event.button==1 and self.RGB_mode == 1: 
            # if left mouse button is pressed and the green channel is visible
            # get current cell position
            x=np.array([event.xdata, event.ydata, self.z])
            # and current track
            tr=self.track_list[self.current_track]
            # save cell position for time point corresponding to image in green channel
            if self.RGB_mode in [0,2]:
                t=self.T[self.time_point+1]
            elif self.RGB_mode == 1:
                t=self.T[self.time_point]
                tr.add_point(x,t) #only add point in current time f
            # update image
            self.show_image()
        elif event.button==3:
            # switch to track closest to mouse position (x,y)
            x=np.array([event.xdata, event.ydata])
            t=self.T[self.time_point]
            self.current_track=self.find_track_closest_to_x(self.track_list,x,t)
            # show image
            self.show_image()

    def on_key_press(self, event):
        if event.key in self.key_z:
            # change z
            ind=self.key_z.index(event.key)
            self.z=self.z+self.dz[ind]
            # check bounds are respected
            if self.z<0:
                self.z=0
            elif self.z>=self.NZ:
                self.z=self.NZ-1
            # calculate image for current z and RGB mode
            self.calc_im_RGB()
            # show image
            self.show_image()
        elif event.key=='n' and self.RGB_mode==1:
            # if only red channel shown, add new track starting at time 
            # corresponding to image in red channel at mouse position (x,y)
            x=np.array([event.xdata, event.ydata, self.z],dtype=float)
            # add track to track list
            t=self.T[self.time_point]
            (self.track_list,self.track_lbl_list,self.current_track) = add_track_to_tracklist(x,t,self.track_list,self.track_lbl_list)
            # show image
            self.show_image()
        elif event.key in self.key_RGB_mode:
            # set RGB mode
            self.RGB_mode=self.key_RGB_mode.index(event.key)
            # calculate image for current z and RGB mode
            self.calc_im_RGB()
            # show image
            self.show_image()
        elif event.key in self.key_t:
            # move in time
            # first determine whether moving forward (dt=+1) or backward (dt=-1)
            if event.key==self.key_t[0]:
                df=-1
            elif event.key==self.key_t[2]:
                df = -5
            elif event.key==self.key_t[1]:
                df=+1
            elif event.key==self.key_t[3]:
                df = +5
            if self.time_point+df>=0 and self.time_point+df<NF-1:
                self.time_point+=df
                # calculate image for current z and RGB mode
                self.calc_im_RGB()
                # show image
                self.show_image()
        
        elif event.key in self.key_rwff:
            tr=self.track_list[self.current_track]
            if event.key==self.key_rwff[0]:
                # get time_point that corresponds to the first timepoint of track
                q=[i for i,j in enumerate(self.T) if j==tr.t[0]]
                if q:
                    # if that time_point exists, then set current time_point to it
                    self.time_point=q[0]
                else:
                    # otherwise, just do first time_point
                    self.time_point=0
            elif event.key==self.key_rwff[1]:
                # get time_point that corresponds to the second-to-last timepoint of track
                q=[i for i,j in enumerate(self.T) if j==tr.t[-2]]
                if q:
                    # if that time_point exists, then set current time_point to it
                    self.time_point=q[0]
                    # but make sure that time_point is never larger than the second-to-last
                    if self.time_point>NF-2:
                        self.time_point=NF-2
                else:
                    # otherwise, just do second-to-last time_point
                    self.time_point=NF-2
            # calculate image for current z and RGB mode
            self.calc_im_RGB()
            # show image
            self.show_image()
        
        elif event.key==' ':
            # move to z-slice of cell in the current track at time point 
            # corresponding to the red channel
            if self.RGB_mode in [0,2]:
                t=self.T[self.time_point+1]
            else:
                t=self.T[self.time_point]
            x_p=self.track_list[self.current_track].get_pos(t)
            if x_p.size:
                # if a point for the current track exists at this time
                # get z-position
                self.z=x_p[2]
                # calculate image for current z and RGB mode
                self.calc_im_RGB()
                # show image
                self.show_image()
        
        elif event.key=='delete':
            # get current track
            tr=self.track_list[self.current_track]
            # get correct time
            if self.RGB_mode in [0,2]:
                t=self.T[self.time_point+1]
            elif self.RGB_mode == 1:
                t=self.T[self.time_point]
            # remove point
            tr.delete_point(t)
            # show image
            self.show_image()

        elif event.key in self.key_contrast:
            # change contr
            ind = self.key_contrast.index(event.key)
            self.contrast = self.contrast + self.dcontrast[ind]
            # check bounds are respected
            if self.contrast < 0:
                self.contrast = 0
            elif self.contrast > .9:
                self.contrast = .9
            # calculate image for current z and RGB mode
            self.calc_im_RGB()
            # show image
            self.show_image()
        
        elif event.key=='o':
            # if an up to date RFP image doesn't exist
            if self.time_point_peek != self.time_point+1 and self.z_peek != self.z:
                # then load proper RFP image
                t=self.T[self.time_point]
                name = 't%0' + str(t_zeros) + 'dc2.tif'
                infile=data_dir + pref + name % t
                print ("Loading brigt field")
                self.im_peek=np.array(tif.imread(infile,key=int(self.z)),dtype=float)
            # use it to make RGB image
            im0=self.im_peek
            im1=self.im_data[self.time_point+1][self.z,:,:]
            self.im_RGB=make_RGB_time_image( [im0,im1], 0, self.contrast)
            self.show_image()

        elif event.key == '0':
            # reset zoom
            self.show_image([self.NY,self.NX])
            
        elif event.key=='v':
            save_track_list(track_list,track_lbl_list,save_dir)
            
        elif event.key=='escape':
            # save data
            save_track_list(track_list,track_lbl_list,save_dir)
            plt.close(self.fig)
            inp = 1

    def find_track_closest_to_x(self,track_list,x,T):
        dist_sq_min=6e66
        ind_track_min=-1
        for tr in track_list:
            for t in [T,T+1]:
                x_tr=tr.get_pos(t)[0:2]
                if x_tr.size:
                    dist_sq=((x-x_tr)**2).sum()
                    if dist_sq<dist_sq_min:
                        dist_sq_min=dist_sq
                        ind_track_min=[i for i,j in enumerate(track_list) if j==tr][0]
        return ind_track_min

    def calc_im_RGB(self):
        self.z =int(self.z)
        im0=self.im_data[self.time_point][self.z,:,:]
        im1=self.im_data[self.time_point+1][self.z,:,:]
        self.im_RGB = make_RGB_time_image( [im1,im0], self.RGB_mode, self.contrast)
 
    def show_image(self, *size):

        self.ax = plt.gca()
        
        # clear figure
        self.fig.clf()
#        plt.axes([0,0,.5,.5])    
        if size: # xy limits defined by image size
            plt.xlim([0,size[0][0]])
            plt.ylim([size[0][1],0])

        # show image
        plt.imshow(self.im_RGB,interpolation='none')

        # plot track
        for i in range(0,len(self.track_list)):
            tr=self.track_list[i]

            if tr==self.track_list[self.current_track]:
#                col=[1,1,1]
                col = 'w'
                foreground='r'
                markersize = 6
                markeredgecolor = 'r'
                markeredgewidth = 1.5
            else:
                col=[.7,0.7,.8]
                foreground='k'
                markersize = 4
                markeredgecolor = 'k'
                markeredgewidth = 1
#                col=[0,0,0]
#                col = 'None'
            if self.RGB_mode in [0,1]:
                # if green channel is visible, plot points of cell in previous time point
                t=self.T[self.time_point]
                x_p=tr.get_pos(t)
                if x_p.size:
                    if x_p[2] in [self.z]:
                        plt.plot(x_p[0],x_p[1],'>',color=[.9,.9,.9], markeredgecolor = markeredgecolor, markersize = 9, markeredgewidth = 2)
                        text = plt.text(x_p[0]+4,x_p[1]+3,'%d' % self.track_lbl_list[i],color=col,weight = 'bold', size = 12)
                        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground=foreground), path_effects.Normal()])
                        
                    elif x_p[2] in [self.z -2,self.z -1,self.z + 1,self.z +2]: 
                        plt.plot(x_p[0],x_p[1],'s',color=col, markeredgecolor = markeredgecolor, markersize = markersize, markeredgewidth = markeredgewidth)
                        text = plt.text(x_p[0]+4,x_p[1]+3,'%d' % self.track_lbl_list[i],color=col,weight = 'bold', size = 12)
                        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground=foreground), path_effects.Normal()])
                        
            if self.RGB_mode in [0,2]:
                # if red channel is visible, plot points of cell in current time point
                t=self.T[self.time_point+1]
                x_p=tr.get_pos(t)
                if x_p.size:
                    if x_p[2] in [self.z -1,self.z,self.z + 1]:
                        plt.plot(x_p[0],x_p[1],'o',color=col, markeredgecolor = 'k')
                           
        # plot time_point
        plt.title('T:%d,z:%d' % (self.T[self.time_point],self.z),color='w',fontsize=32)
        plt.draw()


        
# make dialog window           
fig=plt.figure(num=1, figsize=(8, 8), facecolor='k')
plt.rcParams['keymap.save'] = [u'ctrl+s']
tmp=annotate_track_dialog_window(fig,track_list,track_lbl_list,im_data,T,curr_tr)
