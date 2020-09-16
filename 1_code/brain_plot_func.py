# Linden Parkes, 2019
# lindenmp@seas.upenn.edu

import os
import numpy as np
import nibabel as nib
import mayavi as my
from mayavi import mlab
mlab.init_notebook(backend='png')
import surfer
import math

from matplotlib.pyplot import get_cmap

def roi_to_vtx(roi_data, parcel_names, parc_file):
    # roi_data      = (num_nodes,) array vector of node-level data to plot onto surface
    # 
    # 
    # parcel_names     = (num_nodes,) array vector of strings containg roi names
    #               corresponding to roi_data
    # 
    # parc_file    = full path and file name to surface file
    #               Note, I used fsaverage/fsaverage5 surfaces

    # Load freesurfer file
    labels, ctab, surf_names = nib.freesurfer.read_annot(parc_file)

    # convert FS surf_names to array of strings
    if type(surf_names[0]) != str:
        for i in np.arange(0,len(surf_names)):
            surf_names[i] = surf_names[i].decode("utf-8")

    if 'myaparc' in parc_file:
        hemi = os.path.basename(parc_file)[0:2]

        # add hemisphere to surface surf_names
        for i in np.arange(0,len(surf_names)):
            surf_names[i] = hemi + "_" + surf_names[i]

    # Find intersection between parcel_names and surf_names
    overlap = np.intersect1d(parcel_names, surf_names, return_indices = True)
    overlap_names = overlap[0]
    idx_in = overlap[1] # location of surf_names in parcel_names
    idx_out = overlap[2] # location of parcel_names in surf_names

    # check for weird floating nans in roi_data
    fckn_nans = np.zeros((roi_data.shape)).astype(bool)
    for i in range(0,fckn_nans.shape[0]): fckn_nans[i] = math.isnan(roi_data[i])
    if any(fckn_nans): roi_data[fckn_nans] = 0

    # broadcast roi data to FS space
    # initialise idx vector with the dimensions of the FS labels, but data type corresponding to the roi data
    vtx_data = np.zeros(labels.shape, type(roi_data))
    vtx_data = vtx_data - 1000

    # for each entry in fs names
    for i in range(0, overlap_names.shape[0]):
        vtx_data[labels == idx_out[i]] = roi_data[idx_in[i]]

    # get min/max for plottin
    x = np.sort(np.unique(vtx_data))

    if x.shape[0] > 1:
        vtx_data_min = x[0]
        vtx_data_max = x[-1]
    else:
        vtx_data_min = 0
        vtx_data_max = 0

    i = 0
    while vtx_data_min == -1000: vtx_data_min = x[i]; i += 1

    return vtx_data, vtx_data_min, vtx_data_max


def brain_plot(roi_data, parcel_names, parc_file, fig_str, subject_id = 'fsaverage', hemi = 'lh', surf = 'inflated', color = 'coolwarm', center_anchor = 0, showcolorbar = False):

    vtx_data, plot_min, plot_max = roi_to_vtx(roi_data, parcel_names, parc_file)

    if np.all(vtx_data == -1000) == False:
        if color == 'coolwarm':
            if center_anchor == 0:
                if abs(plot_max) > abs(plot_min): center_anchor = abs(plot_max)
                elif abs(plot_max) < abs(plot_min): center_anchor = abs(plot_min)
                else: center_anchor = abs(plot_max)

                if center_anchor == -1000: center_anchor = 1
            print(center_anchor)
            
            if center_anchor != 0:
                view = 'lat'
                fname1 = view + '_' + fig_str + '.png'
                fig = my.mlab.figure(size = (1000,1000))
                fig = my.mlab.gcf()
                brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
                if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
                elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
                brain.add_data(vtx_data, max = center_anchor, center = 0, thresh = -999, colormap = color, alpha = 1, colorbar = showcolorbar)
                brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
                my.mlab.savefig(figure = fig,filename = fname1)
                my.mlab.close()

                view = 'med'
                fname2 = view + '_' + fig_str + '.png'
                fig = my.mlab.figure(size = (1000,1000))
                fig = my.mlab.gcf()
                brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
                if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
                elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
                brain.add_data(vtx_data, max = center_anchor, center = 0, thresh = -999, colormap = color, alpha = 1, colorbar = showcolorbar)
                brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
                my.mlab.savefig(figure = fig,filename = fname2)
                my.mlab.close()

                view = 'ventral'
                fname2 = view + '_' + fig_str + '.png'
                fig = my.mlab.figure(size = (1000,1000))
                fig = my.mlab.gcf()
                brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
                if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
                elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
                brain.add_data(vtx_data, max = center_anchor, center = 0, thresh = -999, colormap = color, alpha = 1, colorbar = showcolorbar)
                brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
                my.mlab.savefig(figure = fig,filename = fname2)
                my.mlab.close()
            else: print('There''s nothing to plot...')
        elif color == 'viridis' or color == 'viridis_r':

            view = 'lat'
            fname1 = view + '_' + fig_str + '.png'
            fig = my.mlab.figure(size = (1000,1000))
            fig = my.mlab.gcf()
            brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
            if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
            elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
            brain.add_data(vtx_data, max = plot_max, min = plot_min, thresh = -999, colormap = color, alpha = 1, colorbar = showcolorbar)
            brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
            my.mlab.savefig(figure = fig,filename = fname1)
            my.mlab.close()

            view = 'med'
            fname2 = view + '_' + fig_str + '.png'
            fig = my.mlab.figure(size = (1000,1000))
            fig = my.mlab.gcf()
            brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
            if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
            elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
            brain.add_data(vtx_data, max = plot_max, min = plot_min, thresh = -999, colormap = color, alpha = 1, colorbar = showcolorbar)
            brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
            my.mlab.savefig(figure = fig,filename = fname2)
            my.mlab.close()
        elif color == 'hot':
            if center_anchor != 0:
                plot_max = center_anchor

            view = 'lat'
            fname1 = view + '_' + fig_str + '.png'
            fig = my.mlab.figure(size = (1000,1000))
            fig = my.mlab.gcf()
            brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
            if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
            elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
            brain.add_data(vtx_data, min = 0, max = plot_max, thresh = -999, colormap = color, alpha = 1, colorbar = showcolorbar)
            brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
            my.mlab.savefig(figure = fig,filename = fname1)
            my.mlab.close()

            view = 'med'
            fname2 = view + '_' + fig_str + '.png'
            fig = my.mlab.figure(size = (1000,1000))
            fig = my.mlab.gcf()
            brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
            if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
            elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
            brain.add_data(vtx_data, min = 0, max = plot_max, thresh = -999, colormap = color, alpha = 1, colorbar = showcolorbar)
            brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
            my.mlab.savefig(figure = fig,filename = fname2)
            my.mlab.close()
        else:
            my_cmap = list()
            for i in np.arange(0,int(plot_max)):
                my_cmap.append(get_cmap(color)(i))
            view = 'lat'
            fname1 = view + '_' + fig_str + '.png'
            fig = my.mlab.figure(size = (1000,1000))
            fig = my.mlab.gcf()
            brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
            if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
            elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
            brain.add_data(vtx_data, min = plot_min, max = plot_max, thresh = -999, colormap = my_cmap, alpha = 1, colorbar = showcolorbar)
            brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
            my.mlab.savefig(figure = fig,filename = fname1)
            my.mlab.close()

            view = 'med'
            fname2 = view + '_' + fig_str + '.png'
            fig = my.mlab.figure(size = (1000,1000))
            fig = my.mlab.gcf()
            brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
            if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
            elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
            brain.add_data(vtx_data, min = plot_min, max = plot_max, thresh = -999, colormap = my_cmap, alpha = 1, colorbar = showcolorbar)
            brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
            my.mlab.savefig(figure = fig,filename = fname2)
            my.mlab.close()
    else:
        view = 'lat'
        fname1 = view + '_' + fig_str + '.png'
        fig = my.mlab.figure(size = (1000,1000))
        fig = my.mlab.gcf()
        brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
        if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
        elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
        brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
        my.mlab.savefig(figure = fig,filename = fname1)
        my.mlab.close()

        view = 'med'
        fname2 = view + '_' + fig_str + '.png'
        fig = my.mlab.figure(size = (1000,1000))
        fig = my.mlab.gcf()
        brain = surfer.Brain(subject_id, hemi, surf, figure = fig, views = view, background = 'white', alpha = 1)
        if subject_id == 'fsaverage': brain.add_morphometry("avg_sulc", colormap="binary", min = -3, max = 3, colorbar = False)
        elif subject_id == 'lausanne125': brain.add_morphometry("avg_curv", colormap="binary", min = -.5, max = .5, colorbar = False)
        brain.add_annotation(parc_file, hemi = hemi, borders = True, alpha=.25, color = 'lightsteelblue')
        my.mlab.savefig(figure = fig,filename = fname2)
        my.mlab.close()

