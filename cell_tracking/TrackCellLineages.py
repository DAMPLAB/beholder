'''
--------------------------------------------------------------------------------
Description: The purpose of this script is to use binary masks to label and
track cells.

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''

import os
import pickle as pickle
import sys
from io import StringIO

import click
import cv2 as cv
import imageio
import numpy as np
import scipy.ndimage as ndimage
import scipy.sparse
from scipy.signal import convolve

# globals
# used to keep track of which frames have fluorescence images
FLFILES = []
FILTERWINDOW = 10


# ----------------------------- Data Filters -----------------------------------
def standard_deviation_filter(data):
    '''
    Filter to set minimum to zero and standard deviation to 1
    Args:
        data:

    Returns:

    '''
    data = data - np.min(data)
    data = data / np.std(data)
    return data * 1.0


def high_pass_window_filter(data):
    '''
    High pass filter, subtracts running average of 10 frames, normalizes
    standard deviation to 1 in the future - make these easy to pass as
    parameters

    Args:
        data:

    Returns:

    '''
    window = FILTERWINDOW
    data = data - convolve(
        data,
        np.ones(window) / window,
        mode='same',
    )

    data[0:window] = 0.0
    data[-window:] = 0.0

    data = data / (np.std(data))

    return data


def low_pass_window_filter(data):
    '''

    Args:
        data:

    Returns:

    '''
    window = FILTERWINDOW
    data_background = convolve(
        data,
        np.ones(window) / window,
        mode='same',
    )
    return data_background


# add grayscale to color image
def add_gray_to_color(IMG, img):
    for ii in range(3):
        IMG[:, :, ii] = IMG[:, :, ii] + img
    return IMG


# ------------------------- Colorization Utilities -----------------------------
def get_colors():
    '''

    Returns:

    '''
    color_file = 'RGB-codes.csv'
    if not os.path.isfile(color_file):
        output_colors = [
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]
    else:
        color_text = open(color_file, 'rb').read()
        # trick for newline problems
        color_text = StringIO(color_text.replace('\r', '\n\r'))
        output_colors = []
        input_colors = np.loadtxt(
            color_text,
            delimiter=",",
            skiprows=1,
            dtype='str',
        )
        for row in input_colors:
            color = (int(row[1]), int(row[2]), int(row[3]))
            output_colors.append(color)

    return output_colors[2:]


# --------------------------- Main Image Pipeline ------------------------------
@click.command()
@click.option(
    '--root_directory',
    prompt='Root directory for input images',
    help='TODO',
    default='../test/tiffs',
)
@click.option(
    '--file_prefix',
    prompt='Prefix for input files',
    help='TODO',
    default='t',
)
# TODO: Add on the fly mask generation.
@click.option(
    '--mask_directory',
    prompt='Prefix for input files',
    help='TODO',
    default='../test/masks',
)
def lineage_pipeline(
        root_directory: str,
        file_prefix: str,
        mask_directory: str,
):
    # the path to the directory containing original/aligned images relative working directory
    # the following are added to the rootdir to get specific files
    # the filenames should be numbered (in this case with three decimals; %03d)
    # mask image
    # TODO: When we're doing the ND2 split out we need to meet this format.
    file_format = mask_directory + '/' + file_prefix + '%06dxy%dc%d.png'

    # useful parameters for if you want to visualize cell and compare to output
    # set to true to create labeled images with numbers (trajectories)
    write_numbers = True
    # set to true to create labeled images with cell regions false colored
    write_labels = True
    # image to write labels on (currently set to mask images)
    label_image = file_format
    # name to save labeled stuff
    label_name = mask_directory + '/' + file_prefix + '-%03d.png'

    # xy label
    # TODO: I don't know why.
    i_xy = 5

    # output name for cell statistics csv
    cell_statistics_csv = 'iXY' + str(i_xy) + '_cell_statistics.csv'
    # output name for lineages csv
    lineage_csv = 'iXY' + str(i_xy) + '_lineagedata.csv'

    # fluorescence channels (used in filenames)
    # used to get string in filename below.
    # TODO: Use our stuff to pull out the channel names.
    fluorescent_channels = ['YFP', 'CFP', 'mCherry']

    # fluorescence image
    fluorescent_file_format = file_prefix + '%06dxy%dc%d.tif'

    # the number for the first frame in the dataset
    # Jx - TODO: We need to use the glob of the images to get this via sort.
    first_frame = 0
    # used to test every nth image for overlap.
    frame_skip = 1
    # the number of the last frame in the dataset
    # Jx - TODO: Same as above, just get the length of the files via the glob.
    frame_max = 29

    # limits on the area of cells to consider
    # Jx - TODO: You probably need to pull out ImageJ to get the rough
    #  approximation of the dimensions of the cells coming off the scope.
    # Jx - TODO: These are kind of guesses based on ImageJ
    area_min = 2000
    area_max = 4000

    # radius of how far center of masses can be and still check distances
    threshold = area_min * 1.5

    # used in csv output
    fluorescent_labels = fluorescent_channels

    # time in between analyzed frames (different from frames)
    # Jx - TODO: Use the metadata coming off of the nd2 to figure out the
    #  timescale
    time_delta = 0.5

    # period of fluorescence frames (every nth frame)
    # Jx - TODO: This is basically where the fluorescent frames start. I think
    # that means that they're chunked like:
    # Frame | Grayscale -> Fl_Channel_1 -> Fl_Channel_2 -> Fl_Channel_3 | into
    # next frame.
    fluorescent_frame_stride = 6
    fluorescent_skip = fluorescent_frame_stride
    # the first frame that has a fluorescence image
    fluorescent_initial_image = 55

    # minimum trajectory length (in actual segmented frames)
    min_trajectory_length = 10  # frames

    def getLabeledMask(iFRAME):

        global FLFILES

        # get segmentation data
        filename = file_format % (iFRAME, i_xy, 1)
        img = imageio.imread(filename)
        # print 'loaded label file', iFRAME, ' with dimensions', img.shape
        img = 255 - img

        # get fluorescence data
        imgFLALL = []
        for icf in fluorescent_channels:
            if ((iFRAME - fluorescent_initial_image) % fluorescent_skip == 0) and (iFRAME >= fluorescent_initial_image):
                filename = root_directory + fluorescent_file_format % (iFRAME, i_xy, icf)
                # print filename
                imgFL = imageio.imread(filename)
                FLFILES.append(iFRAME)
            else:
                imgFL = None

            # print 'loaded fluorescence file', iFRAME, ' with dimensions', imgFL.shape
            imgFLALL.append(imgFL)

        # unpack segmentation data, after labeling
        label, nlabels = ndimage.label(img)

        return (label, nlabels, imgFLALL)

    ############################################################################
    ############################################################################

    # gets center of mass (CoM) and area for each object
    def getObjectStats(label, nlabels, FLALL, i, iFRAME):
        # measure center of mass and area
        comXY = ndimage.center_of_mass(label * 0 + 1.0, label, list(range(nlabels)))
        AREA = ndimage.sum(label * 0 + 1.0, label, list(range(nlabels)))

        # measure mean fluorescence
        FLMEASURE = []

        if fluorescent_initial_image != 0:
            if iFRAME in FLFILES:
                for img in FLALL:
                    flint = ndimage.sum(img, label, list(range(nlabels)))
                    flmean = flint / AREA
                    FLMEASURE.append(flmean)
            else:
                FLMEASURE.append(None)

        # pdb.set_trace()

        if i == 1:
            labelsum = ndimage.sum(label, label, list(range(nlabels)))
            celllabels = labelsum / AREA

            return (comXY, AREA, celllabels, np.array(FLMEASURE))
        else:
            return (comXY, AREA, np.array(FLMEASURE))

    ############################################################################
    ############################################################################

    # for a given index (i2), compute overlap with all i1 indices
    def getMetricOverlap(ARG):

        i2, label1, label2, nlabels1, nlabels2, comXY1, comXY2 = ARG

        # optimized fastest method?

        # get a subregion of the label image, then compare
        SZ = label1.shape
        Xlow = max([0, int(comXY2[0, i2] - threshold)])
        Ylow = max([0, int(comXY2[1, i2] - threshold)])
        Xhigh = min([SZ[0], int(comXY2[0, i2] + threshold)])
        Yhigh = min([SZ[1], int(comXY2[1, i2] + threshold)])

        # i2c = label2[int(comXY2[0,i2]),int(comXY2[1,i2])]
        # if (i2c == i2):
        #	print i2, i2c

        label1 = label1[Xlow:Xhigh, Ylow:Yhigh]
        label2 = label2[Xlow:Xhigh, Ylow:Yhigh]

        # print i2, SZ, comXY2[:,i2], Xlow, Xhigh,Ylow, Yhigh

        # finally, compute overlap using a simple function
        overlap = ndimage.sum(label2 == i2, label1, list(range(nlabels1)))
        # notzero = np.nonzero(overlap)
        # print notzero, i2

        # overlap0 = ndimage.sum(label2==i2, label2, [i2,])
        # if (np.max(overlap)/np.max(overlap0) > 1.0):
        #	print np.max(overlap)/np.max(overlap0)

        return overlap

    ############################################################################
    ############################################################################

    def runOnce(iFRAME):
        print('Processing frame ', iFRAME, '               ')
        sys.stdout.write('\x1b[1A')

        # make labels from the masks
        label1, nlabels1, FL1ALL = getLabeledMask(iFRAME)
        label2, nlabels2, FL2ALL = getLabeledMask(iFRAME + frame_skip)

        # get statistics
        comXY1, AREA1, celllabels1, FLMEASURE1 = getObjectStats(label1, nlabels1, FL1ALL, 1, iFRAME)
        comXY2, AREA2, FLMEASURE2 = getObjectStats(label2, nlabels2, FL2ALL, 2, iFRAME + frame_skip)

        # process center of mass (CoM) arrays
        comXY1 = np.array(list(zip(*comXY1)))
        comXY2 = np.array(list(zip(*comXY2)))
        comXY1 = np.nan_to_num(comXY1)
        comXY2 = np.nan_to_num(comXY2)

        # make the distance map
        DISTMAT = []
        ARGALL = []
        for i2 in range(nlabels2):
            ARGALL.append((i2, label1, label2, nlabels1, nlabels2, comXY1, comXY2))
        DISTMAT = list(map(getMetricOverlap, ARGALL))
        DISTMAT = np.array(DISTMAT)
        # print DISTMAT.shape

        # return a compressed sparse array, since most entries are zeros
        DISTMAT = scipy.sparse.csr_matrix(DISTMAT)

        # print 'frame ', iFRAME, ' done'

        # pdb.set_trace()

        return (iFRAME, label1, nlabels1, (comXY1, celllabels1, AREA1), DISTMAT, FLMEASURE1)

    ############################################################################
    ############################################################################

    ############################################################################
    ############################################################################

    # track lineages proper
    # currently will only output first channel of fluorescence values in csv and pkl files

    # get all the positions and save them in nested dictionaries by frame then trajectory
    def getPositions(trajnum, XY, iFRAME, label):
        XY = int(XY[1]), int(XY[0])
        LOC[iFRAME][trajnum] = XY
        LABELS[iFRAME][trajnum] = label

    # write to the image and save in images dictionary
    def writeLabel(newtrajnum, XY, label, iFRAME):
        lshape = label.shape
        img = images[iFRAME]

        color = colors[(int(newtrajnum[0:4]) % len(colors))]

        # print newtrajnum, iFRAME, XY, len(label)

        # could create a mask based on single label and combines it with image
        # right now it just edits the image itself
        # fnp = np.empty(img.shape)
        if lshape[-1] == 3:
            for j, k, i in label:
                # first line uses colors from colorfile, second line just uses red
                img[j, k] = color
        # img[j,k] = (255,0,0)
        elif lshape[-1] == 2:
            for j, k in label:
                # first line uses colors from colorfile, second line just uses red
                img[j, k] = color
        # img[j,k] = (255,0,0)

        else:
            print('Error: label does not contain coordinates')

        # pdb.set_trace()
        # img = img + fnp*0.1

        images[iFRAME] = img

    # function that re-lables all trajectories by iFRAME then calls writeImage
    def relableFRAME(iFRAME):
        for loc in LOC[iFRAME]:
            XY = LOC[iFRAME][loc]
            label = LABELS[iFRAME][loc]

            if loc in NEWTRAJ:
                newtrajnum = NEWTRAJ[loc]

            else:
                DIVIDE[loc] = []
                for key in LOC[iFRAME]:
                    # test if location matches that of a previous trajectory
                    if (LOC[iFRAME][loc] == LOC[iFRAME][key]) and (key != loc) and (key in NEWTRAJ):
                        if key in BRANCH:
                            num = BRANCH[key] + 1
                        else:
                            num = 1
                        BRANCH[key] = num
                        newtrajnum = NEWTRAJ[key]
                        newtrajnum2 = newtrajnum + '-%d' % num
                        NEWTRAJ[loc] = newtrajnum2
                        # add division time to both trajectories
                        DIVIDE[key].append(iFRAME)
                        DIVIDE[loc].append(iFRAME)
                if loc not in NEWTRAJ:
                    cellvisit = len(NEWTRAJ) + 1
                    newtrajnum = '%04d' % cellvisit
                    NEWTRAJ[loc] = newtrajnum

            if write_labels:
                writeLabel(newtrajnum, XY, label, iFRAME)

            # for writing label text in images
            towrite = newtrajnum, XY
            TOWRITE[iFRAME].append(towrite)

    # has to run after labels so they aren't written over
    def writeText(iFRAME):
        img = images[iFRAME]
        for line in TOWRITE[iFRAME]:
            XY = line[1]
            newtrajnum = line[0]
            cv.putText(img, newtrajnum, XY, cv.FONT_HERSHEY_DUPLEX, .3, (0, 0, 0), 1)
        return img

    # get measurements in parallel
    MEASUREMENTS = []
    ARGLIST = []
    for iFRAME in range(first_frame, frame_max, 1):
        ARGLIST.append(iFRAME)

    # if pool != None:
    #	pool = Pool(pool)
    #	MEASUREMENTS = pool.map(runOnce, ARGLIST)
    #	pool.close
    # else:
    MEASUREMENTS = list(map(runOnce, ARGLIST))

    TRACKING_RESULTS = MEASUREMENTS

    # pdb.set_trace()

    # ~ ############################################################################
    ############################################################################
    # save results - for use when running piecemeal
    fpkl = open('trackMasks.pkl', 'wb')
    pickle.dump(MEASUREMENTS, fpkl, protocol=pickle.HIGHEST_PROTOCOL)
    fpkl.close()

    # ~ ############################################################################
    # ~ ############################################################################
    # analyzeTracking

    with open('trackMasks.pkl', 'rb') as f:
        TRACKING_RESULTS = pickle.load(f)

    # unpack results into handy variables
    iFRAME, label, nlabels, CELLSTATS, DISTMAT, FLMEASURE = list(zip(*TRACKING_RESULTS))

    # cell statistics specifically
    comXY, celllabels1, AREA = list(zip(*CELLSTATS))

    # number of fluorescence channels
    try:
        FLN = len(FLMEASURE[0][:, 0])
    except:
        FLN = len(FLMEASURE[0])
    print(FLN)

    # ~ ############################################################################
    # ~ ############################################################################
    # get total cell data
    if fluorescent_initial_image != 0:
        # total number of frames
        iTN = FLN

        # place to save mean and std data
        FLMEANALL = []
        FLMEANALLFILTERED = []
        FLMEANALLBACKGROUND = []
        FLSTDALL = []
        FLMEDIANALL = []

        # plot mean and std. dev. intensity vs. time across cells
        print('Analyzing total cell fluorescence               ')
        for ifl in range(0, FLN):
            # print 'analyzing fluorescence channel ', ifl + 1, '           '
            # sys.stdout.write('\x1b[1A')

            fln = []
            fltime = []
            flmean = []
            flstd = []
            flmedian = []

            # for iT in range(0,iTN,dPeriodFL):
            for iT in range((fluorescent_initial_image - first_frame), (frame_max - first_frame),
                            fluorescent_frame_stride):
                # print iT, ifl
                flframe = iT + first_frame
                # pdb.set_trace()
                fl = FLMEASURE[iT][ifl, :]
                area = np.array(AREA[iT])
                iselect = np.where((area >= area_min) * (area <= area_max))[0]

                fln.append(len(iselect))
                fltime.append(time_delta * flframe)
                flmean.append(np.mean(fl[iselect]))
                flmedian.append(np.median(fl[iselect]))
                flstd.append(np.std(fl[iselect]))

            # save background for later use
            # print 'flmean = ', flmean
            FLMEANALL.append(flmean)
            FLMEDIANALL.append(flmedian)
            FLSTDALL.append(flstd)
            FLMEANALLFILTERED.append(standard_deviation_filter(flmean))
            FLMEANALLBACKGROUND.append(low_pass_window_filter(flmean))

        with open('iXY' + str(i_xy) + '_global-cell-statistics.pkl', 'wb') as f:
            pickle.dump((fltime, fln, FLMEANALL, FLMEDIANALL, FLSTDALL, FLMEANALLFILTERED, FLMEANALLBACKGROUND), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

        # save data to CSV
        # need to loop through number of channels

        f = open(cell_statistics_csv, 'w')
        f.write('time,')
        f.write('cell count,')
        for lmn in range(len(FLMEANALL)):
            f.write('%s mean,' % fluorescent_labels[lmn])
            f.write('%s std.,' % fluorescent_labels[lmn])
            f.write('%s median,' % fluorescent_labels[lmn])
        f.write('\n')

        for ijk in range(len(fltime)):
            f.write(str(fltime[ijk]) + ',')
            f.write(str(fln[ijk]) + ',')
            for lmn in range(len(FLMEANALL)):
                f.write(str(FLMEANALL[lmn][ijk]) + ',')
                f.write(str(FLSTDALL[lmn][ijk]) + ',')
                f.write(str(FLMEDIANALL[lmn][ijk]) + ',')
            f.write('\n')

        f.close()

    # pdb.set_trace()

    # begin tracking proper

    # find many trajectories by starting at a multitude of frames
    FRAMEMAXLIST = list(range(frame_max, (first_frame + min_trajectory_length), -1))
    # FRAMEMAXLIST = [FRAMEMAX,FRAMEMAX-30,FRAMEMAX-30*2,FRAMEMAX-30*3,FRAMEMAX-30*4]

    # current number of trajectories
    TRAJCOUNT = 0

    # store trajectories
    TRAJ = []

    # keep track of which indices are visited
    VISITED = []
    for ijk in range(frame_max + 1):
        VISITED.append([])

    print('Tracking cells')
    # scan through the final frame
    for frame_max in FRAMEMAXLIST:
        print('framemax ', frame_max, '              ')
        # sys.stdout.write('\x1b[1A')

        # scan all cell ID's at the final frame
        cellIDStart = list(range(nlabels[frame_max - first_frame - 1]))

        # current cell ID
        cellID = 0

        # loop
        for cellID in cellIDStart:
            # print 'tracking cell ID: ', cellID

            # frame
            frame = []

            # time
            time = []

            # area
            area = []

            # yfp
            if FLN > 0:
                fl0 = []

            # flag for a "bad" trajectory
            bBad = False

            # for cell positions
            cellXY = []

            # for cell label
            celllabels = []

            loop = 0

            for iT in range(frame_max - first_frame - 1, 0, -1):

                # mark cell as visited if not visited, otherwise end trajectory
                if (cellID in VISITED[iT]):
                    bBad = True
                else:
                    VISITED[iT].append(cellID)

                # find next best cell
                dist = DISTMAT[iT - 1]

                # optionally print out shape information
                # print iT, cellID, dist.shape, nlabels[iT], nlabels[iT-1]

                # area
                area1 = AREA[iT][cellID]
                if (not ((area1 >= area_min) and (area1 <= area_max))):
                    # print 'bad area ', area1, ' cell ', cellID
                    bBad = True

                # area of potential matches
                area2 = np.array(AREA[iT - 1])

                iselect2 = np.where((area2 >= area_min) * (area2 <= area_max))[0]
                # print iselect2

                dist2 = np.squeeze(dist[cellID, :].toarray())
                dist2 = dist2[iselect2]

                cellID2 = iselect2[np.argmax(dist2)]
                distmax = np.amax(dist2)

                # record current data

                # print 'iT =', iT, 'cellID = ', cellID
                # iFRAME if off from iT by 1 so add 1 to get frame
                frame.append(iT + first_frame)

                CELLX = comXY[iT][0][cellID]
                CELLY = comXY[iT][1][cellID]
                CELLXY = [CELLX, CELLY]
                # print cellXY
                cellXY.append(CELLXY)

                celllabel = celllabels1[iT][cellID]
                framelabel = label[iT]
                celllabel = np.where(framelabel == celllabel)
                celllabel = np.column_stack(celllabel)
                celllabels.append(celllabel)

                time.append(time_delta * (iT + first_frame))
                area.append(area1)
                if FLN != 0:
                    if (((iT + first_frame - fluorescent_initial_image) % fluorescent_frame_stride == 0) and (
                            (iT + first_frame) >= fluorescent_initial_image)):
                        try:
                            if FLN > 0:
                                fl0.append(FLMEASURE[iT][:, cellID])

                        except:
                            print('no fl data for ' + str(iT + first_frame) + '              ')
                    else:
                        if FLN > 0:
                            emptyfl = np.empty((FLN,))
                            emptyfl[:] = np.nan
                            fl0.append(emptyfl)

                # area in the previous time
                area2 = AREA[iT - 1][cellID2]

                ####################
                ####################

                # only check if not bad
                if (not bBad):

                    # check for wrong rate of change for area
                    if ((area2 - area1) / area1 < -0.6):
                        bBad = True
                    # print 'cell ', cellID, ' failed due to area shrinkage = ', (area2-area1)/area1
                    # print '\tarea 1 = ', area1
                    # print '\tarea 2 = ', area2

                    # check for strong overlap
                    if ((distmax / area1) < 0.5):
                        bBad = True
                # print 'cell ', cellID, ' failed due to low overlap = ', (distmax/area1)
                # print '\tarea 1 = ', area1
                # print '\tarea 2 = ', area2

                if (bBad):
                    break

                ####################
                ####################

                cellID = cellID2

            # pdb.set_trace()

            # if ((np.max(flarea)<800) and (not bBad)):
            # if ((len(fltime)>100) and (np.max(flarea)<800)):
            # pdb.set_trace()
            # print 'fmeasures ', fltime, np.mean(flarea), np.max(fl0)

            # if ((len(fltime)>MINTRAJLENGTH) and (np.mean(flarea)<500) and (np.max(fl0)<4000)):
            if ((len(frame) > min_trajectory_length)):
                if FLN > 0:
                    TRAJ.append((frame, time, area, cellXY, celllabels, fl0))
                else:
                    TRAJ.append((frame, time, area, cellXY, celllabels))

                # pdb.set_trace()
                TRAJCOUNT += 1
                print(TRAJCOUNT, ' trajectories', '                  ')
                sys.stdout.write('\x1b[1A')

    print('\n', TRAJCOUNT, ' total trajectories')

    # ~ #pickle file to output when running piecemeal
    with open('raw_traj.pkl', 'wb') as f:
        pickle.dump(TRAJ, f, protocol=pickle.HIGHEST_PROTOCOL)
    # ~ # pdb.set_trace()

    with open('raw_traj.pkl', 'rb') as f:
        TRAJ = pickle.load(f)
    FLN = len(fluorescent_labels)

    ########################################################################
    ########################################################################

    # keep track of which indices are visited
    LVISITED = []
    # keep track of previous locations
    LOC = {}
    # keep track of labels in each frame
    LABELS = {}
    # keep dictionary of images to save at the end
    images = {}
    # store trajectories with new names
    NEWTRAJ = {}
    # keep track of which trajectories are branches
    CHANGEDTRAJ = {}
    # keep track of how many branches a given trajectory has
    BRANCH = {}
    # keep track of all cell labels
    TOWRITE = {}
    # keep track of cell divisions
    DIVIDE = {}

    colors = get_colors()
    trajnum = -1

    # actually go through all the trajectories
    for traj in TRAJ:
        if FLN == 0:
            frame, time, area, cellXY, celllabels = traj
        else:
            frame, time, area, cellXY, celllabels, fl0 = traj

        trajnum = trajnum + 1

        print('Processing trajectory ', trajnum, '           ')
        sys.stdout.write('\x1b[1A')

        # eachtraj represents each frame a trajectory is in
        for eachtraj in range(0, len(cellXY)):
            cell = cellXY[eachtraj]
            label = celllabels[eachtraj]

            if len(label) > 900000:
                print('Skipping traj ', trajnum, 'label is too large              ')

            else:
                # print cell
                iFRAME = frame[eachtraj]
                # reads all the frame images into images dictionary
                if iFRAME not in LVISITED:
                    LVISITED.append(iFRAME)
                    filename = root_directory + label_image % (iFRAME, i_xy, 1)
                    img = imageio.imread(filename)
                    imgshape = img.shape
                    bimg = np.zeros(imgshape + (3,))
                    img = add_gray_to_color(bimg, img)
                    images[iFRAME] = img
                    LOC[iFRAME] = {}
                    LABELS[iFRAME] = {}
                    TOWRITE[iFRAME] = []
                # area = flarea[eachtraj]
                getPositions(trajnum, cell, iFRAME, label)

    ########################################################################
    ########################################################################

    LVISITED = sorted(LVISITED)
    for iFRAME in LVISITED:
        relableFRAME(iFRAME)

    NEWTRAJLIST = []

    # make NEWTRAJ list using new TRAJ names
    for key in NEWTRAJ:
        traj = TRAJ[key]
        if FLN == 0:
            frame, time, area, cellXY, celllabels = traj
        else:
            frame, time, area, cellXY, celllabels, fl0 = traj

        trajname = NEWTRAJ[key]
        # calculate doubling times
        try:
            divisions = DIVIDE[key]
        except:
            divisions = None
        try:
            j = 0
            k = 1
            dframes = []
            for divide in range(len(divisions) - 1):
                dframe = divisions[k] - divisions[j]
                dframes.append(dframe)
                j += 1
                k += 1
            dframes = np.array(dframes)
            dtime = (np.mean(dframes)) * time_delta
            if dtime == 0:
                dtime = 'nan'
        except:
            dtime = 'nan'
        if FLN == 0:
            traj = trajname, frame, time, area, cellXY, celllabels, divisions, dtime
        else:
            traj = trajname, frame, time, area, cellXY, celllabels, fl0, divisions, dtime

        NEWTRAJLIST.append(traj)
        print('Processed ', len(NEWTRAJLIST), ' lineages')
        sys.stdout.write('\x1b[1A')

    if write_labels:
        if write_numbers:
            # pdb.set_trace()
            # write text to images and save
            for key in images:
                image = writeText(key)
                print('Saving ... ', key, '             ')
                sys.stdout.write('\x1b[1A')
                savename = root_directory + label_name % (key)
                cv.imwrite(savename, image)

        else:
            for key in images:
                image = images[key]
                print('Saving ... ', key, '             ')
                sys.stdout.write('\x1b[1A')
                savename = root_directory + label_name % (key)
                cv.imwrite(savename, image)

    ########################################################################
    ########################################################################

    # output CSV

    # things to add to csvfile
    length = []
    name = []
    itimes = []
    etimes = []
    meanAREA = []
    stdAREA = []
    if FLN > 0:
        meanFL = []
        stdFL = []
    dtimes = []

    for traj in NEWTRAJLIST:
        if FLN == 0:
            trajname, frame, time, area, cellXY, celllabels, divisions, dtime = traj
        else:
            trajname, frame, time, area, cellXY, celllabels, fl0, divisions, dtime = traj

        name.append(trajname)
        itimes.append(time[1])
        etimes.append(time[-1])
        area = np.array(area)
        meanAREA.append(np.mean(area))
        stdAREA.append(np.std(area))
        if FLN > 0:
            flmean = []
            flstd = []
            flarray = np.array(fl0, dtype=np.float)
            for fc in range(FLN):
                flmean.append(np.nanmean(flarray[:, fc]))
                flstd.append(np.nanstd(flarray[:, fc]))
            meanFL.append(flmean)
            stdFL.append(flstd)
        dtimes.append(dtime)

    NAMES = np.array(name)
    ITIMES = np.array(itimes)
    ETIMES = np.array(etimes)
    MEANAREA = np.array(meanAREA)
    STDAREA = np.array(stdAREA)
    if FLN > 0:
        MEANFL = np.array(meanFL)
        STDFL = np.array(stdFL)
    DTIMES = np.array(dtimes)

    f = open(lineage_csv, 'w')
    f.write('traj name,')
    f.write('final time,')
    f.write('initial time,')
    f.write('mean area,')
    f.write('std. area,')
    if FLN > 0:
        for fllab in fluorescent_labels:
            f.write('mean ' + fllab + ',')
            f.write('std. ' + fllab + ',')
    f.write('doubling time')
    f.write('\n')

    for ijk in range(len(name)):
        f.write(str(NAMES[ijk]) + ',')
        f.write(str(ITIMES[ijk]) + ',')
        f.write(str(ETIMES[ijk]) + ',')
        f.write(str(MEANAREA[ijk]) + ',')
        f.write(str(STDAREA[ijk]) + ',')
        if FLN > 0:
            for fc in range(FLN):
                f.write(str(MEANFL[ijk][fc]) + ',')
                f.write(str(STDFL[ijk][fc]) + ',')
        f.write(str(DTIMES[ijk]) + ',')
        f.write('\n')

    f.close()

    # save picklefile with NEWTRAJLIST
    # traj = frame,time,area, cellXY, celllabels,fl0
    with open('iXY' + str(i_xy) + '_lineagetracking.pkl', 'wb') as f:
        pickle.dump(NEWTRAJLIST, f, protocol=pickle.HIGHEST_PROTOCOL)

    if FLN != 0:
        with open('iXY' + str(i_xy) + '_lineagetrackingsummary.pkl', 'wb') as f:
            pickle.dump((NAMES, ITIMES, ETIMES, MEANAREA, STDAREA, MEANFL, STDFL), f, protocol=pickle.HIGHEST_PROTOCOL)

    # saves TOWRITE in pkl file to use in Image_Analysis.ipynb
    with open('iXY' + str(i_xy) + '_lineagetext.pkl', 'wb') as f:
        pickle.dump(TOWRITE, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nCell tracking complete')


if __name__ == "__main__":
    lineage_pipeline()