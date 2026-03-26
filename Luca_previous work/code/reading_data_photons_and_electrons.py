import sys
from line_profiler import LineProfiler

sys.path.append("/home/luca/Documents/allpix/ROOT/root_install/lib")  # replace with directory of your folder lib
import ROOT
from ROOT import std
import gc
# import cppyy
from ROOT import TFile, gDirectory, gSystem, TClass
import pandas as pd
import numpy as np
import os
import argparse
import os.path as path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import norm
from scipy.optimize import curve_fit
import pickle
from tqdm import tqdm
from datetime import datetime
import tracemalloc


# tracemalloc.start()
# import psutil, os
# proc = psutil.Process(os.getpid())

def generate_detector_list_full(n):
    """This is just used to parametrize the detector names depending on how many arrays were used"""
    detector_list = []
    for i in range(0, n):
        detector_list.append(f"detector{i}")  # Use default naming for 0 and positive numbers
    return detector_list


def gaussian(x, a, mu, sigma):
    """ This is a simple gaussian function with mean mu, std sigma and amplitude a"""
    return a * np.exp(-((np.abs(x - mu)) ** 2) / (2 * sigma ** 2))


def find_non_zero_square(matrix):
    """This function takes a matrix and cuts out of it a smaller matrix containing all the non-zero terms
    The matrix cut out is squared and padded with one line of zeros around the edges"""

    # Step 1: Find indices of non-zero elements
    non_zero_indices = np.argwhere(matrix != 0)

    if non_zero_indices.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan

        # Step 2: Identify rows and columns with non-zero elements
    _, row_min, col_min = np.min(non_zero_indices, axis=0)  # min row and col indices
    _, row_max, col_max = np.max(non_zero_indices, axis=0)  # max row and col indices

    # return the smaller square matrix along with the row and col delimiters of its position in the bigger one
    return row_min, row_max, col_min, col_max


def crop_matrices_tracks(matrix, x_source_index, y_source_index, size, flag):
    """This function takes a matrix, the source indices and a flag. If flag == 0 it means that we are going to
    plot an estimated position source from either the gaussian fit or the gridsum algorithm, while if the flag==1
    it means we are going to plot the real source position from ground truth"""

    # first we crop the matrix to cut out non zero elements as much as possible
    row_min, row_max, col_min, col_max = find_non_zero_square(matrix)
    if np.isnan(row_min):
        return np.nan, np.nan, np.nan, np.nan, np.zeros(size)
    # find the center of the non zero matrix
    row_offset = (row_max - row_min) // 2
    col_offset = (col_max - col_min) // 2

    # here we define the limits of the patches in which we add some padding according to size so that they will be all of the same size
    bot = row_min + row_offset - size // 2 + 1
    top = bot + size
    left = col_min + col_offset - size // 2 + 1
    right = left + size

    # here we crop the matrix
    cropped_matrix_signal = matrix[:, bot:top, left:right]
    middle_indx = int(len(cropped_matrix_signal) / 2 - 0.5)

    # here we start the plotting
    '''plt.imshow(cropped_matrix_signal[middle_indx], cmap='grey', interpolation='nearest')
    plt.colorbar()

    _,rows, cols = matrix.shape
    x_ticks_labels = np.arange(left, right+ 1)
    y_ticks_labels = np.arange(bot,top +1)
    # for the y ticks we need to do this to have displayed the indices from bottom (0) to top (799) as per the 
    #local reference frame of allpix2
    y_ticks_labels = rows-1 - y_ticks_labels
    plt.xticks(ticks=np.arange(len(x_ticks_labels)), labels=x_ticks_labels)  
    plt.yticks(ticks=np.arange(len(y_ticks_labels)), labels=y_ticks_labels)  

    # Add axis labels
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    x_index = (x_source_index - left)
    #same thing here, need this to adjust the y indices
    y_index = (y_source_index - np.min(y_ticks_labels))
    y_index = len(y_ticks_labels)-y_index-1
    #here we plot the source position with a red dot
    if flag == 0:
        #estimated source
        plt.scatter(x_index, y_index, color='red', s=70, label="Estimated source position")
    if flag == 1:
        #real source position
        plt.scatter(x_index, y_index, color='red', s=70, label="True source position")


    plt.legend()

    plt.show()'''

    return bot, top, left, right, cropped_matrix_signal


def check_water_before_detector(group):
    """ This is a function to assign a new code column to the dataframe. In this case the column brings true information about wether or not the
    interactions come from a primary or scattered photon:

    0: the photon is scattered in the object
    1: the photon is primary
    2: the photon is primary but after the detector it interacts with water and then re-interacts with a detector

    input: the event_id subgroup from the dataset
    output: the group is returned with added column
    """

    # Find index of first occurrences of "water" and "detector" using partial match
    detector_mask = group['volume_origin'].str.contains('sensor', case=False, na=False)
    water_mask = group['volume_origin'].str.contains('water|world', case=False, na=False)

    detector_indices = group.index[detector_mask].tolist()
    water_indices = group.index[water_mask].tolist()

    # Default value for Photon_primary
    photon_primary = 1  # Default: detector appears before water or only detector is present

    if water_indices:  # If there is at least one "water"
        first_water_idx = min(water_indices)

        # Case 1: Water appears before any detector -> Assign 0
        if detector_indices and first_water_idx < min(detector_indices):
            photon_primary = 0
        # Case 2: Detector appears first but another detector is present after water -> Assign 2
        elif any(idx > first_water_idx for idx in detector_indices):
            photon_primary = 2

    # Assign Photon_primary to all rows in the group
    group['photon_primary'] = photon_primary
    # if ((group['photon_primary'] == 0) & group['volume_origin'].str.contains('detector', case=False, na=False)).any():
    #    a=0

    # if ((group['photon_primary'] == 1) & group['volume_origin'].str.contains('detector', case=False, na=False)).any():
    #    group.loc[group['photon_primary'] == 0, 'photon_primary'] = 1
    return group


# these below are the only things one need to change all the rest is parametrized
### 0----------------------------------------0000000000000000000--------------------------------------------0 ###


root_file_name = "/home/luca/Desktop/PhD/allpix2/Dead_volume_new/output/SPECT_20x20x5cm_30cm_140kev_photons_and_electrons_pt2.root"

# These next 3 lines are important as they add a world global time to the local time stamps generated by allpix, which restart to zero for each new photon generated.
source_activity = 500  # in MBq

# this needs to be changed accordingly to how many events you want to read.
N_events_to_read = 100000

water_radius = 300

ind_det_max = 67
detector_names = generate_detector_list_full(ind_det_max)

int_patch_side = 40
pixel_grid_len = 2000
# all_pixels = np.zeros(200000)
px_ind = 0
signal_pulse_length = 4000  # this is in time steps for the integration parameter of the simulations (so 0.05 usually). depending on which simulation it is run you need to adjust this
pixel_size = 0.1  # in mm
pixel_height = 0.75  # in mm
starting_iev = 0
### 0----------------------------------------0000000000000000000--------------------------------------------0 ###
cum_sum_time = 0
lambda_source = source_activity * 10e6  # emissions per second

# the columns in the dataframe, advisable not to change them unless you know what you are doing
column_names_finalPD = ["event_ID", "init_x", "init_y", "init_z", "init_kin", "time", "E_dep", "photon_ID",
                        "process_name", "volume_interaction", "photon_primary", 'track_ID']
column_names_smallPD = ["init_x", "init_y", "init_z", "time", "init_kin", "process_name", "particle_ID", "track_ID",
                        "parent_ID", "volume_origin"]

lib_path = "/home/luca/Documents/allpix/allpix-squared/lib/libAllpixObjects.so"

# here are just some checks
if lib_path is not None:  # Try to find Allpix Library
    lib_file_name = (str(lib_path))
    if (not os.path.isfile(lib_file_name)):
        print("WARNING: ", lib_file_name, " does not exist, exiting")
        exit(1)
elif os.path.isfile(path.abspath(
        path.join(__file__, "..", "..", "opt", "allpix-squared", "lib", "libAllpixObjects.so"))):  # For native installs
    lib_file_name = path.abspath(path.join(__file__, "..", "..", "opt", "allpix-squared", "lib", "libAllpixObjects.so"))

elif os.path.isfile(path.join(path.sep, "opt", "allpix-squared", "lib", "libAllpixObjects.so")):  # For Docker installs
    lib_file_name = path.join(path.sep, "opt", "allpix-squared", "lib", "libAllpixObjects.so")

else:
    print("WARNING: No Allpix Objects Library found, exiting")
    exit(1)

if (not os.path.isfile(lib_file_name)):
    print("WARNING: no allpix library found, exiting (Use -l to manually set location of libraries)")
    exit(1)
if (not os.path.isfile(root_file_name)):
    print("WARNING: " + root_file_name + " does not exist, exiting")
    exit(1)

gSystem.Load(lib_file_name)
rootfile = ROOT.TFile(root_file_name)
gDirectory.ls()

### 0----------------------------------------0000000000000000000--------------------------------------------0 ###
# to be changed in case one wants to modify the objects read
McTrack = rootfile.Get('MCTrack')
total_events = McTrack.GetEntries()
PixelHit = rootfile.Get('PixelHit')
for detector_name in detector_names:
    # here is to give an option for all possible detectors present in case the one specified cannot be found
    if not rootfile.GetDirectory("detectors/" + detector_name):
        print("\nDetector does not exist. Please choose one of the following detectors:")
        gDirectory.cd("detectors")
        list_of_keys = gDirectory.GetListOfKeys()
        for key in list_of_keys:
            print(key.GetName())
        exit(1)
if total_events < N_events_to_read + starting_iev:
    print("Number of events to be read exceedes the total number of events simulated, exiting")
    exit(1)
if N_events_to_read > 1000000:
    user_input = input(
        "Watch out, reading more than 1000000 events might require a lot of memory, do you want to continue? (Y/N): ")
    if user_input.lower() in ["Y", "y"]:
        print("Continuing...")
    else:
        print("Exiting...")
        exit(1)

    if user_input.lower() in ["Y", "y"]:
        print("Continuing...")
    else:
        print("Exiting...")
        exit(1)

print("Preallocating memory for dataframe")
smallPD_len = 200  # this is just to be bigger than the number of rows an event can have
# finalPD_len = 8*N_events_to_read #this is set to contain all events interacting with a detector


# df_final = pd.DataFrame(columns=column_names_finalPD, index=range(finalPD_len))
# df_init_en = pd.DataFrame(columns=column_names_smallPD, index=range(finalPD_len))
# df_pixel_activations = pd.DataFrame(columns=["event_ID", "detector_name", "pixel_x", "pixel_y", "charge",'time'], index=range(finalPD_len*4))

final_records = []
pixel_records = []

events_not_interacting_detector = 0
eBrem_skip = 0

# just used to reference where we are in the two dataframes. I tried some things and this seemed like the fastest way
smallPD_row_counter = 0
finalPD_row_counter = 0
activation_pixel_counter = 0
for count in tqdm(range(N_events_to_read), desc="Processing", unit="step", leave=True,
                  miniters=5000):  # int(McTrack.GetEntries()/2)
    iev = count + starting_iev

    # if iev % 1000 == 0:
    #    current, peak = tracemalloc.get_traced_memory()
    #    print(f"Event {iev}: current={current / 1e6:.1f} MB, peak={peak / 1e6:.1f} MB")
    #    print(f"Event {iev}, Memory: {proc.memory_info().rss / 1024 ** 2:.1f} MB")
    # we initialize the small dataframe, used to read directly from root and process to keep only the interactions and then being merged
    # into the final one. this is to speed up the process compared to manipulate each time a very long dataframe
    df_small = pd.DataFrame(columns=column_names_smallPD, index=range(smallPD_len))
    McTrack.GetEntry(iev)
    PixelHit.GetEntry(iev)
    McTrack_branch = McTrack.GetBranch("global")
    PixelHit_branch = []
    if (not McTrack_branch):
        Warning("WARNING: cannot find McTrack branch in the TTree,  exiting")
        exit(1)
    for detector_name in detector_names:
        PixelHit_branch.append(PixelHit.GetBranch(detector_name))
    br_mc_track = getattr(McTrack, McTrack_branch.GetName())
    br_pix_hit = []
    time_to_add = np.random.exponential(1 / lambda_source) * 10e9  # in ns
    cum_sum_time += time_to_add
    cc = 0
    for detector_name in detector_names:
        br_pix_hit.append(getattr(PixelHit, PixelHit_branch[cc].GetName()))

        cc += 1
    # McTrack.Clear()
    # PixelHit.Clear()
    n = br_mc_track.size()
    # for mc_track in br_mc_track:
    for i in range(n):
        mc_track = br_mc_track[i]
        # ROOT.SetOwnership(mc_track, False)

        # here we record all the important information that we will use to characterize the interaction and debug if needed

        df_small.at[smallPD_row_counter, 'init_x'] = mc_track.getStartPoint().x()
        df_small.at[smallPD_row_counter, 'init_y'] = mc_track.getStartPoint().y()
        df_small.at[smallPD_row_counter, 'init_z'] = mc_track.getStartPoint().z()
        df_small.at[smallPD_row_counter, 'time'] = mc_track.getGlobalStartTime() + cum_sum_time
        df_small.at[smallPD_row_counter, 'init_kin'] = mc_track.getKineticEnergyInitial()
        df_small.at[smallPD_row_counter, 'process_name'] = mc_track.getCreationProcessName()
        df_small.at[smallPD_row_counter, 'particle_ID'] = mc_track.getParticleID()
        df_small.at[smallPD_row_counter, 'parent_ID'] = hex(
            ROOT.addressof(mc_track.getParent()))  # hex(ROOT.addressof(mc_track.getParent()))
        df_small.at[smallPD_row_counter, 'track_ID'] = hex(ROOT.addressof(mc_track))  # hex(ROOT.addressof(mc_track))
        df_small.at[smallPD_row_counter, 'volume_origin'] = mc_track.getOriginatingVolumeName()

        smallPD_row_counter += 1

    # if iev ==  140558:
    #    adfsd=0

    df_small = df_small.sort_values(by="time").reset_index(drop=True)
    primary_photon_ID = df_small['track_ID'].iloc[0]
    # df_init_en.iloc[iev] = df_small.iloc[0].values
    mask = df_small['particle_ID'] == 11
    if mask.any():
        first_index = mask.idxmax()  # Get the index of the first occurrence
        df_small = df_small.loc[first_index:].reset_index(drop=True)
    else:
        events_not_interacting_detector += 1
        smallPD_row_counter = 0
        continue

    df_small = df_small.groupby('parent_ID', group_keys=False)[
        ["init_x", "init_y", "init_z", "time", "init_kin", "process_name", "particle_ID", "track_ID", "parent_ID",
         "volume_origin"]].apply(check_water_before_detector)
    # TODO: fix here the is primary check since for fluorescent photons it will say yes but it is not if it originated from a non primary event --> should be solved now
    df_small[['init_x', 'init_y', 'init_z']] = df_small[
        ['init_x', 'init_y', 'init_z']].apply(pd.to_numeric, errors='coerce')
    df_small = df_small[np.sqrt(df_small['init_y'] ** 2 + df_small['init_z'] ** 2) >= water_radius]
    if len(df_small) == 0:
        events_not_interacting_detector += 1
        smallPD_row_counter = 0
        continue

    detect_index = 0
    for br in br_pix_hit:
        n = br.size()
        # for pix_hit in br:
        for i in range(n):
            pix_hit = br[i]
            # ROOT.SetOwnership(pix_hit, False)
            # we now iterate every activated pixel and get the signal registered in it
            position_x = pix_hit.getPixel().getIndex().x()
            position_y = pix_hit.getPixel().getIndex().y()

            # particle = pix_hit.getMCParticles() #hex(ROOT.addressof(pix_hit.getMCParticles()))
            # VecMC = cppyy.gbl.std.vector[cppyy.gbl.allpix.MCParticle]
            # particles = hex(VecMC(pix_hit.getMCParticles()))
            # ids = [hex(ROOT.addressof(p)) for p in particles]
            # particle_ids = [hex(ROOT.addressof(p)) for p in pix_hit.getMCParticles()]

            time = pix_hit.getGlobalTime() + cum_sum_time
            # all_pixels[px_ind] = pix_hit.getSignal() *3.6 /1000
            # px_ind += 1

            signal = pix_hit.getSignal()
            if signal >= 2:
                # df_pixel_activations.at[activation_pixel_counter, 'event_ID'] = iev
                # df_pixel_activations.at[activation_pixel_counter, 'detector_name'] = detector_names[detect_index]
                # df_pixel_activations.at[activation_pixel_counter, 'pixel_x'] = position_x
                # df_pixel_activations.at[activation_pixel_counter, 'pixel_y'] = position_y
                # df_pixel_activations.at[activation_pixel_counter, 'charge'] = signal
                # df_pixel_activations.at[activation_pixel_counter, 'time'] = time

                # activation_pixel_counter += 1
                pixel_records.append({
                    "event_ID": iev,
                    "detector_name": detector_names[detect_index],
                    "pixel_x": position_x,
                    "pixel_y": position_y,
                    "charge": signal,
                    "time": time
                })

            # timing_matrix_pxhit_full[detect_index,pixel_grid_len - 1 - position_y, position_x] += pix_hit.getGlobalTime()

        detect_index += 1
    # for now, we skip fluorescence and bremstrallung containing events, this can be changed but it would require
    # to change the algorithm that follows
    # if (df_small['particle_ID'] == 22).any():
    #    smallPD_row_counter = 0
    #    eBrem_skip +=1
    #    continue

    mask = df_small['particle_ID'] == 11
    df_small = df_small[mask].reset_index(drop=True)
    df_small = df_small.sort_values(by="time")
    if len(df_small) == 0:
        events_not_interacting_detector += 1
        smallPD_row_counter = 0
        continue

    df_small.loc[df_small['parent_ID'] != primary_photon_ID, 'process_name'] = 'compt'
    edep_sum = df_small['init_kin'].iloc[
        0]  # summing all the deposited energy from different electrons in the same interaction
    current_int_pos = [df_small['init_x'].iloc[0], df_small['init_y'].iloc[0], df_small['init_z'].iloc[0]]
    current_int_time = df_small['time'].iloc[0]
    current_volume_int = df_small['volume_origin'].iloc[0]
    current_process_int = df_small['process_name'].iloc[0]
    current_parent_ID = df_small['parent_ID'].iloc[0]
    current_photon_primary = df_small['photon_primary'].iloc[0]
    current_init_kin = df_small['init_kin'].iloc[0]
    current_track_ID = df_small['track_ID'].iloc[0]
    # current_parent_ID = df_small['parent_ID'].iloc[0]
    # this next check is to avoid reading a ionization interaction taking place in a detector that originates from an electron
    # created by a photon interaction in the phantom very close to the detector face.
    if current_process_int == 'eIoni':
        smallPD_row_counter = 0
        continue

    for j in range(1, len(df_small), 1):

        if df_small['process_name'].iloc[j] == 'eIoni':
            # if we have an electron that creates by ionization another electron we simplify skip since we are looking at the initial deposited energies

            continue

        point1 = np.array([df_small['init_x'].iloc[j], df_small['init_y'].iloc[j], df_small['init_z'].iloc[j]])
        point2 = np.array(current_int_pos)
        if np.linalg.norm(point1 - point2) < 0.05:
            # if the electron track we are looking at started close enough to the previous one we sum the energies togheter as they correspond to one single interaction
            # then we go to the next interaction
            edep_sum += df_small['init_kin'].iloc[j]





        else:
            # otherwise we save all the info on the interaction on the current point and we reset to move on and check on the next interaction

            '''df_final.at[finalPD_row_counter, 'event_ID'] = iev
            df_final.at[finalPD_row_counter, 'E_dep'] = edep_sum
            edep_sum = df_small['init_kin'].iloc[j]
            df_final.at[finalPD_row_counter, 'init_kin'] = current_init_kin
            df_final.at[finalPD_row_counter, 'init_x'] = current_int_pos[0]
            df_final.at[finalPD_row_counter, 'init_y'] = current_int_pos[1]
            df_final.at[finalPD_row_counter, 'init_z'] = current_int_pos[2]
            df_final.at[finalPD_row_counter, 'time'] = current_int_time
            df_final.at[finalPD_row_counter, 'volume_interaction'] = current_volume_int
            df_final.at[finalPD_row_counter, 'process_name'] = current_process_int
            df_final.at[finalPD_row_counter, 'photon_primary'] = current_photon_primary
            df_final.at[finalPD_row_counter, 'photon_ID'] = current_parent_ID
            df_final.at[finalPD_row_counter, 'track_ID'] = current_track_ID'''
            # df_final.at[finalPD_row_counter, 'parent_ID'] = current_parent_ID

            final_records.append({
                "event_ID": iev,
                "E_dep": edep_sum,
                "init_kin": current_init_kin,
                "init_x": current_int_pos[0],
                "init_y": current_int_pos[1],
                "init_z": current_int_pos[2],
                "time": current_int_time,
                "volume_interaction": current_volume_int,
                "process_name": current_process_int,
                "photon_primary": current_photon_primary,
                "photon_ID": current_parent_ID,
                "track_ID": current_track_ID
            })
            edep_sum = df_small['init_kin'].iloc[j]
            current_photon_primary = df_small['photon_primary'].iloc[j]
            # df_final.at[finalPD_row_counter, 'init_kin_event'] = initial_kin_event
            current_int_pos = [df_small['init_x'].iloc[j], df_small['init_y'].iloc[j],
                               df_small['init_z'].iloc[j]]  # the element in 0 will always be the primary photon
            current_int_time = df_small['time'].iloc[j]
            current_volume_int = df_small['volume_origin'].iloc[j]
            current_process_int = df_small['process_name'].iloc[j]
            current_parent_ID = df_small['parent_ID'].iloc[j]
            current_init_kin = df_small['init_kin'].iloc[j]
            current_track_ID = df_small['track_ID'].iloc[j]
            # current_parent_ID = df_small['parent_ID'].iloc[j]
            finalPD_row_counter += 1

    '''df_final.at[finalPD_row_counter, 'photon_primary'] = current_photon_primary
    df_final.at[finalPD_row_counter, 'event_ID'] = iev
    df_final.at[finalPD_row_counter, 'E_dep'] = edep_sum
    df_final.at[finalPD_row_counter, 'init_kin'] = current_init_kin
    df_final.at[finalPD_row_counter, 'init_x'] = current_int_pos[0]
    df_final.at[finalPD_row_counter, 'init_y'] = current_int_pos[1]
    df_final.at[finalPD_row_counter, 'init_z'] = current_int_pos[2]
    df_final.at[finalPD_row_counter, 'time'] = current_int_time
    df_final.at[finalPD_row_counter, 'volume_interaction'] = current_volume_int
    df_final.at[finalPD_row_counter, 'process_name'] = current_process_int
    df_final.at[finalPD_row_counter, 'photon_ID'] = current_parent_ID
    df_final.at[finalPD_row_counter, 'track_ID'] = current_track_ID'''

    final_records.append({
        "event_ID": iev,
        "E_dep": edep_sum,
        "init_kin": current_init_kin,
        "init_x": current_int_pos[0],
        "init_y": current_int_pos[1],
        "init_z": current_int_pos[2],
        "time": current_int_time,
        "volume_interaction": current_volume_int,
        "process_name": current_process_int,
        "photon_primary": current_photon_primary,
        "photon_ID": current_parent_ID,
        "track_ID": current_track_ID
    })

    # df_final.at[finalPD_row_counter, 'parent_ID'] = current_parent_ID
    # df_final.at[finalPD_row_counter, 'init_kin_event'] = initial_kin_event

    finalPD_row_counter += 1
    smallPD_row_counter = 0
    # del df_small, br_mc_track, br_pix_hit
    # gc.collect()
    # del mc_track#, pix_hit
    # McTrack.Clear()
    # PixelHit.Clear()

now = datetime.now()
curr_time = now.strftime("_%H:%M:%S")
# df_final = df_final.dropna(how = 'all')
# df_pixel_activations = df_pixel_activations.dropna(how = 'all')
print('Total number of events read', N_events_to_read)
print('Events non interacting with the detector:', events_not_interacting_detector)
print('eBrem and fluorescent skipped counts:', eBrem_skip)
output_dir = "root_read_df"
file_name = os.path.basename(root_file_name)
file_name = os.path.splitext(file_name)[0]
source_activity = str(source_activity)
# df_init_en = df_init_en.dropna(how = 'all')


file_name_df = os.path.join(output_dir, file_name + source_activity + '_MBq_' + str(
    starting_iev) + '_starting_iev' + '_no_noise_pd.pkl')
df_final = pd.DataFrame(final_records, columns=column_names_finalPD)
df_pixel_activations = pd.DataFrame(pixel_records,
                                    columns=["event_ID", "detector_name", "pixel_x", "pixel_y", "charge", 'time'])
df_final.to_pickle(file_name_df)
output_dir = "root_read_df_px_activations"
file_name_df = os.path.join(output_dir, file_name + source_activity + '_MBq_' + str(
    starting_iev) + '_starting_iev' + '_no_noise_pd_px_activations.pkl')

df_pixel_activations.to_pickle(file_name_df)

# file_name_df = file_name +  source_activity + '_MBq'+'_init_en'+ '_no_noise_pd.pkl'
# df_init_en.to_pickle(file_name_df)