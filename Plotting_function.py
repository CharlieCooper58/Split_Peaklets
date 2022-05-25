#REQUIRES RECORDS
import straxen
import strax
import cutax
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
st = straxen.contexts.xenonnt('global_v8', output_folder = '/scratch/midway2/cmcooper')
%matplotlib notebook

def get_records(start, end):
    for i in runs:
        try:
            rec = st.get_array(i, targets = 'records', time_range = [start, end])
            return rec
        except ValueError:
            continue
    return null

def check_double_peaks_tr(start, end, scale_fact = 50000):
    pmt_pos = np.array(straxen.pmt_positions()[['x','y']])
    top_pmt_pos = pmt_pos[:253]
    rec = get_records(start, end)
    hits_in_peaklet = rec[(rec['time'] >= start) & (rec['time'] <= end) & (rec['channel']<253)]
    pmt_wts = []
    thresh = np.sum(hits_in_peaklet['area'])/scale_fact
    hits_in_peaklet = hits_in_peaklet[hits_in_peaklet['area'] > thresh]
    start_time = np.min(hits_in_peaklet['time'])
    end_time_ind = np.argmax(hits_in_peaklet['time'])
    end_time = hits_in_peaklet['time'][end_time_ind] + hits_in_peaklet['length'][end_time_ind]*hits_in_peaklet['dt'][end_time_ind]
    length = end_time - start_time
    #Set dt_factor to either 600 or 0.04*length
    #dt_factor = 0.04*length
    dt_factor = 600
    #print(len(hits_in_peaklet))
    hit_positions = pmt_pos[hits_in_peaklet['channel'][hits_in_peaklet['channel']<253]]
    time_shifted_hits = hits_in_peaklet[hits_in_peaklet['channel']<253]['time'] - np.min(hits_in_peaklet[hits_in_peaklet['channel']<253]['time'])
    time_shifted_hits = time_shifted_hits/dt_factor
    arr = np.stack((np.array(hit_positions.T[0]), np.array(hit_positions.T[1]), np.array(time_shifted_hits)), axis=-1)
    clusters = KMeans(n_clusters=2, random_state=0).fit(arr, sample_weight=hits_in_peaklet['area'])
    labels = clusters.labels_
    print(np.unique(labels))
    fig = plt.figure(figsize = (12, 12))
    ax = Axes3D(fig)
    time_shifted_hits = time_shifted_hits * dt_factor
    p = ax.scatter(hit_positions.T[0], hit_positions.T[1], time_shifted_hits,
               c = hits_in_peaklet['area'],
               s = 100*hits_in_peaklet['area']/np.max(hits_in_peaklet['area']),
               #vmin = 1,
               #vmax = 1048253,
               norm = LogNorm(),
               )
    #plt.colorbar()
    ax.set_zlim(0,time_shifted_hits[-1])
    #ax.set_zscale(0.4)
    ax.set_xlabel('x [cm]', fontsize = 12)
    ax.set_ylabel('y [cm]', fontsize = 12)
    ax.set_zlabel('Time [ns]', fontsize = 12, labelpad = 7)
    plt.colorbar(p, ax = ax, label = "Area [PE]", shrink = 0.6)

    plt.show()
    for peak in range(2):
        fig = plt.figure(figsize = (10, 10))
        ax = Axes3D(fig)
        p2 = ax.scatter(hit_positions.T[0][labels==peak], hit_positions.T[1][labels==peak], time_shifted_hits[labels==peak],
                   c = hits_in_peaklet['area'][labels==peak],
                   s = 100*hits_in_peaklet['area'][labels==peak]/np.max(hits_in_peaklet['area'][labels==peak]),
                   #vmin = 1,
                   #vmax = 1048253,
                   norm = LogNorm(),
                   )
        #plt.colorbar()
        ax.set_zlim(0,time_shifted_hits[-1])
        ax.set_xlabel('x [cm]', fontsize = 12)
        ax.set_ylabel('y [cm]', fontsize = 12)
        #ax.set_zlabel(r'Time [$\frac{\mu s}{2}$]')
        ax.set_zlabel('Time [ns]', fontsize = 12, labelpad = 7)
        plt.colorbar(p2, ax = ax, label = "Area [PE]", shrink = 0.6)

        plt.show()
