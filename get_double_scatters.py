import straxen
import strax
import cutax
import numpy as np
from SplitPeaklets import SplitPeaklets
st = straxen.contexts.xenonnt('global_v8', output_folder = '/scratch/midway2/cmcooper')
st.register(SplitPeaklets)

corresponding_peaks = []
save_count = 0
for j in runs:
    if(save_count != 0 and save_count%5 == 0):
        corresponding_peaks_tosave = np.array(corresponding_peaks)
        np.save("mislabeled_doubles", corresponding_peaks_tosave)
    s_min = 0
    save_count += 1
    print(j)
    while(s_min < 2000):
        if(not st.is_stored(j, target = ('split_peaklets'))):
            print("Missing data for {run_name}".format(run_name = j))
            break
        try:
            events = st.get_array(j, targets = ('split_peaklets'),config=dict(cluster_min=50),
                      seconds_range = (s_min, s_min + 500),
                      progress_bar=True)
            ds_peaks = events[events['kmeans_clusters'] > 1]

            s_min += 500
            print(len(ds_peaks))
            for i in ds_peaks:
                corresponding_peaks.append(i)
            print(len(corresponding_peaks))
            
        except ValueError:
            break
corresponding_peaks = np.array(corresponding_peaks)
np.save("mislabeled_doubles", corresponding_peaks)
