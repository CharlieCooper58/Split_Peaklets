import strax
import straxen
import cutax
import numpy as np
import sys
import admix
import os
import numpy as np
import shutil
print("Finished importing, now start to load context.")
# Modify below for the strax.storage path
print(straxen.print_versions())
from ds_plugin import SplitPeaklets
from SplitPeakletsTest import SplitPeakletsTest, SplitPeakletsTest2, SplitPeakletsTest3, SplitPeakletsTest4
print("This is working")
st = cutax.xenonnt_v8(output_folder='/scratch/midway2/cmcooper')

st.register(SplitPeaklets)
st.register(SplitPeakletsTest)
st.register(SplitPeakletsTest2)
st.register(SplitPeakletsTest3)
st.register(SplitPeakletsTest4)
_, runid = sys.argv
print("Loaded the context successfully, and the run id to process:", runid)
os.chdir('/scratch/midway2/cmcooper')
print(os.getcwd())
os.system('admix-download {name} raw_records'.format(name = runid))
#admix.download(runid, 'raw_records')
print("Downloading")
st.make(runid, targets = ('split_peaklets_test'),config=dict(cluster_min=50))
#st.make(runid, targets = ('split_peaklets_test_2'),config=dict(cluster_min=50))
#st.make(runid, targets = ('split_peaklets_test_3'),config=dict(cluster_min=50))
#st.make(runid, targets = ('split_peaklets_test_4'),config=dict(cluster_min=50))


shutil.rmtree('{name}-raw_records-rfzvpzj4mf'.format(name = runid))
"""
st_fake_context.make(runid,targets="true_peaks",config=dict(parent_s1_type='Ar', s1_min_coincidence=3,
                    replace_hit=True, n_repeat=10, upper_rhits_parent_fraction=0.9))
"""
print('Done!')
