import numpy as np
import time
import os, shlex
#from subprocess import Popen, PIPE, STDOUT, TimeoutExpired
import utilix
from utilix.batchq import *
print(utilix.__file__)
#import straxen
#import strax
class Submit(object):
    '''
        Take maximum number of nodes to use at once
        Submit each group to a node and excute
    '''
    def name(self):
        return self.__class__.__name__

    def execute(self, *args, **kwargs):
        eval('self.{name}(*args, **kwargs)'.format(name = self.name().lower()))

    def submit(self, loop_over=[], max_num_submit=10, nmax=3):
        _start = 0
        self.max_num_submit = max_num_submit
        self.loop_over = loop_over
        self.p = True

        index = _start
        while (index < len(self.loop_over) and index < nmax):
            if (self.working_job() < self.max_num_submit):
                #os.chdir('/scratch/midway2/cmcooper/raw_data')
                #os.system('module load singularity')
                #os.system('singularity shell --bind /dali --bind /project2 /project2/lgrandi/xenonnt/singularity-images/xenonnt-development.simg')
                #os.system('admix-download {run_num} raw_records'.format(run_num = loop_over[index]))
                self._submit_single(loop_index=index,
                                    loop_item=self.loop_over[index])

                time.sleep(1.0)
                index += 1

    # check my jobs
    def working_job(self):
        cmd='squeue --user={user} | wc -l'.format(user = 'cmcooper')
        jobNum=int(os.popen(cmd).read())
        return  jobNum -1

    def _submit_single(self, loop_index, loop_item):
        jobname = 'ds_peaklets{:03}'.format(loop_index)
        run_id = loop_item
        # Modify here for the script to run
        jobstring = "python /home/cmcooper/make_ds_peaklets.py %s"%(run_id)
        print(jobstring)

        # Modify here for the log name
        utilix.batchq.submit_job(
            jobstring, log='/home/cmcooper/split_peaklets%s.log'%(run_id), partition='xenon1t', qos='xenon1t',
            account='pi-lgrandi', jobname=jobname,
            delete_file=True, dry_run=False, mem_per_cpu=35000,
            container='xenonnt-development.simg',
            cpus_per_task= 1)

p = Submit()

# Modify here for the runs to process

"""
loop_over = np.array(['034277', '034370', '033850', 
              '033784', '034707', 
              '034710'])
"""
#st = straxen.contexts.xenonnt('global_v7')
#loop_over = np.array(['021288', '021457', '021455', '021453', '021451'])
loop_over = np.array(['021377', '021375', '021373', '021371', '021369', '021367', '021365', '021363', '021361', '021359', '021357', '021355', '021353', '021349', '021347', '021345', '021343', '021341', '021288'])
#loop_over = np.array(['021427', '021425', '021423', '021421', '021417', '021415', '021413', '021411', '021409', '021405', '021403','021401', '021397', '021395', '021393', '021391', '021389', '021387', '021381', '021379'])
#runs = st.select_runs(#exclude_tags = ['missing_one_pmt,_sr0_preliminary,ambe_source_top_cw3d0', '_sr0_preliminary,ambe_source_top_far'],
#                      available="event_info", 
#                      run_mode="*ambe_link")
#loop_over = runs[40:61]
print('Runs to process: ', len(loop_over))

p.execute(loop_over=loop_over, max_num_submit=61, nmax=10000)
