import straxen
import strax
import cutax #you may have to manually updatethis to v0.1.0
from straxen.plugins.pulse_processing import HITFINDER_OPTIONS
from strax.processing.general import _touching_windows
from straxen.get_corrections import is_cmt_option
import numba
import tensorflow as tf
import keras
import numpy as np
from scipy.ndimage.filters import maximum_filter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('peak_area_clustering_threshold', default=200,
                 help="Only recluster peaks with S2 above this threshold"),
    strax.Option('peaklet_gap_threshold', default=700,
                 help="No hits for this many ns triggers a new peak"),
    strax.Option('peak_left_extension', default=30,
                 help="Include this many ns left of hits in peaks"),
    strax.Option('peak_right_extension', default=200,
                 help="Include this many ns right of hits in peaks"),
    strax.Option('peak_min_pmts', default=2,
                 help="Minimum number of contributing PMTs needed to define a peak"),
    strax.Option('peak_split_gof_threshold',
                 # See https://xe1t-wiki.lngs.infn.it/doku.php?id=
                 # xenon:xenonnt:analysis:strax_clustering_classification
                 # #natural_breaks_splitting
                 # for more information
                 default=(
                     None,  # Reserved
                     ((0.5, 1.0), (6.0, 0.4)),
                     ((2.5, 1.0), (5.625, 0.4))),
                 help='Natural breaks goodness of fit/split threshold to split '
                      'a peak. Specify as tuples of (log10(area), threshold).'),
    strax.Option('peak_split_filter_wing_width', default=70,
                 help='Wing width of moving average filter for '
                      'low-split natural breaks'),
    strax.Option('peak_split_min_area', default=40.,
                 help='Minimum area to evaluate natural breaks criterion. '
                      'Smaller peaks are not split.'),
    strax.Option('peak_split_iterations', default=20,
                 help='Maximum number of recursive peak splits to do.'),
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"),
    strax.Option('gain_model',
                 help='PMT gain model. Specify as '
                 '(str(model_config), str(version), nT-->boolean'),
    strax.Option('tight_coincidence_window_left', default=50,
                 help="Time range left of peak center to call "
                      "a hit a tight coincidence (ns)"),
    strax.Option('tight_coincidence_window_right', default=50,
                 help="Time range right of peak center to call "
                      "a hit a tight coincidence (ns)"),
    strax.Option('n_tpc_pmts', default = 494, type=int,
                 help='Number of TPC PMTs'),
    strax.Option('saturation_correction_on', default=True,
                 help='On off switch for saturation correction'),
    strax.Option('saturation_reference_length', default=100,
                 help="Maximum number of reference sample used "
                      "to correct saturated samples"),
    strax.Option('saturation_min_reference_length', default=20,
                 help="Minimum number of reference sample used "
                      "to correct saturated samples"),
    strax.Option('peaklet_max_duration', default=int(10e6),
                 help="Maximum duration [ns] of a peaklet"),
    *HITFINDER_OPTIONS,
)
class SplitPeaklets(strax.Plugin):
    """
    Split records into:
        -peaklets
        -lone_hits
    Peaklets are very aggressively split peaks such that we are able
    to find S1-S2s even if they are close to each other. (S2) Peaks
    that are split into too many peaklets will be merged later on.
    To get Peaklets from records apply/do:
        1. Hit finding
        2. Peak finding
        3. Peak splitting using the natural breaks algorithm
        4. Compute the digital sum waveform
    Lone hits are all hits which are outside of any peak. The area of
    lone_hits includes the left and right hit extension, except the
    extension overlaps with any peaks or other hits.
    """
    depends_on = ('records')
    provides = 'split_peaklets'
    data_kind = 'split_peaklets'
    parallel = 'process'
    compressor = 'zstd'
    #Real version: 0.1.11
    __version__ = '0.1.16'


    def setup(self):
        if self.config['peak_min_pmts'] > 2:
            # Can fix by re-splitting,
            raise NotImplementedError(
                f"Raising the peak_min_pmts to {self.config['peak_min_pmts']} "
                f"interferes with lone_hit definition. "
                f"See github.com/XENONnT/straxen/issues/295")

        self.to_pe = straxen.get_correction_from_cmt(self.run_id,
                                       self.config['gain_model'])

        # Check config of `hit_min_amplitude` and define hit thresholds
        # if cmt config
        if is_cmt_option(self.config['hit_min_amplitude']):
            self.hit_thresholds = straxen.get_correction_from_cmt(self.run_id,
                self.config['hit_min_amplitude'])
        # if hitfinder_thresholds config
        elif isinstance(self.config['hit_min_amplitude'], str):
            self.hit_thresholds = straxen.hit_min_amplitude(
                self.config['hit_min_amplitude'])
        else: # int or array
            self.hit_thresholds = self.config['hit_min_amplitude']

        #input_pattern = keras.Input(shape=(253,), name = "pattern" ) 
        #converter_layer = AxialTopConverter()
        #self.converter_model = keras.Model(input_pattern, converter_layer(input_pattern))

        pos_array_info = straxen.pmt_positions()
        self.top_pmt_array = np.vstack((pos_array_info.iloc[0:253].x, pos_array_info.iloc[0:253].y)).T
        self.coords = np.array(self.calcPairs()[:,1:])

        self.p_thresh = 0.1
        self.abs_thresh = 500
        self.pad_time = 1000

        xlen, ylen = 19, 19
        xcoords, ycoords = np.arange(xlen), np.arange(ylen)
        xx, yy = np.meshgrid(xcoords, ycoords)
        xx, yy = xx.T.flatten(), yy.T.flatten()
        self.xy = np.array([xx,yy]).T

    dtype = strax.peak_dtype(n_channels=494)+[(('Labels which cluster the peaklet belongs to','cluster_number'), np.int16)]+[(('Inner-cluster distance parameter health','eta'), np.float32)]+[(('Number of clusters detected','kmeans_clusters'), np.int16)]


    def peak_max_indices(self, image):
        #Find Local maxima
        local_max = maximum_filter(image, size=(0,3,3))==image
        #Find size ordered peak indices #Consider coming back to have 2D array size be detector config option
        maxima_indices = np.argsort((image*local_max).reshape(len(image),19*19),axis=1).T[::-1].T
        return maxima_indices

    def compute(self, records, start, end):
        r = records

        hits = strax.find_hits(r, min_amplitude=self.hit_thresholds)

        # Remove hits in zero-gain channels
        # they should not affect the clustering!
        hits = hits[self.to_pe[hits['channel']] != 0]

        hits = strax.sort_by_time(hits)

        # Use peaklet gap threshold for initial clustering
        # based on gaps between hits
        peaklets = strax.find_peaks(
            hits, self.to_pe,
            gap_threshold=self.config['peaklet_gap_threshold'],
            left_extension=self.config['peak_left_extension'],
            right_extension=self.config['peak_right_extension'],
            min_channels=self.config['peak_min_pmts'],
            result_dtype=self.dtype_for('peaklets'),
            max_duration=self.config['peaklet_max_duration'],
        )

        # Make sure peaklets don't extend out of the chunk boundary
        # This should be very rare in normal data due to the ADC pretrigger
        # window.
        straxen.plugins.peaklet_processing.Peaklets.clip_peaklet_times(peaklets, start, end)

        # Get hits outside peaklets, and store them separately.
        # fully_contained is OK provided gap_threshold > extension,
        # which is asserted inside strax.find_peaks.
        is_lone_hit = strax.fully_contained_in(hits, peaklets) == -1
        lone_hits = hits[is_lone_hit]

        strax.integrate_lone_hits(
            lone_hits, records, peaklets,
            save_outside_hits=(self.config['peak_left_extension'],
                               self.config['peak_right_extension']),
            n_channels=len(self.to_pe))

        # Compute basic peak properties -- needed before natural breaks
        hits = hits[~is_lone_hit]
        # Define regions outside of peaks such that _find_hit_integration_bounds
        # is not extended beyond a peak.
        outside_peaks = straxen.plugins.peaklet_processing.Peaklets.create_outside_peaks_region(peaklets, start, end)
        strax.find_hit_integration_bounds(
            hits, outside_peaks, records,
            save_outside_hits=(self.config['peak_left_extension'],
                               self.config['peak_right_extension']),
            n_channels=len(self.to_pe),
            allow_bounds_beyond_records=True,
        )

        # Transform hits to hitlets for naming conventions. A hit refers
        # to the central part above threshold a hitlet to the entire signal
        # including the left and right extension.
        # (We are not going to use the actual hitlet data_type here.)
        hitlets = hits

        hitlet_time_shift = (hitlets['left'] - hitlets['left_integration']) * hitlets['dt']
        hitlets['time'] = hitlets['time'] - hitlet_time_shift
        hitlets['length'] = (hitlets['right_integration'] - hitlets['left_integration'])
        hitlets = strax.sort_by_time(hitlets)
        rlinks = strax.record_links(records)

        strax.sum_waveform(peaklets, hitlets, r, rlinks, self.to_pe)

        strax.compute_widths(peaklets)

        # Split peaks using low-split natural breaks;
        # see https://github.com/XENONnT/straxen/pull/45
        # and https://github.com/AxFoundation/strax/pull/225

        peaklets = strax.split_peaks(
            peaklets, hitlets, r, rlinks, self.to_pe,
            algorithm='natural_breaks',
            threshold=self.natural_breaks_threshold,
            split_low=True,
            filter_wing_width=self.config['peak_split_filter_wing_width'],
            min_area=self.config['peak_split_min_area'],
            do_iterations=self.config['peak_split_iterations'])

        # Saturation correction using non-saturated channels
        # similar method used in pax
        # see https://github.com/XENON1T/pax/pull/712
        # Cases when records is not writeable for unclear reason
        # only see this when loading 1T test data
        # more details on https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
        if not r['data'].flags.writeable:
            r = r.copy()

        if self.config['saturation_correction_on']:
            peak_list = straxen.plugins.peaklet_processing.peak_saturation_correction(
                r, rlinks, peaklets, hitlets, self.to_pe,
                reference_length=self.config['saturation_reference_length'],
                min_reference_length=self.config['saturation_min_reference_length'])

            # Compute the width again for corrected peaks
            strax.compute_widths(peaklets, select_peaks_indices=peak_list)

        ####
        #Find the clusters of PMTs
        ####
        #Distort 1d PMT array to 2d rectangular aary

        #Calculate eta values
        peaklets['eta'] = self.calc_etas(peaklets, hits)
        print(np.sum(peaklets['eta']!=0))

        #Identify local maxima
        pmt_hit_pattern =  self.distort(peaklets['area_per_channel'][:, :253], self.coords)

        #Find local maxima
        detected_peaks_ind = self.peak_max_indices(pmt_hit_pattern)
        reclustered_peaks = self.recluster_max_indices(detected_peaks_ind)
        max_peak_areas = self.first_second_max_areas(pmt_hit_pattern, *reclustered_peaks)

        valid_ds = ( max_peak_areas[1] > self.p_thresh* max_peak_areas[0])&( max_peak_areas[1]>self.abs_thresh)
        kmeans_clusters = valid_ds+1
        peaklets['kmeans_clusters'] = kmeans_clusters
        #Cluster using kmeans
        cluster_channels = self.cluster(kmeans_clusters, peaklets['area_per_channel'][:, :253], self.top_pmt_array)

        ds_peaklets = peaklets[kmeans_clusters> 1]

        #ds_cluster_channels = [ np.pad(x[0],(0, 241), constant_values=(0, 1)) for x in cluster_channels[kmeans_clusters>1]]
        dsp = []
        #Recluster double scatters
        for peaklet in ds_peaklets:
            hits_selection = hits['time']>(peaklet['time']-self.pad_time)
            hits_selection &= hits['time']<(peaklet['time']+peaklet['length']*peaklet['dt']+self.pad_time)
            ds_hits = hits[hits_selection]

            hits_used = ds_hits[(ds_hits['channel']<253)]
            start_time = np.min(hits_used['time'])
            end_time_ind = np.argmax(hits_used['time'])
            end_time = hits_used['time'][end_time_ind] + hits_used['length'][end_time_ind]*hits_used['dt'][end_time_ind]
            length = end_time - start_time
            #Select dt scale factor based on length of peaklet; can also get good results using values around 500
            dt_factor = 0.04*length
            pmt_pos = np.array(straxen.pmt_positions()[['x','y']])
            top_pmt_pos = pmt_pos[:253]
            hit_positions = pmt_pos[hits_used['channel']]
            time_shifted_hits = hits_used['time'] - np.min(hits_used['time'])
            time_shifted_hits = time_shifted_hits/dt_factor
            #Get the 3D hit positions
            arr = np.stack((np.array(hit_positions.T[0]), np.array(hit_positions.T[1]), np.array(time_shifted_hits)), axis=-1)
            #We only apply the cluster threshold here, since we're going to want to be able to access all the hits again later
            clusters = KMeans(n_clusters=2, random_state=0).fit(arr[hits_used['area'] > (np.sum(hits_used['area'])/50000)], sample_weight=hits_used[hits_used['area'] > (np.sum(hits_used['area'])/50000)]['area'])
            centers = clusters.cluster_centers_
            labels = clusters.labels_
            labels_test = []
            #Reassign all hits to clusters
            for i in range(len(hits_used)):
                dist1 = np.linalg.norm(arr[i] - centers[0])
                dist2 = np.linalg.norm(arr[i] - centers[1])
                if(dist1 < dist2):
                    labels_test.append(0)
                else:
                    labels_test.append(1)
            labels_test = np.array(labels_test)
            #Make new peaks
            for i in [0,1]:
                ds_peaklet= strax.find_peaks(
                    hits_used[labels_test == i], self.to_pe,
                    gap_threshold=self.config['peaklet_gap_threshold'],
                    left_extension=self.config['peak_left_extension'],
                    right_extension=self.config['peak_right_extension'],
                    min_channels=self.config['peak_min_pmts'],
                    result_dtype=self.dtype_for('peaklets'),
                    max_duration=self.config['peaklet_max_duration'],
                )

                #strax.sum_waveform(ds_peaklet, ds_hits, r, rlinks, self.to_pe)#*pmt_masks[i])
                strax.sum_waveform(ds_peaklet, hits_used[labels_test == i], r, rlinks, self.to_pe)#*pmt_masks[i])
                ds_peaklet[0]['cluster_number'] = i+1

                if self.config['saturation_correction_on']:
                    peak_list = straxen.plugins.peaklet_processing.peak_saturation_correction(
                    #r, rlinks, ds_peaklet, ds_hits, self.to_pe,#*pmt_masks[i],
                    r, rlinks, ds_peaklet, hits_used[labels_test == i], self.to_pe,
                    reference_length=self.config['saturation_reference_length'],
                    min_reference_length=self.config['saturation_min_reference_length'])

                    # Compute the width again for corrected peaks
                    strax.compute_widths(ds_peaklet, select_peaks_indices=peak_list)


                dsp.append(ds_peaklet[0])
        dsp = np.array(dsp)

        strax.compute_widths(dsp)      

        ##Add resolved peaks to peaklets
        if len(dsp):
            peaklets = np.concatenate([peaklets, dsp])
        del hits

        # Compute tight coincidence level.
        # Making this a separate plugin would
        # (a) doing hitfinding yet again (or storing hits)
        # (b) increase strax memory usage / max_messages,
        #     possibly due to its currently primitive scheduling.
        hit_max_times = np.sort(
            hitlets['time']
            + hitlets['dt'] * straxen.plugins.peaklet_processing.hit_max_sample(records, hitlets)
            + hitlet_time_shift  # add time shift again to get correct maximum
        )
        peaklet_max_times = (
                peaklets['time']
                + np.argmax(peaklets['data'], axis=1) * peaklets['dt'])
        tight_coincidence_channel = straxen.plugins.peaklet_processing.get_tight_coin(
            hit_max_times,
            hitlets['channel'],
            peaklet_max_times,
            self.config['tight_coincidence_window_left'],
            self.config['tight_coincidence_window_right'],
        )

        peaklets['tight_coincidence'] = tight_coincidence_channel[1]

        if self.config['diagnose_sorting'] and len(r):
            assert np.diff(r['time']).min(initial=1) >= 0, "Records not sorted"
            assert np.diff(hitlets['time']).min(initial=1) >= 0, "Hits/Hitlets not sorted"
            assert np.all(peaklets['time'][1:]
                          >= strax.endtime(peaklets)[:-1]), "Peaks not disjoint"

        # Update nhits of peaklets:
        counts = strax.touching_windows(hitlets, peaklets)
        counts = np.diff(counts, axis=1).flatten()
        #peaklets['n_hits'] = counts

        return peaklets        

    def natural_breaks_threshold(self, peaks):
        rise_time = -peaks['area_decile_from_midpoint'][:, 1]

        # This is ~1 for an clean S2, ~0 for a clean S1,
        # and transitions gradually in between.
        f_s2 = 8 * np.log10(rise_time.clip(1, 1e5) / 100)
        f_s2 = 1 / (1 + np.exp(-f_s2))

        log_area = np.log10(peaks['area'].clip(1, 1e7))
        thresholds = self.config['peak_split_gof_threshold']
        return (
            f_s2 * np.interp(
                log_area,
                *np.transpose(thresholds[2]))
            + (1 - f_s2) * np.interp(
                log_area,
                *np.transpose(thresholds[1])))
    @staticmethod
    @numba.njit()
    def distort(flattened_hitpatterns, coords):
        images = np.zeros((len(flattened_hitpatterns), 19, 19))
        for i, hp in enumerate(flattened_hitpatterns):
            for j, pmt_area in enumerate(hp):
                images[i][coords[j][0]][coords[j][1]] = pmt_area
        return images

    def cluster(self, kmeans_clusters, hit_patterns, top_pmt_array):
        cluster_labels = np.zeros(len(kmeans_clusters), dtype=[(('Contributing PMTs in cluster', 'cluster_channels'), np.int32, 253)])
        for index, peaklet in enumerate(zip(hit_patterns, kmeans_clusters)):
                if peaklet[1]>1:
                    kmeans = KMeans(n_clusters=peaklet[1], random_state=0).fit(top_pmt_array, sample_weight=peaklet[0])
                    cluster_labels[index][0]=np.array(kmeans.labels_)
        return cluster_labels

    def recluster_max_indices(self, maxima_indices, distance_thresh = 3):
        first_maxes = np.repeat(0,len(maxima_indices))
        second_maxes = np.repeat(0,len(maxima_indices))
        for i,m in enumerate(maxima_indices):
            fm = m[0]
            sm = m[1]
            count = 0
            for em in m:
                if (np.sqrt(np.sum((self.xy[em]-self.xy[fm])**2))>distance_thresh)&(count==0):
                    sm = em
                    count+=1
                if count>0:
                    break
            first_maxes[i] = fm
            second_maxes[i] = sm
        return first_maxes, second_maxes

    @staticmethod
    @numba.njit()
    def first_second_max_areas(image, fm, sm):
        fm_area = np.zeros(len(fm))
        sm_area = np.zeros(len(sm))
        flat_im = image.reshape(len(image),19*19)
        for i in range(len(fm)):
            fm_area[i] = flat_im[i][fm[i]]
            sm_area[i] = flat_im[i][sm[i]]
        return fm_area, sm_area

    #Stolen from Axial Convertor Class    
    def calcPairs(self):
        """
        This function calculates the relation between 
        PMT channel number and position on 2D "image"
        """
        radius = 10
        out_2D_shape = tf.constant([2*radius-1,2*radius-1],
                                        dtype=tf.int32)

        row_nPMT = np.array([6, 9, 12, 13, 14, 15, 
                             16, 17, 16, 17, 16, 17, 16, 
                             15, 14, 13, 12, 9, 6])
        row_nPMT_cumulative = np.cumsum(np.append([0],row_nPMT))
        ##
        offsets = np.zeros(2*radius -1,dtype=int)
        offsets[:]=-1
        offsets[0:radius] = (np.arange(radius,radius*2)-
                             row_nPMT[0:radius])/2
        offsets[radius:] = ((radius*2-1)-
                             np.arange(radius,2*radius-1)[::-1] + # pure HEX offset
                            (np.arange(radius,2*radius-1)[::-1] - 
                             row_nPMT[radius:])/2
                            )
        assert offsets.shape[0]==(2*radius-1)


        pairs = []
        for i in range(0, 253):
            n_row = np.argwhere( (i>=row_nPMT_cumulative[0:-1])*
                                 (i<row_nPMT_cumulative[1:])  
                               )[0][0]   
            x_coord = i - row_nPMT_cumulative[n_row]+offsets[n_row]
            y_coord = (2*radius-1) - n_row-1
            pairs.append((i,x_coord, y_coord))
        return(np.array(pairs) )
    def calc_etas(self, peaklets, hits):
        etas = np.zeros(len(peaklets['time']), dtype=np.float32)
        ha_mask = ((peaklets['width'][:, 5] > 500) & (peaklets['area'] > 4000))
        for p_i, peaklet in enumerate(peaklets):
            if ha_mask[p_i]:
                hits_selection = hits['time']>(peaklet['time']-self.pad_time)
                hits_selection &= hits['time']<(peaklet['time']+peaklet['length']*peaklet['dt']+self.pad_time)
                ds_hits = hits[hits_selection]

                hits_used = ds_hits[(ds_hits['channel']<253)]
                #Apply threshold here, since we only need the threshold-ed hits for this phase
                hits_used = hits_used[hits_used['area'] > np.sum(hits_used['area'])/50000]

                if len(hits_used['time']) <= 2:
                    print("Error found, number of hits: {}".format(len(ds_hits)))
                    etas[p_i] = float("NaN")
                    continue
                start_time = np.min(hits_used['time'])
                end_time_ind = np.argmax(hits_used['time'])
                end_time = hits_used['time'][end_time_ind] + hits_used['length'][end_time_ind]*hits_used['dt'][end_time_ind]
                length = end_time - start_time
                #dt_factor can also be set to a constant like 500 if desired
                dt_factor = 0.04*length
                pmt_pos = np.array(straxen.pmt_positions()[['x','y']])
                top_pmt_pos = pmt_pos[:253]
                hit_positions = pmt_pos[hits_used['channel'][hits_used['channel']<253]]
                time_shifted_hits = hits_used[hits_used['channel']<253]['time'] - np.min(hits_used[hits_used['channel']<253]['time'])
                time_shifted_hits = time_shifted_hits/dt_factor
                arr = np.stack((np.array(hit_positions.T[0]), np.array(hit_positions.T[1]), np.array(time_shifted_hits)), axis=-1)
                #Cluster with one cluster and take the inner-cluster distance
                transformed_1 = KMeans(n_clusters=1, random_state=0).fit_transform(arr, sample_weight=hits_used['area'])
                dists_1 = [i[0] for i in transformed_1]
                inner_distance_1_cluster = np.average(dists_1, weights = hits_used['area'])
                #Cluster with two clusters, using fit and fit_transform, to get labels and inner-cluster distance
                #I think there should be a way to extract the transformed space from the initial clustering, but I kept getting errors
                two_clusters = KMeans(n_clusters=2, random_state=0).fit(arr, sample_weight=hits_used['area'])
                two_clusters_labels = two_clusters.labels_
                transformed_2 = KMeans(n_clusters=2, random_state=0).fit_transform(arr, sample_weight=hits_used['area'])
                dists_cluster_1 = [i[0] for i in transformed_2[two_clusters_labels == 0]]
                dists_cluster_2 = [i[1] for i in transformed_2[two_clusters_labels == 1]]
                inner_distance_2_cluster = np.average(dists_cluster_1, weights = hits_used['area'][two_clusters_labels == 0]) + np.average(dists_cluster_2, weights = hits_used['area'][two_clusters_labels == 1])
                etas[p_i] = (inner_distance_1_cluster - inner_distance_2_cluster)/(inner_distance_1_cluster + inner_distance_2_cluster)
            else:
                etas[p_i] = float("NaN")
        # loop over peaks

        return etas 
