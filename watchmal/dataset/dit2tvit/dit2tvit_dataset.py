# torch imports
from torch import from_numpy

# generic imports
import numpy as np
import h5py # NEW: We need h5py for our custom initialize

np.set_printoptions(threshold=np.inf)
# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
import watchmal.dataset.data_utils as du

from watchmal.utils.math import direction_from_angles

class DualImageDataset(H5Dataset):
    # CHANGED: The __init__ signature is simplified
    def __init__(
        self,
        h5file,  # Now points to the merged file
        pmt_positions_file,
        mpmt_positions_file,
        num_valid_mpmt_modules=816,
        use_times=True,
        use_charges=True,
        use_padding=False,
        padding_to_fixed_dimension=[192, 192],
        transforms=None,
        one_indexed=False,
        use_memmap=True,
        channel_scale_factor=None,
        channel_scale_offset=None,
        use_isHit=False,
        use_positions=False,
        use_orientations=False,
        geometry_file=None,
        use_invalid_value=False,
        use_log_charge=False,
    ):
        # REMOVED: No more h5file_mpmt parameter
        
        # CHANGED: We call super().__init__ with the single merged file path
        # The base class will still handle the lazy-loading concept.
        super().__init__(h5file, use_memmap)

        # REMOVED: The second dataset object is no longer needed
        # self.mpmt_dataset = H5Dataset(h5file_mpmt, use_memmap)
        
        # The rest of the __init__ method is mostly the same as it deals with
        # processing logic, not file loading.
        self.pmt_positions = np.load(pmt_positions_file)["pmt_image_positions"].astype(
            int
        )
        self.num_valid_mpmt_modules = num_valid_mpmt_modules
        self.mpmt_positions = np.load(mpmt_positions_file)[
            "pmt_image_positions"
        ].astype(int)
        self.use_times = use_times
        self.use_charges = use_charges
        self.use_isHit = use_isHit
        self.use_positions = use_positions
        self.use_orientations = use_orientations
        self.use_invalid_value = use_invalid_value
        self.use_log_charge = use_log_charge
        self.data_size = np.max(self.pmt_positions, axis=0) + 1
        self.data_size_mpmt = np.max(self.mpmt_positions, axis=0) + 1
        if use_padding:
            self.data_size = padding_to_fixed_dimension
            self.data_size_mpmt = padding_to_fixed_dimension

        self.barrel_rows = [
            row
            for row in range(self.data_size[0])
            if np.count_nonzero(self.pmt_positions[:, 0] == row) == self.data_size[1]
        ]
        self.transforms = None
        self.one_indexed = one_indexed
        
        # ... (The rest of your original __init__ logic is fine) ...
        if use_positions:
            self.real_3Dpositions = np.load(geometry_file)["positions"]
        else:
            self.real_3Dpositions = None
        if use_orientations:
            self.real_3Dorientations = np.load(geometry_file)["orientations"]
        else:
            self.real_3Dorientations = None

        if channel_scale_offset is None:
            channel_scale_offset = {}
        self.scale_offset = channel_scale_offset
        if channel_scale_factor is None:
            channel_scale_factor = {}
        self.scale_factor = channel_scale_factor

        self.channel_map = {}
        current_channel = 0
        if use_times:
            self.channel_map["time"] = current_channel
            current_channel += 1
        if use_charges:
            self.channel_map["charge"] = current_channel
            current_channel += 1
        # ... (and so on for the rest of your channel map logic) ...
        if use_isHit:
            self.channel_map["isHit"] = current_channel
            current_channel+=1
        if use_positions:
            self.channel_map['position_X'] = current_channel
            self.channel_map['position_Y'] = current_channel+1
            self.channel_map['position_Z'] = current_channel+2
            current_channel+=3
        if use_orientations:
            self.channel_map['orientation_X'] = current_channel
            self.channel_map['orientation_Y'] = current_channel+1
            self.channel_map['orientation_Z'] = current_channel+2
            current_channel+=3
        if 'time' not in self.channel_map and 'charge' not in self.channel_map:
            raise ValueError('No time or charge information loaded.')
        
        self.n_channels = current_channel
        self.n_channels_mpmt = 38
        self.data_size = np.insert(self.data_size, 0, self.n_channels)
        self.data_size_mpmt = np.insert(self.data_size_mpmt, 0, self.n_channels_mpmt)


    # NEW: A custom initialize method to handle the group structure
    def initialize(self):
        """
        Initialises the arrays from the merged HDF5 file.
        This method is called once per worker process.
        """
        self.h5_file = h5py.File(self.h5_path, "r")

        # CORRECTED: Load shared, event-level data from the 'main' group
        # Since this data is identical in both original files, we only need to load it once.
        # We'll use the 'main' group as the authoritative source.
        main_group_for_shared_data = self.h5_file['main']
        self.labels = np.array(main_group_for_shared_data["labels"])
        self.positions  = np.array(main_group_for_shared_data["positions"])
        self.angles     = np.array(main_group_for_shared_data["angles"])
        self.energies   = np.array(main_group_for_shared_data["energies"])
        
        # Load hit-level data from the 'main' group
        main_group = self.h5_file['main']
        self.event_hits_index_main = np.append(main_group["event_hits_index"], main_group["hit_pmt"].shape[0]).astype(np.int64)
        self.hit_pmt_main = self.load_hits("main/hit_pmt")
        self.hit_charge_main = self.load_hits("main/hit_charge")
        self.hit_time_main = self.load_hits("main/hit_time")

        # Load hit-level data from the 'mpmt' group
        mpmt_group = self.h5_file['mpmt']
        self.event_hits_index_mpmt = np.append(mpmt_group["event_hits_index"], mpmt_group["hit_pmt"].shape[0]).astype(np.int64)
        self.hit_pmt_mpmt = self.load_hits("mpmt/hit_pmt")
        self.hit_charge_mpmt = self.load_hits("mpmt/hit_charge")
        self.hit_time_mpmt = self.load_hits("mpmt/hit_time")

        self.initialized = True
    
    # process_data_main and process_data_second methods remain UNCHANGED
    def process_data_main(self, hit_pmts, hit_times, hit_charges):
        # ... (no changes needed here) ...
        if self.one_indexed:
            hit_pmts = hit_pmts - 1
        hit_rows = self.pmt_positions[hit_pmts, 0]
        hit_cols = self.pmt_positions[hit_pmts, 1]
        invalid_value = 0.0
        if self.use_invalid_value:
            invalid_value = -100.0
        time_offset = self.scale_offset.get('time', 0.)
        time_scale = self.scale_factor.get('time', 1.0)
        charge_offset = self.scale_offset.get('charge', 0.)
        charge_scale = self.scale_factor.get('charge', 1.0)
        positions_offset = self.scale_offset.get('positions', 0.)
        positions_scale = self.scale_factor.get('positions', 1.0)
        orientations_offset = self.scale_offset.get('orientations', 0.)
        orientations_scale = self.scale_factor.get('orientations', 1.0)

        if self.use_log_charge:
            hit_charges = np.log10(hit_charges)
        data = np.full(self.data_size, invalid_value, dtype=np.float32)
        if self.use_positions:
            data[self.channel_map['position_X'], self.pmt_positions[:,0], self.pmt_positions[:,1]] = (self.real_3Dpositions[:,0]-positions_offset)/positions_scale
            data[self.channel_map['position_Y'], self.pmt_positions[:,0], self.pmt_positions[:,1]] = (self.real_3Dpositions[:,1]-positions_offset)/positions_scale
            data[self.channel_map['position_Z'], self.pmt_positions[:,0], self.pmt_positions[:,1]] = (self.real_3Dpositions[:,2]-positions_offset)/positions_scale
        if self.use_orientations:
            data[self.channel_map['orientation_X'], self.pmt_positions[:,0], self.pmt_positions[:,1]] = (self.real_3Dorientations[:,0]-orientations_offset)/orientations_scale
            data[self.channel_map['orientation_Y'], self.pmt_positions[:,0], self.pmt_positions[:,1]] = (self.real_3Dorientations[:,1]-orientations_offset)/orientations_scale
            data[self.channel_map['orientation_Z'], self.pmt_positions[:,0], self.pmt_positions[:,1]] = (self.real_3Dorientations[:,2]-orientations_offset)/orientations_scale

        if "time" in self.channel_map:
            data[self.channel_map["time"], hit_rows, hit_cols] = (hit_times-time_offset)/time_scale
        if "charge" in self.channel_map:
            data[self.channel_map["charge"], hit_rows, hit_cols] = (hit_charges-charge_offset)/charge_scale
        
        if "isHit" in self.channel_map:
            data[self.channel_map["isHit"], hit_rows, hit_cols] = 1.0
        
        return data

    def process_data_second(self, hit_pmts, hit_times, hit_charges):
        sparse_data = np.zeros((38, self.num_valid_mpmt_modules), dtype=np.float32)
        if hit_pmts.size > 0:
            hit_pmts_int = hit_pmts.astype(int)
            if self.one_indexed:
                hit_pmts_int = hit_pmts_int - 1
            location_indices = hit_pmts_int // 19   
            pmt_in_module_indices = hit_pmts_int % 19 
            
            if self.use_log_charge:
                hit_charges = np.log10(hit_charges + 1e-6)
            time_channels = pmt_in_module_indices * 2
            charge_channels = pmt_in_module_indices * 2 + 1

            sparse_data[time_channels, location_indices] = hit_times
            sparse_data[charge_channels, location_indices] = hit_charges
        
        return sparse_data


    def __getitem__(self, item):

        if not self.initialized:
            self.initialize()
        start_main = self.event_hits_index_main[item]
        stop_main = self.event_hits_index_main[item + 1]

        hit_pmts_main = self.hit_pmt_main[start_main:stop_main]
        hit_times_main = self.hit_time_main[start_main:stop_main]
        hit_charges_main = self.hit_charge_main[start_main:stop_main]

        data_main = from_numpy(
            self.process_data_main(hit_pmts_main, hit_times_main, hit_charges_main)
        )
        if self.transforms:
            data_main = du.apply_random_transformations(self.transforms, data_main)

        start_mpmt = self.event_hits_index_mpmt[item]
        stop_mpmt = self.event_hits_index_mpmt[item + 1]

        hit_pmts_mpmt = self.hit_pmt_mpmt[start_mpmt:stop_mpmt]
        hit_times_mpmt = self.hit_time_mpmt[start_mpmt:stop_mpmt]
        hit_charges_mpmt = self.hit_charge_mpmt[start_mpmt:stop_mpmt]

        sparse_mpmt_data_np = self.process_data_second(hit_pmts_mpmt, hit_times_mpmt, hit_charges_mpmt)
        data_second = from_numpy(sparse_mpmt_data_np)

        data_dict = {
            "labels": self.labels[item].astype(np.int64),
            "energies": self.energies[item].copy(),
            "angles": self.angles[item].copy(),
            "positions": self.positions[item].copy(),
            "directions": direction_from_angles(self.angles[item]),
            "indices": item,
            "data_main": data_main,
            "data_second": data_second
        }

        return data_dict