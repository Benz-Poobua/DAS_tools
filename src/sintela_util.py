import os
import sys
import datetime
import glob
import h5py
import re
import logging
import numpy as np
import scipy.interpolate
import scipy.signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def sintela_to_datetime(sintela_times):
    """
    Convert Sintela timestamps (in microseconds since Unix epoch) to datetime.datetime objects (UTC).

    :param sintela_times: NumPy integer or NumPy array of microseconds since epoch
    :type sintela_times: np.integer or np.ndarray
    :return: datetime object or NumPy array of datetime objects
    :rtype: datetime.datetime or np.ndarray
    """
    if isinstance(sintela_times, np.integer):
        return datetime.datetime.fromtimestamp(sintela_times / 1e6, tz=datetime.timezone.utc)
    elif isinstance(sintela_times, np.ndarray):
        return np.vectorize(lambda t: datetime.datetime.fromtimestamp(t / 1e6, tz=datetime.timezone.utc))(sintela_times)
    else:
        raise ValueError("Input must be either a single NumPy integer or a NumPy array")

def get_Onyx_file_time(file):
    """
    Extract datetime object from Onyx-style filename.

    :param file: full filename string
    :type file: str
    :return: datetime object or None if pattern not matched
    :rtype: datetime.datetime or None
    """
    pattern = r'.*(\d{4})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2}).*'
    filename = os.path.basename(file)
    match = re.match(pattern, filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime.datetime(year, month, day, hour, minute, second)
    else:
        logging.warning("Invalid datetime format in filename: %s", filename)
        return None
    
def get_Onyx_h5(dir_path, t_start, t_end=None, length=60):
    """
    Get list of Onyx HDF5 files within a specified time range.

    :param dir_path: Directory path
    :type dir_path: str
    :param t_start: Start time
    :type t_start: datetime.datetime
    :param t_end: End time (optional)
    :type t_end: datetime.datetime or None
    :param length: Duration in seconds if t_end is not given
    :type length: int
    :return: List of file paths
    :rtype: list
    """
    if not t_end:
        t_end = t_start + datetime.timedelta(seconds=length)

    all_files = glob.glob(os.path.join(dir_path, '*.h5'))
    out_files = []

    for i, file in enumerate(sorted(all_files)):
        file_timestamp = get_Onyx_file_time(file)
        if file_timestamp and t_start <= file_timestamp <= t_end:
            if not out_files and i > 0:
                out_files.append(all_files[i - 1])
            out_files.append(file)

    logging.info("Selected %d files from %s to %s", len(out_files), t_start, t_end)
    return sorted(out_files)



def read_Onyx_h5_to_list(files, cha_start=None, cha_end=None, t_start=None, t_end=None, verbose=False):
    """
    Read list of Onyx HDF5 files and return data, timestamps, and metadata.

    :param files: List of file paths
    :type files: list
    :param cha_start: Start channel index (inclusive)
    :type cha_start: int or None
    :param cha_end: End channel index (exclusive)
    :type cha_end: int or None
    :param t_start: Start datetime
    :type t_start: datetime.datetime or None
    :param t_end: End datetime
    :type t_end: datetime.datetime or None
    :param verbose: Verbose mode
    :type verbose: bool
    :return: Tuple of time list, data list, and attribute list
    :rtype: (list, list, list)
    """
    data_read, time_read, attrs_read = [], [], []

    # Normalize t_start and t_end to timezone-aware UTC if needed
    if t_start and t_start.tzinfo is None:
        t_start = t_start.replace(tzinfo=datetime.timezone.utc)
    if t_end and t_end.tzinfo is None:
        t_end = t_end.replace(tzinfo=datetime.timezone.utc)

    for i, file in enumerate(files):
        if verbose:
            logging.info("Reading file %d of %d: %s", i + 1, len(files), file)

        try:
            with h5py.File(file, 'r') as f:
                time_rec = np.array(f['Acquisition/Raw[0]/RawDataTime'], dtype='int64')
                data = f['Acquisition/Raw[0]/RawData']

                if time_rec.shape[0] != data.shape[0]:
                    logging.warning("Skipping file due to shape mismatch: %s", file)
                    continue

                # Convert to datetime (UTC-aware)
                t_datetime = sintela_to_datetime(time_rec)

                # Filter time range
                t_start_idx = 0 if not t_start else np.min(np.where([(t - t_start).total_seconds() > 0 for t in t_datetime]))
                t_end_idx = None if not t_end else np.max(np.where([(t - t_end).total_seconds() < 0 for t in t_datetime])) + 1

                # Subset data
                data_rec = np.array(data[t_start_idx:t_end_idx, cha_start:cha_end], dtype='float32')
                time_rec = time_rec[t_start_idx:t_end_idx]

                # Attributes
                attrs_rec = dict(f['Acquisition'].attrs)
                if cha_start is not None:
                    attrs_rec['StartLocusIndex'] = cha_start
                    attrs_rec['NumberOfLoci'] = data_rec.shape[1]

                # Save
                if len(time_rec) > 0:
                    time_read.append(time_rec)
                    data_read.append(data_rec)
                    attrs_read.append(attrs_rec)

        except Exception as e:
            logging.error("Error reading %s: %s", file, e)

    return time_read, data_read, attrs_read

def pad_array_with_nans(arr, target_shape):
    """
    Pad 2D array with NaNs to a target shape.

    :param arr: Original array
    :type arr: np.ndarray
    :param target_shape: Desired shape (rows, cols)
    :type target_shape: tuple
    :return: Padded array
    :rtype: np.ndarray
    """
    if arr.shape[0] > target_shape[0] or arr.shape[1] > target_shape[1]:
        raise ValueError("Target shape must be larger than input array.")

    padded = np.full(target_shape, np.nan)
    padded[:arr.shape[0], :arr.shape[1]] = arr
    return padded

def comb_Onyx_data(time_read, data_read, attrs_read):
    """
    Combine data with identical metadata attributes.

    :param time_read: List of timestamp arrays
    :param data_read: List of data arrays
    :param attrs_read: List of attribute dicts
    :return: Tuple of combined timestamps, data, and one set of attributes
    :rtype: (np.ndarray, np.ndarray, dict) or None
    """
    keys = [(a['PulseRate'], np.round(a['SpatialSamplingInterval'], 2), a['StartLocusIndex']) for a in attrs_read]

    if all(k == keys[0] for k in keys):
        maxcha = max(arr.shape[1] for arr in data_read)
        times = np.concatenate(time_read)
        data = np.concatenate([pad_array_with_nans(arr, (arr.shape[0], maxcha)) for arr in data_read])
        return times, data, attrs_read[0]

    logging.warning("Metadata mismatch: cannot combine files.")
    return None

def split_continuous_data(t_rec, data_rec, attrs):
    """
    Split data into continuous chunks based on gaps in timestamp.

    :param t_rec: Timestamp array (microseconds)
    :type t_rec: np.ndarray
    :param data_rec: Data array
    :type data_rec: np.ndarray
    :param attrs: Metadata dictionary with 'PulseRate'
    :type attrs: dict
    :return: Lists of timestamps and data chunks
    :rtype: (list, list)
    """
    dt = np.diff(t_rec) / 1e6
    gap_idx = np.where(np.abs(dt) > 1 / attrs['PulseRate'])[0]

    if len(gap_idx) == 0:
        return [t_rec], [data_rec]

    time_list, data_list = [], []
    start = 0
    for idx in gap_idx:
        time_list.append(t_rec[start:idx+1])
        data_list.append(data_rec[start:idx+1])
        start = idx + 1

    time_list.append(t_rec[start:])
    data_list.append(data_rec[start:])
    return time_list, data_list

def fill_data_gaps(time_list, data_list, attrs, fill_value=np.nan, t_format=None):
    """
    Merge continuous segments, filling gaps with fill_value.

    :param time_list: List of timestamp arrays
    :param data_list: List of corresponding data arrays
    :param attrs: Metadata dictionary with 'PulseRate'
    :param fill_value: Value to use for filling gaps
    :param t_format: 'datetime' to convert to datetime format
    :return: Tuple of uniform time array and data array
    :rtype: (np.ndarray, np.ndarray)
    """
    dt_eq = 1 / attrs['PulseRate'] * 1e6
    t_eq = np.arange(time_list[0][0], time_list[-1][-1] + dt_eq, dt_eq)

    filled = np.full((len(t_eq), data_list[0].shape[1]), fill_value)
    i = 0
    for t_arr, d_arr in zip(time_list, data_list):
        while t_arr[0] > t_eq[i]:
            i += 1
        minidx = np.min(np.where(t_arr >= t_eq[i]))
        lenidx = len(np.where(t_arr >= t_eq[i])[0])
        filled[i:i+lenidx] = d_arr[minidx:]
        i += lenidx

    if t_format == 'datetime':
        t_eq = sintela_to_datetime(t_eq)

    return t_eq, filled

def apply_sosfiltfilt_with_nan(sos, data, axis=-1, padtype='odd', padlen=None, verbose=True):
    """
    Apply zero-phase filtering with sosfiltfilt, handling errors gracefully.

    :param sos: Second-order sections filter coefficients
    :type sos: np.ndarray
    :param data: Input data array
    :type data: np.ndarray
    :param axis: Axis to filter along
    :type axis: int
    :param padtype: Padding type
    :type padtype: str
    :param padlen: Padding length
    :type padlen: int or None
    :param verbose: Whether to print errors
    :type verbose: bool
    :return: Filtered data array or NaN-filled array if error
    :rtype: np.ndarray
    """
    try:
        return scipy.signal.sosfiltfilt(sos, data, axis=axis, padtype=padtype, padlen=padlen)
    except Exception as e:
        if verbose:
            logging.error("sosfiltfilt error: %s", e)
        return np.full(data.shape, np.nan)

def ensure_utc(dt):
    """
    Ensure a datetime object is timezone-aware in UTC.

    If the input datetime is naive (no timezone info), this function
    assigns UTC timezone to it. If it is already timezone-aware,
    it returns the datetime unchanged.

    :param dt: Input datetime object, naive or timezone-aware
    :type dt: datetime.datetime
    :return: Timezone-aware datetime in UTC
    :rtype: datetime.datetime
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=datetime.timezone.utc)

if __name__ == '__main__':
    logging.info("This is a utility module.")