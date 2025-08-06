import logging
import re
import segyio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_trace_headers(segyfile):
    """
    Extract selected SEG-Y trace header fields into a pandas DataFrame for 2D seismic lines.

    :param segyfile: An opened SEG-Y file using segyio in read mode.
    :type segyfile: segyio.SegyFile

    :return: DataFrame where each row corresponds to a trace and each column corresponds to a selected SEG-Y trace header field.
    :rtype: pd.DataFrame
    """
    logging.info("Starting trace header extraction.")
    all_headers = segyio.tracefield.keys

    useful_keys = [
        "TraceNumber", "FieldRecord", "CDP", "CDP_X", "CDP_Y",
        "SourceX", "SourceY", "GroupX", "GroupY", "offset",
        "TRACE_SAMPLE_COUNT", "TRACE_SAMPLE_INTERVAL",
        "DelayRecordingTime", "MuteTimeStart", "MuteTimeEND",
        "SourceGroupScalar", "CoordinateUnits", "GainType",
        "TotalStaticApplied", "YearDataRecorded", "DayOfYear"
    ]

    n_traces = segyfile.tracecount
    logging.info(f"Number of traces detected: {n_traces}")

    df = pd.DataFrame(index=range(n_traces))

    for key in useful_keys:
        if key in all_headers:
            try:
                df[key] = segyfile.attributes(all_headers[key])[:]
                logging.debug(f"Successfully read header '{key}'")
            except Exception as e:
                logging.warning(f"Could not read header '{key}': {e}")
                df[key] = None
        else:
            logging.warning(f"Header '{key}' not found in SEG-Y tracefield map.")
            df[key] = None

    logging.info("Trace header extraction completed.")
    return df

def parse_text_header(segyfile):
    """
    Parse the textual (EBCDIC) header of the SEG-Y file into a clean dictionary.

    :param segyfile: An opened SEG-Y file using segyio.
    :type segyfile: segyio.SegyFile

    :return: Dictionary with keys C01 through C40 and corresponding text lines from the SEG-Y header.
    :rtype: dict
    """
    logging.info("Parsing textual (EBCDIC) header.")
    
    try:
        raw_text = segyio.tools.wrap(segyfile.text[0])
    except Exception as e:
        logging.error(f"Failed to read text header: {e}")
        return {}

    header_lines = re.split(r'C ', raw_text)[1:]
    header_lines = [line.replace('\n', ' ').strip() for line in header_lines]
    formatted_header = {f"C{str(i+1).zfill(2)}": line for i, line in enumerate(header_lines)}

    logging.info("Text header parsed successfully.")
    return formatted_header

def plot_segy(segyfile, dpi=1200, clip_percentile=99, cmap='RdBu'):
    """
    Plot a 2D seismic SEG-Y line using matplotlib.

    Loads a SEG-Y file, extracts trace data and two-way travel time samples (TWT in ms),
    then displays the seismic section as an image. Amplitudes are clipped at a given
    percentile to enhance visual contrast.

    :param segyfile: Path to the SEG-Y file (.sgy or .segy) to be visualized.
    :type segyfile: str

    :param dpi: Dots per inch for the figure. Higher values yield higher resolution.
    :type dpi: int, optional

    :param clip_percentile: Percentile value used to clip amplitude extremes (default is 99).
                            This enhances contrast for visualization.
    :type clip_percentile: float, optional

    :param cmap: Colormap used for seismic amplitude display. Default is 'RdBu'.
    :type cmap: str, optional

    :return: None. Displays a matplotlib plot of the seismic section.
    :rtype: None
    """
    with segyio.open(segyfile, ignore_geometry=True) as f:
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000  # In milliseconds
        n_samples = f.samples.size
        twt = f.samples  # Two-way travel time [ms]
        data = f.trace.raw[:]  # Numpy array of shape (n_traces, n_samples)

    # Clip amplitude for better visualization
    vm = np.percentile(data, clip_percentile)

    # Set up plot
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(18, 8), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    # Define plot extent: [xmin, xmax, ymin, ymax]
    extent = [1, n_traces, twt[-1], twt[0]] # Reverse time axis (seismic convention)

    ax.imshow(data.T, cmap=cmap, vmin=-vm, vmax=vm, aspect='auto', extent=extent) 
    ax.set_xlabel('CDP number (Trace number)')
    ax.set_ylabel('Two-Way Travel Time (ms)')
    ax.set_title(f'Seismic Line: {segyfile}')
    plt.tight_layout()
    plt.show()
    # Optional: plt.savefig('seismic_vector.pdf', dpi=1200, bbox_inches='tight')

if __name__ == '__main__':
    logging.info("This is a utility module.")