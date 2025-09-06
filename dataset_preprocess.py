import pandas as pd
import numpy as np
import mne

class EDFDatasetReader:
    def __init__(self, edf_files):
        self.edf_files = edf_files

    def scan_dataset(self):
        dataset_info = pd.DataFrame()

        for file_path in self.edf_files:
            try:
                # Step 1: Read the raw data without preloading
                raw = mne.io.read_raw_edf(file_path, preload=False)

                # Step 2: Extract metadata
                filename = file_path.split("/")[-1]
                duration = raw.times[-1] - raw.times[0]
                sampling_rate = raw.info['sfreq']
                channels = raw.info['ch_names']

                # Step 3: Compute PSD using the helper function
                psd_data = extract_power_spectral_density(raw)

                # Step 4: Compute average power across all frequencies and channels
                average_power = np.mean(psd_data['psd'])

                # Step 5: Append new row to DataFrame
                dataset_info = dataset_info.append({
                    'filename': filename,
                    'file_path': file_path,
                    'duration': duration,
                    'sampling_rate': sampling_rate,
                    'channels': channels,
                    'average_power': average_power
                }, ignore_index=True)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return dataset_info

    def export_to_csv(self, output_file):
        dataset = self.scan_dataset()
        dataset.to_csv(output_file, index=False)
