import os
from typing import List

import h5py

CHUNK_SIZE = 100

def merge_hdf5_files(input_files: List[str], output_file: str, keys: List[str]):
    """
    Concatenates specified datasets from multiple HDF5 input files into a single output file.

    It initializes datasets in the output file as resizable and appends data
    from each source file.
    """
    if not input_files:
        print("Error: Input file list is empty.")
        return

    keys = keys if keys is not None else ['images', 'masks']

    print(f"\nStarting merge process to create {output_file}...")

    # --- 1. Determine structure from the first file ---
    try:
        with h5py.File(input_files[0], 'r') as f_init:
            # We need the shape (excluding the batch dimension), dtype, and number of dimensions
            initial_dset_info = {}
            for k in keys:
                if k not in f_init:
                    raise KeyError(f"Key '{k}' not found in initial file: {input_files[0]}")

                dset = f_init[k]
                initial_dset_info[k] = {
                    'shape_suffix': dset.shape[1:],
                    'dtype': dset.dtype,
                    'ndim': dset.ndim
                }
    except Exception as e:
        print(f"Error determining file structure: {e}")
        return

    # --- 2. Create the output file and resizable datasets ---
    # We use h5py.File context manager for guaranteed closing
    with h5py.File(output_file, 'w') as f_out:
        output_datasets = {}
        for k, info in initial_dset_info.items():
            print(f"  > Initializing dataset '{k}' (Shape: {info['shape_suffix']}, DType: {info['dtype']})")

            # shape=(0,) means starting with zero items
            # maxshape=(None,) is CRUCIAL: it makes the first dimension unlimited (resizable)
            output_datasets[k] = f_out.create_dataset(
                k,
                shape=(0,) + info['shape_suffix'],
                maxshape=(None,) + info['shape_suffix'],
                dtype=info['dtype'],
                chunks=True,  # Chunking is required for resizable datasets
                compression='gzip'
            )

        # --- 3. Iterate and Append Data ---
        for file_path in input_files:
            print(f"\n  > Appending data from file: {file_path}")

            try:
                with h5py.File(file_path, 'r') as f_in:

                    # We iterate over the source data in chunks to handle potentially large files
                    dset_len = len(f_in[keys[0]])

                    for start in range(0, dset_len, CHUNK_SIZE):
                        end = min(start + CHUNK_SIZE, dset_len)

                        for k in keys:
                            in_dset = f_in[k]

                            # Read the data chunk from the input file
                            data_chunk = in_dset[start:end]
                            data_len = len(data_chunk)

                            # Get the current length of the output dataset
                            current_len = output_datasets[k].shape[0]

                            # Resize the output dataset to accommodate the new chunk
                            new_len = current_len + data_len
                            output_datasets[k].resize(new_len, axis=0)

                            # Write the new data chunk into the extended slice
                            output_datasets[k][current_len:new_len] = data_chunk

                        print(f"    - Wrote samples {start} to {end} for all keys.")

            except Exception as e:
                print(f"!! Error processing {file_path}: {e}. Skipping this file.")
                continue

    # --- 4. Verification ---
    with h5py.File(output_file, 'r') as f_out:
        total_samples = len(f_out[keys[0]])
        print(f"\nSuccessfully merged data into {output_file}.")
        print(f"Final total samples: {total_samples}")


if __name__ == '__main__':

    # 1. Define files and cleanup old runs
    file_list = ['../segmenter/data/dresden_preprocessed.h5',
                     '../segmenter/data/all_data.h5']
    output_filename = '../segmenter/merged_dataset.hdf5'

    for f in file_list + [output_filename]:
        if os.path.exists(f):
            os.remove(f)

    merge_hdf5_files(file_list, output_filename, keys=['images', 'masks'])
