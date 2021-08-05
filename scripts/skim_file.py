import optparse
import h5py
from pathlib import Path
import numpy as np
import json
import sys
sys.path.append('../')
from muon_regression.data_proc import get_dataset, Detector  # noqa E402


if __name__ == "__main__":
    parser = optparse.OptionParser(usage=__doc__)
    parser.add_option("-i", "--input",    dest="input",    action="store", type='string',  help="Input file")
    parser.add_option("-o", "--output",   dest="output",   action="store", type='string',  help="Output file")
    parser.add_option("-t", "--tree",     dest="tree",     action="store", type='string',  help="Tree name",
                      default='B4')
    parser.add_option("-f", "--features", dest="features", action="append", type='string', help="Input features",
                      default=['rechit_energy'])
    opts, args = parser.parse_args()
    if opts.input is None:  raise ValueError("Input file (-i) must be set")
    if opts.output is None: raise ValueError("Output file (-o) must be set")

    print(f'Received options: {opts}')
    df = get_dataset(opts.input, tree_name=opts.tree, as_flat=False, inputs=opts.features)
    detector = Detector(Path(opts.input).parts[-2])
    if detector.shape is None: detector = Detector(Path(opts.input).parts[-1])
    assert detector.shape is not None, f"Foldername {Path(opts.input).parts[-2]} not recognised by Detector"
    x = np.concatenate([np.vstack(df[f].values).reshape((-1,len(opts.features),*detector.shape)) for f in opts.features], axis=1)
    y = df['true_energy'].values
    print("Loaded")
    out_file = h5py.File(opts.output, "w")
    out_file.create_dataset('X', shape=x.shape, dtype='float32', data=x.astype('float32'), compression="lzf")
    print("X saved")
    out_file.create_dataset('y', shape=y.shape, dtype='float32', data=y.astype('float32'), compression="lzf")
    print("y saved")
    out_file.create_dataset('meta', data=json.dumps({'mean':x.mean(),'sum':x.sum(), 'square_sum':np.square(x).sum(),'len':x.shape[0]}))
    print("Meta saved")
    out_file.close()
    print(f'Saved {x.shape[0]} muons')
