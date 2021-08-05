# import uproot
import ROOT
from optparse import OptionParser
from pathlib import Path
import os


def file_is_valid(filename:str, tree:str='B4') -> bool:
    try:
        f = ROOT.TFile.Open(filename)
        t = f.Get(tree)
        if t.GetEntries() < 1: raise ValueError("")
    except Exception as e: print(e); return False
    return                 True


if __name__ == '__main__':
    parser = OptionParser(usage=__doc__)
    parser.add_option("-d", "--directory",dest="directory",action="store", type='string',  help="Input file")
    parser.add_option("-t", "--tree",     dest="tree",     action="store", type='string',  help="Tree name", default='B4')
    opts, args = parser.parse_args()

    path = Path(opts.directory)
    os.system(f'rm {path}/use_files.txt')
    with open(path/'use_files.txt', 'w') as fout:
        for f in path.glob('*.root'):
            print(f'Checking {f}')
            if file_is_valid(str(f), opts.tree): fout.write(f'{f}\n')
