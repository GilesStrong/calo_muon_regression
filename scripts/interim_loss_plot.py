import sys
from typer import Typer, Argument, Option
from pathlib import Path
import json
from lumin.plotting.training import plot_train_history
sys.path.append('../')
from muon_regression.data_proc import get_dataset, Detector, calc_hl_feats  # noqa E402
from muon_regression.basics import plot_settings  # noqa E402

app = Typer()


@app.command()
def plot_histories(in_dir:str=Argument(..., help="Path to directory with loss histories"),
                   last_n:int=Option(-1, help="If set, will only plot the last n updates")) -> None:
    in_dir = Path(in_dir)
    plot_settings.savepath = in_dir
    for hist in in_dir.glob("*.txt"):
        with open(hist, 'r') as fin: loss = json.loads(fin.read())
        xlow = max(len(loss['val_loss'])-last_n,0) if last_n > 0 else 0
        plot_train_history([loss], ignore_trn=False, xlow=xlow, savename=hist.stem, settings=plot_settings, show=False)


if __name__ == "__main__": app()
