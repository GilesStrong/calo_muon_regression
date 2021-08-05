from typing import Optional, Any
from pathlib import Path
import os
import json

__all__ = ['ExpJournal']


class ExpJournal():
    r'''
    Generic storage class for various experiment results. Also stores hard-coded machine hardware configurations, along with assigned random-seeds for running
    multiple re-runs of the same experimental setup.
    Results are stored in a `results` dictionary, and can be stored and accessed either from the `results` property, or through the overloaded `[]` operator.

    Arguments:
        exp_name: name for the particular experiment (note that the machine name will be appended to the save filename).
        machine: name of the machine on which the experiment is being performed. Can correspond to a hard-coded hardware configuration.
        path: path to directory in which to create savefiles.

    Examples::
       >>> exp = ExpJournal('experiment_0', 'helios_cuda', Path('./results'))
       >>> exp['train time'] = time
       >>> exp['validation performance'] = val_loss
       >>> exp.save()
       >>> exp = ExpJournal.from_json('./results/experiment_0_helios_cuda.json')
       >>> print(exp)
    '''

    def __init__(self, exp_name:str, machine:str, path:Optional[Path]=None):
        self.exp_name,self.machine,self.path = exp_name,machine,path
        self.device = self.lookup_machine(self.machine)
        self.results = {}
        self.seed = self.lookup_seed(self.machine)
    
    def __repr__(self) -> str:
        rep = f'Experiment:\t{self.exp_name}\nMachine:\t{self.machine}\nDevice:\t{self.device}'
        for r in self.results: rep += f'\n{r}\t{self[r]}'
        return rep

    def __getitem__(self, idx:str) -> Any: return self.results[idx]

    def __setitem__(self, idx:str, val:Any) -> None: self.results[idx] = val   
        
    def save(self, path:Optional[Path]=None) -> None:
        r'''
        Saves the experiment results to `{path}/{exp_name}_{machine}.json`.

        Arguments:
            path: path to directory in which to create savefiles, unless a path was specificed during initialisation.
        '''

        path = path if path is not None else self.path
        assert path is not None, 'Path is not set'
        os.makedirs(path, exist_ok=True)
        with open(path/f'{self.exp_name}_{self.machine}.json', 'w') as fout:
            json.dump({'exp_name':self.exp_name, 'machine':self.machine, 'results':self.results},  fout)
    
    @classmethod
    def from_json(cls, fname:str):
        r'''
        Instanciates an `ExpJournal` object from the stored results of a previous `ExpJournal`.

        Arguments:
            fname: filename pointing to stored results

        Examples::
           >>> exp = ExpJournal.from_json('./results/experiment_0_helios_cuda.json')
           >>> print(exp)
        '''

        with open(fname) as fin: data = json.load(fin)
        if not isinstance(fname, str): fname = str(fname)
        e = cls(data['exp_name'], data['machine'], Path(fname[:fname.rfind('/')]))
        e.results = data['results']
        return e
            
    @staticmethod
    def lookup_machine(machine:str) -> str:
        r'''
        Returns the hardware configuration corresponding to the machine name

        Arguments:
            machine: name of machine used to look up the hardware configuration

        Returns:
            The hardware configuration description
        '''

        if machine == 'helios_cuda': return 'Nvidia GeForce GTX 1080 Ti GPU'
        if machine == 'helios_cpu':  return 'Intel Core i7-8700K CPU @ 3.7 GHz (6x2)'
        if machine == 'mbp':         return 'Intel Core i7-8559U CPU @ 2.7 GHz (4x2)'
        if machine == 'daedalus':    return 'Intel Xenon Skylake CPU @ 2.2 GHz (4x1)'
        if machine == 'icarus':      return 'Intel Xenon Skylake CPU @ 2.2 GHz (4x1)'
        if machine == 'morpheus':    return 'Intel Xenon Skylake CPU @ 2.2 GHz (2x1)'
        if machine == 'lxplus':      return 'Intel Xenon Skylake CPU @ 2.3 GHz (10x1)'
        if machine == 'caltech':     return 'Nvidia GeForce GTX 1080 GPU'
        if machine == 'padova_v100': return 'Nvidia Tesla V100 GPU'
        if machine == 'padova_xp':   return 'Nvidia TitanXP'
        if machine == 'padova_t4':   return 'Nvidia T4'
        if machine == 'jan_cluster': return 'Nvidia Tesla V100 GPU?'
        print(f'Machine {machine} not yet hardcoded; please add. Returning "Unknown" for the hardware description')
        return 'Unknown'
    
    @staticmethod
    def lookup_seed(machine:str) -> int:
        r'''
        Returns the random seed assigned to the machine name

        Arguments:
            machine: name of machine used to look up the random seed

        Returns:
            The random seed to use for experiments run on the particular machine
        '''

        if machine == 'helios_cuda': return 1111
        if machine == 'helios_cpu':  return 2222
        if machine == 'mbp':         return 3333
        if machine == 'daedalus':    return 4444
        if machine == 'icarus':      return 5555
        if machine == 'morpheus':    return 6666
        if machine == 'lxplus':      return 7777
        if machine == 'caltech':     return 8888
        if machine == 'padova_v100': return 9999
        if machine == 'padova_xp':   return 1113
        if machine == 'jan_cluster': return 1112
        print(f'Machine {machine} not yet hardcoded; please add. Returning default seed of 0')
        return 0
