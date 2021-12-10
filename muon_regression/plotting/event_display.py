import numpy as np
import uproot

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns

__all__ = ['load_tree', 'restrict_energy', 'plot_eventdisplay', 'plot_multiple']


def readandshape(tree, varname):
    
    r'''
    Read and reshape the branch of the input tree
    
    Arguments:
        tree: ROOT TTree with the simulated calo data
        varname: string name of the variable to be loaded
    Returns:
        Numpy array with the reshaped loaded data
    '''
    
    entry = np.array(list(tree[varname].array()), dtype='float32')
    return np.reshape(entry, [-1, 50,32,32,1])

    
def load_tree(file_path):
    
    r'''
    Load the essential branches

    Arguments:
        file_path: string filepath to the input ROOT file
    Returns:
        Numpy array with the calo data,
        Numpy array with the true energy
    '''
    
    tree = uproot.open(file_path)["B4"]
    
    rechit_energy = readandshape(tree, "rechit_energy")
    rechit_x   = readandshape(tree, "rechit_x")
    rechit_y   = readandshape(tree, "rechit_y")
    rechit_z   = readandshape(tree, "rechit_z")
    rechit_vxy = readandshape(tree, "rechit_vxy")
    rechit_vz  = readandshape(tree, "rechit_vz")
    
    true_energy = np.array(list(tree["true_energy"].array()), dtype='float32')
    
    calo = np.concatenate([rechit_energy,
                           rechit_x,
                           rechit_y,
                           rechit_z,
                           rechit_vxy,
                           rechit_vz],axis=-1)
    
    return calo, true_energy
    
    
def restrict_energy(calo, true_energy, e_range):
    
    r'''
    Select energy range of muons
    
    Arguments:
        calo: Numpy array with the calo data
        true_energy: Numpy array with the true energy
        e_range: list with [up, down] range of the energy
    Returns:
        Numpy array with the calo data in selected range,
        Numpy array with the true energy in selected rannge
    '''    
    
    mask = (true_energy > e_range[0]) & (true_energy < e_range[1]) 
    return calo[mask], true_energy[mask]

    
def cuboid_data(pos, size=(1,1,3)):
    
    r'''
    Return a cuboid element
    
    Arguments:
        pos: tuple with the xyz coordinates of the cell
        size: tuple with the size of a cell
    Returns:
        Numpy array with x coordinates,
        Numpy array with y coordinates,
        Numpy array with z coordinates
    '''
    
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317 CC BY-SA 4.0 Patrick B.
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)


def plotCubeAt(pos=(0,0,0),size=(1,1,3),color=0,ax=None, alpha=1, m=None):
    
    r'''
    Plotting a cube element at position pos
    
    Arguments:
        pos: tuple with the xyz coordinates of the cell
        size: tuple with the size of a cell
        color: float with the normalized energy of the cell
        ax: axis object
        alpha: float with the normalized energy of the cell
        m: color map
    '''
    if ax is not None:
        X, Y, Z = cuboid_data(pos, size)
        with sns.axes_style('whitegrid',{'patch.edgecolor': 'None'}):
            ax.plot_surface(X, Z, Y, color=m.to_rgba(color), rstride=1, cstride=1, alpha=alpha,
                            antialiased=True, shade=False)
   
        
# e, x, y, z
def plotevent(event, arr, ax, usegrid=False, iscalo=True):
    
    r'''
    Plot a single event
    
    Arguments
        event: int Number of event to plot
        arr: Numpy array with the calo data
        ax: Axis object
        usegrid: if True plot the cells in a grid
        iscalo: if False fix the opacity and color of the cells
    '''    
    
    usearr = arr[event]
        
    scaled_emax = np.log(np.max(usearr[:,:,:,0])+1)
        
    norm = mpl.colors.Normalize(vmin=0, vmax=scaled_emax)
    cmap = cm.viridis
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    alpha = 0.5
    if not iscalo:
        alpha = 0.01
    for i in range(usearr.shape[0]):
        for j in range(usearr.shape[1]):
            for k in range(usearr.shape[2]):
                e = np.log(usearr[i,j,k,0]+1)
                x = usearr[i,j,k,1]
                y = usearr[i,j,k,2]
                z = usearr[i,j,k,3]
                dxy_indiv = usearr[i,j,k,4]
                dz = usearr[i,j,k,5]
                
                if usegrid:
                    x, y, z, dxy_indiv, dz = i, j, k, 1, 1
                
                alpha = (e / scaled_emax)
                if alpha < 0.0005:
                    continue
                plotCubeAt(pos=(x,y,dz/2.+z), size=(dxy_indiv,dxy_indiv,dz), color=alpha, ax=ax, m=m, alpha=alpha)
   

def plot_eventdisplay(event, calo, true_energy, usegrid=False, save=False, save_path='./'):
    
    r'''
    Plot a single event display
    
    Arguments:
        event: Number of event to plot
        calo: Numpy array with the calo data
        true_energy: Numpy array with the true energy
        usegrid: if True plot the cells in a grid
        save: if True save plot
        save_path: output path for the .png file
    '''
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.grid(False)

    en = true_energy[event]
    ax.text2D(0.05, 0.95, "Energy="+str(en)+" GeV", transform=ax.transAxes)

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    print('plotting...')
    plotevent(event, calo, ax, usegrid=usegrid, iscalo=True)
    plt.tight_layout()

    if save is True:
        plt.savefig(save_path + "muoncal_" + str(int(true_energy[event])) + "GeV_" + str(event) + ".png")
    

def plot_multiple(n_evt, calo, true_energy, e_range=None, save_path='plots/'):
    
    r'''
    Plot multiple event displays
    
    Arguments:
        n_evt: Number of events to plot
        calo: Numpy array with the calo data
        true_energy: Numpy array with the true energy
        e_range: list with [up, down] range of the energy
        save_path: output path for the .png file
    '''
    
    if e_range is not None:
        calo, true_energy = restrict_energy(calo, true_energy, e_range=e_range)
    
    if len(true_energy) < n_evt:
        raise Exception('The maximum number of events in the energy range ' + str(e_range) + ' is ' + 
                        str(len(true_energy)) + ', cannot do ' + str(n_evt) + ' plots!')
    
    for event in range(n_evt):
        plot_eventdisplay(event, calo, true_energy, save=True, save_path=save_path)
        
        
