from ronswanson.band_simulation import BandSimulation as Simulation
from joblib import Parallel, delayed
import json
import h5py
import numpy as np
from tqdm.auto import tqdm
from ronswanson import ParameterGrid
from ronswanson.utils.logging import setup_logger
from ronswanson.utils import Colors
from ronswanson.simulation import gather
log = setup_logger(__name__)

with open('completed_parameters.json', 'r') as f:
    complete_params = json.load(f)
pg = ParameterGrid.from_yaml('/Users/tparsotan/Library/CloudStorage/Box-Box/Theory-Interface/parameters.yml')
n_points = pg.n_points
def func(i, silent: bool=True):
    params = pg.at_index(i)
    if not silent:
        log.info(f'{params}')
    for p in complete_params:
        if np.alltrue(np.array(p) == params):
            log.debug('parameters already exists in file!')
            return
    simulation = Simulation(i, params, pg.energy_grid,'/Users/tparsotan/Library/CloudStorage/Box-Box/Theory-Interface/database.h5')
    simulation.run()
iteration = [i for i in range(0, n_points)]
Parallel(n_jobs=1)(delayed(func)(i) for i in tqdm(iteration, colour=Colors.RED.value, desc='simulating function'))
gather('/Users/tparsotan/Library/CloudStorage/Box-Box/Theory-Interface/database.h5', 18, clean=True)
