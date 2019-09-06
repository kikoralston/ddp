import numpy as np
import pandas as pd
import json
from scipy.optimize import linprog
from cvxopt import matrix, solvers
#import matplotlib.pyplot as plt
import copy


class PowerPlant:
    def __init__(self, idx, cap):
        self.idx = idx
        self.cap = cap
        self.type = None


class TermoPlant(PowerPlant):
    def __init__(self, idx, cap, cost):
        super().__init__(idx=idx, cap=cap)
        self.type = 'thermo'
        self.cost = cost


class HydroPlant(PowerPlant):
    def __init__(self, idx, cap, prod_fac=0., vmax=0., vini=0.):
        super().__init__(idx=idx, cap=cap)
        self.type = 'hydro'
        self.prod_fac = prod_fac
        self.vmax = vmax
        self.vini = vini
        if prod_fac > 0:
            self.turbmax = np.round(cap / prod_fac)
        else:
            self.turbmax = 0


class CaseConfig:
    def __init__(self):
        self.nhydro = 0
        self.ntherm = 0
        self.nper = 0

        self.list_hydro = []
        self.list_thermal = []

        self.inflows = {}
        self.hours = []
        self.load = []

    def read_config(self):

        with open('./config.txt', 'r') as f:
            d = json.load(f)

        # read general data
        self.nhydro = int(d['general data']['nhyd'])
        self.ntherm = int(d['general data']['nterm'])
        self.nper = int(d['general data']['nper'])

        # read data from thermal plants
        self.list_thermal = [None] * self.ntherm
        for i, tp in enumerate(d['term data']):
            idx = int(tp['idx_t'])
            cap = float(tp['pot'])
            cost = float(tp['cost'])
            self.list_thermal[i] = TermoPlant(idx, cap, cost)

        # read data from hydro plants
        self.list_hydro = [None] * self.nhydro
        for i, hp in enumerate(d['hydro data']):
            idx = int(hp['idx_h'])
            cap = float(hp['pot'])
            vmax = float(hp['vmax'])
            vini = float(hp['vini'])
            prodfac = float(hp['prod_factor'])

            self.list_hydro[i] = HydroPlant(idx, cap, prodfac, vmax, vini)

        # read inflow data
        for i, inflows in enumerate(d['inflows']):
            idx_h = inflows['idx_h']
            y = inflows['values']
            self.inflows[idx_h] = y

        # read number of hours in each period
        self.hours = d['hours']

        # read load (MW)
        self.load = d['load']


def set_lp(c, stage, vinihydro, previous_cuts):
    """

    :param c: object of class CaseConfig
    :param stage: (int) current stage of optimization (NOTE: first stage == 0)
    :param vinihydro: (dict) dictionary with initial volume of hydro plants {idx_hydro: volume} (**keys are integers**)
    :param previous_cuts: (list of dicts) list of dictionaries with "cuts" of future value function
    :return: dictionary with vectors and matrices (CVX format) for LP problem
    """
    load = c.load[stage]
    hours = c.hours[stage]

    n_term = c.ntherm
    n_hydro = c.nhydro

    # vector of decision variables follows the following rule:
    #   - first the n_{term} variables for thermal generation (in MWh)
    #   - then for each hydro generator we have:
    #       * water turbined (in Hm^3,  =prod_fac * generation)
    #       * final volume (in Hm^3)
    #       * water spilled (in Hm^3)
    #   - next we have the future cost value (\alpha)
    #
    # therefore, cost vector will have size n_{term} + 3*n_{hydro} + 1

    n_var = n_term + 3*n_hydro + 1
    list_plants = c.list_thermal + c.list_hydro

    # dictionary mapping indexes in list of plants to position in variable vector
    dict_pos = {}

    # first get only the thermo plants
    j = 0
    for i, p in enumerate(list_plants):
        if p.type == 'thermo':
            dict_pos[i] = j
            j = j+1

    # next get the hydro
    for i, p in enumerate(list_plants):
        if p.type == 'hydro':
            dict_pos[i] = j
            # each hydro uses 3 positions
            j = j+3

    # create cost vector -------

    # present cost
    present_cost = [p.cost for p in list_plants if p.type == 'thermo']

    # append cost zero for hydro decision variables
    present_cost = present_cost + n_hydro*3*[0.]

    # append future cost (\alpha) to create cost vector
    if stage == 0:
        cost_vector = present_cost + [1.]
    else:
        cost_vector = present_cost + [0.]

    # Equality constraints -------

    A_eq = np.array([])
    b_eq = np.array([])

    # supply = demand

    row = np.array(n_term*[1.] + n_hydro*[0., 0., 0.] + [0.])
    for i, p in enumerate(list_plants):
        if p.type == 'hydro':
            pos_plant = dict_pos[i]
            row[pos_plant] = np.round(1e6*p.prod_fac/3600., 2)

    A_eq = (np.concatenate((A_eq, row))).reshape((1, n_var))
    rhs = load*hours
    b_eq = np.concatenate((b_eq, [rhs]))

    # water balance for each hydro
    for i, p in enumerate(list_plants):
        if p.type == 'hydro':
            pos_plant = dict_pos[i]
            row = np.array(n_var * [0.])
            row[pos_plant:pos_plant+3] = [1., 1., 1.]

            if stage == 0:
                vini = p.vini
            else:
                vini = vinihydro[p.idx]

            inflow = c.inflows[p.idx][stage]

            rhs = vini + np.round(inflow*(3600.*hours)/1e6, 2)

            A_eq = np.concatenate((A_eq, [row]), axis=0)
            b_eq = np.concatenate((b_eq, [rhs]))

    # Inequality constraints -------

    A_ub = np.array([])
    b_ub = np.array([])

    # gen <= cap (thermo)
    for i, p in enumerate(list_plants):
        if p.type == 'thermo':
            pos_plant = dict_pos[i]
            row = np.array([0.]*n_var)
            row[pos_plant] = 1.
            rhs = np.round(p.cap * hours, 2)

            A_ub = np.concatenate((A_ub, row))
            b_ub = np.concatenate((b_ub, [rhs]))

    # fix shape of A_ub
    A_ub = A_ub.reshape((n_term, n_var))

    # hydro turb < turbmax
    for i, p in enumerate(list_plants):
        if p.type == 'hydro':
            pos_plant = dict_pos[i]
            row = np.array(n_var * [0.])
            row[pos_plant:pos_plant+3] = [1., 0., 0.]
            rhs = p.turbmax*3600*hours/1e6
            A_ub = np.concatenate((A_ub, [row]), axis=0)
            b_ub = np.concatenate((b_ub, [rhs]))

    # hydro volume < vmax
    for i, p in enumerate(list_plants):
        if p.type == 'hydro':
            pos_plant = dict_pos[i]
            row = np.array(n_var * [0.])
            row[pos_plant:pos_plant+3] = [0., 1., 0.]
            rhs = p.vmax
            A_ub = np.concatenate((A_ub, [row]), axis=0)
            b_ub = np.concatenate((b_ub, [rhs]))

    # alpha >= FC
    if stage == 0 and previous_cuts is not None:
        for cut in previous_cuts:
            # get cuts from previous iterations
            print('Adding cut from iter')
            if cut is not None:
                row = np.array(n_var * [0.])

                row[n_var-1] = -1.
                # row[4] = - np.round(rr2['y'][1], 2)
                row[4] = -np.round(cut['slope'], 2)

                rhs = -np.round(cut['value'], 2) - np.round(cut['slope']*cut['x'], 2)

                A_ub = np.concatenate((A_ub, [row]), axis=0)
                b_ub = np.concatenate((b_ub, [rhs]))

    # add non negativity constraints
    for i in np.arange(n_var):
        row = np.array([0.] * n_var)
        row[i] = -1.
        rhs = 0.
        A_ub = np.concatenate((A_ub, [row]), axis=0)
        b_ub = np.concatenate((b_ub, [rhs]))

    # convert everything to cvx matrix
    c = matrix(cost_vector)
    G = matrix(A_ub)
    h = matrix(b_ub)
    A = matrix(A_eq)
    b = matrix(b_eq)

    # return dict with cvx matrices for LP
    return {'c':c, 'G':G, 'h':h, 'A':A, 'b':b}


def get_cut(result_iter):
    """

    :param result_iter:
    :return:
    """
    cut = {'slope': result_iter[1]['y'][1], 'value': result_iter[1]['primal objective'], 'x': result_iter[0]['x'][4]}

    return cut


def compute_line_cut(result_i):
    """

    :param result_i:
    :return:
    """
    x1 = np.arange(0, 140)
    y1 = result_i[1]['primal objective'] - result_i[1]['y'][1] * (x1 - result_i[0]['x'][4])

    return x1, y1


def estimate_fcf(list_plants):
    """

    :param list_plants:
    :return:
    """
    fcf = []

    # previous_cuts
    previous_cuts = []

    vfinal_range = np.arange(0, 140, step=0.1)

    for vfinal in vfinal_range:

        x = np.array(7*[0.])
        x[4] = vfinal
        results = [{'x': x}]

        # vini = bookkeep[curr_iter][stage - 1]['x'][pos_plant + 1]

        dict_lp = set_lp(list_plants, 12, 1, 672, 0, results, None)

        result = solvers.lp(c=dict_lp['c'], G=dict_lp['G'], h=dict_lp['h'], A=dict_lp['A'], b=dict_lp['b'],
                            solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})

        fcf = fcf + [result['primal objective']]

    return vfinal_range, fcf