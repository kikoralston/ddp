import numpy as np
import pandas as pd
from scipy.optimize import linprog
from cvxopt import matrix, solvers
#import matplotlib.pyplot as plt
import copy


class PowerPlant:
    def __init__(self, cap):
        self.cap = cap
        self.type = None


class TermoPlant(PowerPlant):
    def __init__(self, cap, cost):
        super().__init__(cap=cap)
        self.type = 'thermo'
        self.cost = cost


class HydroPlant(PowerPlant):
    def __init__(self, cap, prod_fac=0., vmax=0., vini=0.):
        super().__init__(cap=cap)
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

    def read_config(self):

        with open('./config.txt', 'r') as f:
            d = f.readlines()

        # read general data
        self.nhydro = int(d[3].split(',')[1].strip())
        self.ntherm = int(d[4].split(',')[1].strip())
        self.nper = int(d[5].split(',')[1].strip())

        # read data from thermal plants
        self.list_thermal = [None] * self.ntherm
        for i in range(self.ntherm):
            data_therm = d[10+i].split(',')
            cap = float(data_therm[1].strip())
            cost = float(data_therm[2].strip())
            self.list_thermal[i] = TermoPlant(cap, cost)

        # read data from hydro plants
        self.list_hydro = [None] * self.nhydro
        for i in range(self.nhydro):
            data_hyd = d[17+i].split(',')
            cap = float(data_hyd[1].strip())
            vmax = float(data_hyd[2].strip())
            vini = float(data_hyd[3].strip())
            prodfac = float(data_hyd[4].strip())

            self.list_hydro[i] = HydroPlant(cap, prodfac, vmax, vini)

        # read inflow data
        for i in range(self.nhydro):
            data_inflow = d[22+i].split(',')
            #idx_hyd = data_inflow[0].strip()
            y = list(map(lambda x: float(x.strip()), data_inflow[1].split(';')))
            self.inflows[i] = y

        # read number of hours in each period
        self.hours = list(map(lambda x: int(x.strip()), d[27].split(';')))


def update_col_results_df(df_results, res, iter, stage):

    if stage == 1:

        results_stage1 = [np.round(x, 2) for x in list(res['x'])] + [np.round(list(res['y'])[1], 2)] + [np.round(res['primal objective'], 2)]

        df_results.iloc[0:9, iter] = results_stage1

    else:
        results_stage2 = [np.round(x, 2) for x in list(res['x'])[:6]] + [np.round(list(res['y'])[1], 2)] + [np.round(res['primal objective'], 2)]

        cost_stage1 = df_results.iloc[8, iter]
        cost_stage2 = res['primal objective']
        alpha = df_results.iloc[6, iter]

        lb = np.round(cost_stage1, 2)
        ub = np.round(cost_stage1 - alpha + cost_stage2, 2)
        gap = np.round(np.abs(ub-lb)/ub, 2)

        results_stage2 = results_stage2 + [lb, ub, '{0:.0f}%'.format(100*gap)]

        df_results.iloc[9:, iter] = results_stage2

    return df_results


def save_results_df(res):

    rownames = ['G1', 'G2', 'G3', 'turb', 'Vfinal', 'Spill', 'Alpha', 'water value',  'cost',
                'G1', 'G2', 'G3', 'turb', 'Vfinal', 'Spill', 'water value', 'cost',
                'Lower Bound', 'Upper Bound', 'Gap']

    g1_stage1 = [rr[0]['x'][0] for rr in res]
    g2_stage1 = [rr[0]['x'][1] for rr in res]
    g3_stage1 = [rr[0]['x'][2] for rr in res]
    turb_stage1 = [rr[0]['x'][3] for rr in res]
    vfinal_stage1 = [rr[0]['x'][4] for rr in res]
    spill_stage1 = [rr[0]['x'][5] for rr in res]
    alpha = [rr[0]['x'][6] for rr in res]
    water_value_stage1 = [rr[0]['y'][1] for rr in res]
    cost_stage1 = [rr[0]['primal objective'] for rr in res]

    g1_stage2 = [rr[1]['x'][0] for rr in res]
    g2_stage2 = [rr[1]['x'][1] for rr in res]
    g3_stage2 = [rr[1]['x'][2] for rr in res]
    turb_stage2 = [rr[1]['x'][3] for rr in res]
    vfinal_stage2 = [rr[1]['x'][4] for rr in res]
    spill_stage2 = [rr[1]['x'][5] for rr in res]
    water_value_stage2 = [rr[1]['y'][1] for rr in res]
    cost_stage2 = [rr[1]['primal objective'] for rr in res]

    lb = cost_stage1
    ub = [cost_stage1[i] - alpha[i] + cost_stage2[i] for i in range(len(lb))]
    gap = [np.abs(ub[i]-lb[i])/ub[i] for i in range(len(lb))]

    list_data = [g1_stage1, g2_stage1, g3_stage1, turb_stage1, vfinal_stage1, spill_stage1, alpha,
                 water_value_stage1, cost_stage1, g1_stage2, g2_stage2, g3_stage2, turb_stage2, vfinal_stage2,
                 spill_stage2, water_value_stage2, cost_stage2, lb, ub, gap]

    list_data = [[rownames[i]] + row for i, row in enumerate(list_data)]

    df_results = pd.DataFrame(data=list_data)

    return df_results


def print_summary(res):

    n_iter = len(res)

    print('                       |{0:>10}|{1:>10}|{2:>10}|{3:>10}|'.format('iter1', 'iter2', 'iter3', 'iter4'))
    g1_stage1 = [rr[0]['x'][0] for rr in res]
    fmt_print = ['G1                     |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*g1_stage1))

    g2_stage1 = [rr[0]['x'][1] for rr in res]
    fmt_print = ['G2                     |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*g2_stage1))

    g3_stage1 = [rr[0]['x'][2] for rr in res]
    fmt_print = ['G3                     |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*g3_stage1))

    turb_stage1 = [rr[0]['x'][3] for rr in res]
    fmt_print = ['turb                   |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*turb_stage1))

    vfinal_stage1 = [rr[0]['x'][4] for rr in res]
    fmt_print = ['Vfinal                 |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*vfinal_stage1))

    spill_stage1 = [rr[0]['x'][5] for rr in res]
    fmt_print = ['spill                  |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*spill_stage1))

    alpha = [rr[0]['x'][6] for rr in res]
    fmt_print = ['alpha                  |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*alpha))

    water_value_stage1 = [rr[0]['y'][1] for rr in res]
    fmt_print = ['water value            |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*water_value_stage1))

    cost_stage1 = [rr[0]['primal objective'] for rr in res]
    fmt_print = ['cost (includes alpha)  |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*cost_stage1))

    print('------------------------------------------------------------------------------------------')

    g1_stage2 = [rr[1]['x'][0] for rr in res]
    fmt_print = ['G1                     |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*g1_stage2))

    g2_stage2 = [rr[1]['x'][1] for rr in res]
    fmt_print = ['G2                     |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*g2_stage2))

    g3_stage2 = [rr[1]['x'][2] for rr in res]
    fmt_print = ['G3                     |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*g3_stage2))

    turb_stage2 = [rr[1]['x'][3] for rr in res]
    fmt_print = ['turb                   |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*turb_stage2))

    vfinal_stage2 = [rr[1]['x'][4] for rr in res]
    fmt_print = ['Vfinal                 |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*vfinal_stage2))

    spill_stage2 = [rr[1]['x'][5] for rr in res]
    fmt_print = ['spill                  |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*spill_stage2))

    water_value_stage2 = [rr[1]['y'][1] for rr in res]
    fmt_print = ['water value            |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*water_value_stage2))

    cost_stage2 = [rr[1]['primal objective'] for rr in res]
    fmt_print = ['cost                   |'] + ['{:>10.2f}|']*n_iter + ['   -      |']*(4-n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*cost_stage2))

    print('------------------------------------------------------------------------------------------')

    lb = cost_stage1
    fmt_print = ['Lower Bound            |'] + ['{:>10.2f}|'] * n_iter + ['   -      |'] * (4 - n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*lb))

    ub = [cost_stage1[i] - alpha[i] + cost_stage2[i] for i in range(len(lb))]
    fmt_print = ['Upper Bound            |'] + ['{:>10.2f}|'] * n_iter + ['   -      |'] * (4 - n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*ub))

    gap = [np.abs(ub[i]-lb[i])/ub[i] for i in range(len(lb))]
    fmt_print = ['gap                    |'] + ['{:>10.2%}|'] * n_iter + ['   -      |'] * (4 - n_iter)
    fmt_print = ''.join(fmt_print)
    print(fmt_print.format(*gap))


def set_lp(c, stage, vinihydro, previous_cuts):

    load = 12
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
                vini = vinihydro

            inflow = c.inflows[0][stage]

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
    cut = {'slope': result_iter[1]['y'][1], 'value': result_iter[1]['primal objective'], 'x': result_iter[0]['x'][4]}

    return cut


def compute_line_cut(result_i):

    # first get all cuts
    x1 = np.arange(0, 140)
    y1 = result_i[1]['primal objective'] - result_i[1]['y'][1] * (x1 - result_i[0]['x'][4])

    return x1, y1


def estimate_fcf(list_plants):

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


# def run_optim():
#
#     # hours in each stage (in this case month)
#     duration = [744., 672.]
#
#     # m3/s
#     inflow = [40., 0.]
#
#     # MW
#     load = 12.
#
#     list_plants = []
#
#     list_plants.append(TermoPlant(cap=5., cost=8.))
#     list_plants.append(TermoPlant(cap=5., cost=12.))
#     list_plants.append(TermoPlant(cap=20., cost=15.))
#     list_plants.append(HydroPlant(cap=11., prod_fac=0.2, vmax=130.))
#
#     # create bookkeeping dictionary for storing results
#     results = []
#
#     # previous_cuts
#     previous_cuts = []
#
#     plt.ion()
#
#     # initiate plot window
#     f, (ax1, ax2) = plt.subplots(2, 1)
#
#     # add real sampled fcf to plot
#     xfcf, yfcf = estimate_fcf(list_plants)
#
#     ax1.plot(xfcf, yfcf, ls='--', c='blue', alpha=0.1)
#
#     plt.pause(0.2)
#     plt.show()
#
#     for iter in np.arange(0, 4):
#
#         result_iter = [None, None]
#         results = results + [result_iter]
#
#         for stage in np.arange(0, 2):
#
#             dict_lp = set_lp(list_plants, load, stage, duration[stage], inflow[stage], result_iter, previous_cuts)
#
#             result = solvers.lp(c=dict_lp['c'], G=dict_lp['G'], h=dict_lp['h'], A=dict_lp['A'], b=dict_lp['b'],
#                                 solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})
#
#             result_iter[stage] = copy.deepcopy(result)
#
#         previous_cuts = previous_cuts + [get_cut(result_iter)]
#
#         plot_cuts(results, f, ax1, ax2, realfcf=(xfcf, yfcf))
#
#         plt.show()
#         plt.pause(0.02)
#
#         print_summary(results)
#
#         input("Press Enter to continue...")
#
#     plt.ioff()
#     plt.show()
#
#     return results
#
#
# def plot_cuts(results, f, ax1, ax2, realfcf=None):
#
#     # f, (ax1, ax2) = plt.subplots(2, 1)
#
#     # clear top plot
#     ax1.cla()
#
#     curr_iter = len(results) - 1
#
#     if realfcf is not None:
#         ax1.plot(realfcf[0], realfcf[1], ls='--', c='blue', alpha=0.1)
#
#     # first get all cuts
#     y1 = []
#     for i in range(curr_iter+1):
#         result_iter = results[i]
#         xaux, yaux = compute_line_cut(result_iter)
#         y1 = y1 + [yaux]
#
#         ax1.plot(result_iter[0]['x'][4], result_iter[1]['primal objective'], 'ro')
#         ax1.plot(xaux, yaux, ls='-', c='grey', alpha=0.2)
#
#     fcf = y1[0]
#     for i in np.arange(1, len(y1)):
#         fcf = np.maximum(fcf, y1[i])
#
#     ax1.plot(xaux, fcf, ls='-', c='red')
#
#     # clear bottom plot
#     ax2.cla()
#
#     x_iter = np.arange(curr_iter+1) + 1
#
#     lb = [rr[0]['primal objective'] for rr in results]
#     alpha = [rr[0]['x'][6] for rr in results]
#     cost_stage2 = [rr[1]['primal objective'] for rr in results]
#     ub = [lb[i] - alpha[i] + cost_stage2[i] for i in range(len(lb))]
#
#     ax2.plot(x_iter, lb, ls='--', c='black', marker='o')
#     ax2.plot(x_iter, ub, ls='--', c='black', marker='o')
#
#     ax2.set_xlim([0, 5])
#     ax2.set_xticks(np.arange(1, 5))
#
#     plt.show()