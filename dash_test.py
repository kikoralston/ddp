# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import pickle as pck
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from ddp import *
import json
import logging
logging.basicConfig(level=logging.DEBUG)


def generate_empty_table():

    rownames = ['G1', 'G2', 'G3', 'turb', 'Vfinal', 'Spill', 'Alpha', 'water value',  'cost',
                'G1', 'G2', 'G3', 'turb', 'Vfinal', 'Spill', 'water value', 'cost',
                'Lower Bound', 'Upper Bound', 'Gap']

    return html.Table(
        # Header
        [html.Tr([html.Th('')] + [html.Th(str(col)) for col in np.arange(4)])] +
        # Body
        [html.Tr([html.Td(r)] + [html.Td('-') for col in np.arange(4)]) for r in rownames],
        id="customers")


def generate_table(dataframe, n_iter):

    n_col_max = 4
    n_iter = np.minimum(n_col_max, n_iter)

    empty_iter = [i for i in np.arange(n_iter, n_col_max)]

    n_empty = len(empty_iter)

    return html.Table(
        # Header
        [html.Tr([html.Th('')] + [html.Th(col) for col in np.arange(n_iter)] + [html.Th(i) for i in empty_iter])] +
        # Body
        [html.Tr([html.Td(dataframe.iloc[i, 0])] +
                 [html.Td(np.round(dataframe.iloc[i, col], 2)) for col in np.arange(1, n_iter+1)] +
                 [html.Td('-')]*n_empty) for i in range(len(dataframe))], id="customers")


def get_data_table(table, n_clicks):
    # assumes that 'table' is a dictionary with the HTML table structure (it is not a html.table object)

    if n_clicks > 0:
        table_data = table['props']['children']

        col_names = [th['props']['children'] for th in table_data[0]['props']['children']]
        #col_names = col_names[1:]

        table_data = table_data[1:]

        data_values = [[td['props']['children'] for i, td in enumerate(tr['props']['children'])] for tr in table_data]

        df_table = pd.DataFrame(data=data_values, columns=col_names)
    else:
        df_table = generate_empty_table()

    return df_table


def get_cuts_df(df, n_iter):

    cuts = []

    for i in range(1, n_iter+1):
        cuts = cuts + [{'slope': df.iloc[15, i], 'value': df.iloc[16, i], 'x': df.iloc[4, i]}]

    return cuts


def compute_cut_plot(cut):

    # uses signal convention from function 'get_cuts_df'
    x1 = np.arange(0, 140)
    y1 = cut['value'] - cut['slope'] * (x1 - cut['x'])

    return x1, y1


def run_iteration_ddp(previous_cuts):

    list_plants = []

    list_plants.append(TermoPlant(cap=5., cost=8.))
    list_plants.append(TermoPlant(cap=5., cost=12.))
    list_plants.append(TermoPlant(cap=20., cost=15.))
    list_plants.append(HydroPlant(cap=11., prod_fac=0.2, vmax=130.))

    # hours in each stage (in this case month)
    duration = [744., 672.]

    # m3/s
    inflow = [40., 0.]

    # MW
    load = 12.

    results_iter = [None, None]

    for stage in np.arange(0, 2):
        dict_lp = set_lp(list_plants, load, stage, duration[stage], inflow[stage], results_iter, previous_cuts)

        rr = solvers.lp(c=dict_lp['c'], G=dict_lp['G'], h=dict_lp['h'], A=dict_lp['A'], b=dict_lp['b'],
                            solver='glpk', options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})

        results_iter[stage] = copy.deepcopy(rr)

    return results_iter


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [html.Div(id='table-out', style={'display': 'inline-block', 'width': '49%', 'vertical-align': 'top'}),
     html.Div(
         [html.Button(id='submit-button', n_clicks=0, children='Increment Iteration', type='submit'),
          dcc.Graph(id='example-graph'), dcc.Graph(id='example-graph-2')],
         style={'display': 'inline-block', 'width': '49%'})])


@app.callback(Output('table-out', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('table-out', 'children')])
def update_results(n_clicks, children):

    html_tbl = None
    df_table = get_data_table(children, n_clicks)

    print(df_table)

    if n_clicks < 2:
        previous_cuts = []
        html_tbl = generate_empty_table()
    elif 0 < n_clicks < 5:
        previous_cuts = get_cuts_df(df_table, n_clicks-1)

    if 0 < n_clicks < 5:
        rr = run_iteration_ddp(previous_cuts)

        df_iter = save_results_df([rr])
        icol = n_clicks

        print(df_table.iloc[:, icol])

        print('df iter')
        print(df_iter)
        print(df_iter.iloc[:, 1])

        df_table.iloc[:, icol] = df_iter.iloc[:, 1]

        print()
        print('n_iter = {}'.format(n_clicks))
        html_tbl = generate_table(df_table, n_clicks)

        print(df_table)

    elif n_clicks >= 5:
        html_tbl = generate_table(df_table, n_clicks)

    print(df_table)

    return html_tbl


@app.callback(Output('example-graph', 'figure'),
              [Input('table-out', 'children')],
              [State('example-graph', 'figure'),
               State('submit-button', 'n_clicks')])
def update_plot(children, figure, n_clicks):

    list_out = []
    dict_out = figure

    if n_clicks == 0:
        dict_out = {'data': list_out, 'layout': {'title': 'Future Cost Function'}}

    if 0 < n_clicks < 5:

        df_table = get_data_table(children, n_clicks)

        # compute cuts
        n_iter = n_clicks

        cuts = get_cuts_df(df_table, n_iter)

        cuts_plot = [None] * n_iter
        for i in np.arange(n_iter):
            cuts_plot[i] = compute_cut_plot(cuts[i])

        list_out = [{'x': c[0], 'y': c[1], 'type': 'lines+markers', 'name': 'cut {}'.format(i+1)}
                    for i, c in enumerate(cuts_plot)]

        dict_out = {'data': list_out, 'layout': {'title': 'Future Cost Function'}}

    return dict_out


@app.callback(Output('example-graph-2', 'figure'),
              [Input('table-out', 'children')],
              [State('example-graph-2', 'figure'),
               State('submit-button', 'n_clicks')])
def update_plot_2(children, figure, n_clicks):

    list_out = []
    dict_out = figure

    if n_clicks == 0:
        dict_out = {'data': list_out, 'layout': {'title': 'Gap'}}

    if 0 < n_clicks < 5:

        df_table = get_data_table(children, n_clicks)

        # lower bound
        lb = df_table.iloc[17, 1:n_clicks].values

        # upper bound
        ub = df_table.iloc[18, 1:n_clicks].values

        # iter
        iterations = np.arange(n_clicks+1)

        list_out = [{'x': iterations, 'y': lb, 'type': 'lines+markers', 'name': 'Lower Bound'},
                    {'x': iterations, 'y': ub, 'type': 'lines+markers', 'name': 'Upper Bound'}]

        dict_out = {'data': list_out, 'layout': {'title': 'Gap'}}

    return dict_out


#with (open('./save.p', 'rb')) as f:
#    dataframe = pck.load(f)

if __name__ == '__main__':
    app.run_server(debug=True)
