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
import plotly.graph_objects as go
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


def generate_table(dataframe):

    df = dataframe.applymap(lambda x: np.round(x, 2) if type(x) != str else x)
    nrow, ncol = df.shape

    html_table = html.Table(
        # Header
        [html.Tr([html.Th('')] + [html.Th(i) for i in range(1, ncol)])] +
        # Body
        [html.Tr([html.Td(dataframe.iloc[row, col]) for col in range(ncol)]) for row in range(nrow)], id="customers")

    return html_table


def get_data_table(table, n_clicks):
    # assumes that 'table' is a dictionary with the HTML table structure (it is not a html.table object)

    if n_clicks > 0:
        table_data = table['props']['children']

        col_names = [th['props']['children'] for th in table_data[0]['props']['children']]

        table_data = table_data[1:]

        data_values = [[td['props']['children'] for i, td in enumerate(tr['props']['children'])] for tr in table_data]

        df_table = pd.DataFrame(data=data_values, columns=col_names)
    else:
        df_table = generate_empty_table()
        #df_table = get_data_table(df_table, 1)

    return df_table


def get_cuts_df(df, n_iter):

    cuts = []

    for i in range(1, n_iter):
        cuts = cuts + [{'slope': df.iloc[15, i], 'value': df.iloc[16, i], 'x': df.iloc[4, i]}]

    return cuts


def compute_cut_plot(cut):

    # uses signal convention from function 'get_cuts_df'
    x1 = np.arange(0, 140)
    y1 = cut['value'] - cut['slope'] * (x1 - cut['x'])

    return x1, y1


def run_step_ddp(c, stage, vinihydro, previous_cuts):

    idx_stage = stage - 1

    dict_lp = set_lp(c, idx_stage, vinihydro, previous_cuts)

    rr = solvers.lp(c=dict_lp['c'], G=dict_lp['G'], h=dict_lp['h'], A=dict_lp['A'], b=dict_lp['b'],
                    solver='glpk', options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})

    results_step = copy.deepcopy(rr)

    return results_step


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [html.H2('A simple 2 stage hydro-scheduling example'),
     html.Div([dcc.Link('https://github.com/kikoralston/ddp/', href='https://github.com/kikoralston/ddp/')], style={'font-family': 'sans-serif'}),
     html.Br(),
     html.Div([html.Div(html.Button(id='submit-button', n_clicks=0, children='Increment Iteration', type='submit'),
                        style={'display': 'inline-block', 'width': '49%'}),
               html.Div(dcc.Graph(id='example-graph-3', style={'height': '200px'}, config={'displayModeBar': False}),
                        style={'display': 'inline-block', 'width': '49%'})]),
     html.Br(),
     html.Div([html.Div(id='table-out',
                        style={'display': 'inline-block', 'width': '49%', 'vertical-align': 'top'}),
               html.Div([dcc.Graph(id='example-graph'), dcc.Graph(id='example-graph-2')],
                        style={'display': 'inline-block', 'width': '49%'})]),
     # Hidden div inside the app that stores the intermediate value
     html.Div(id='hidden-value') #, style={'display': 'none'}
     ])


@app.callback(Output('table-out', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('table-out', 'children'),
               State('hidden-value', 'children')])
def update_results(n_clicks, children, jsonvalue):

    c = CaseConfig()
    c.read_config()

    if n_clicks == 0:
        stage = 1
        iter = 1
    else:
        # get CURRENT stage and iteration
        state_dict = json.loads(jsonvalue)

        print(type(state_dict))
        print(state_dict)

        stage = state_dict['stage']
        iter = state_dict['iter']

    html_tbl = None

    df_table = get_data_table(children, n_clicks)

    print("******df_table*******")
    print(df_table)

    if iter <= 1:
        # no cuts generated yet!
        previous_cuts = []
        html_tbl = generate_empty_table()
    else:
        previous_cuts = get_cuts_df(df_table, iter)

    if stage > 1:
        vinihydro = df_table.iloc[4, iter]
    else:
        vinihydro = None

    if n_clicks > 0 and iter <= 4:
        res = run_step_ddp(c, stage, vinihydro, previous_cuts)

        df_table = update_col_results_df(df_table, res, iter, stage)

        html_tbl = generate_table(df_table)

    elif iter >= 5:

        html_tbl = generate_table(df_table)

    print(html_tbl)

    return html_tbl


@app.callback(Output('hidden-value', 'children'),
              [Input('example-graph-2', 'figure')],
              [State('submit-button', 'n_clicks'),
               State('hidden-value', 'children')])
def update_hidden_div(figure, n_clicks, children):

    c = CaseConfig()
    c.read_config()

    print("N CLICKS!!!" + str(n_clicks))

    if n_clicks == 0:
        state_dict = dict(iter=1, stage=1)
    else:
        state_dict = json.loads(children)

        # increment stage and iteration
        if state_dict['stage'] + 1 > c.nper:
            state_dict['stage'] = 1
            state_dict['iter'] = state_dict['iter'] + 1
        else:
            state_dict['stage'] = state_dict['stage'] + 1

    return json.dumps(state_dict)


@app.callback(Output('example-graph', 'figure'),
              [Input('table-out', 'children')],
              [State('example-graph', 'figure'),
               State('submit-button', 'n_clicks'),
               State('hidden-value', 'children')])
def update_plot(children, figure, n_clicks, jsonvalue):

    list_out = []
    dict_out = figure

    if n_clicks == 0:
        dict_out = {'data': list_out, 'layout': {'title': 'Future Cost Function'}}
    else:
        # get CURRENT stage and iteration
        state_dict = json.loads(jsonvalue)

        stage = state_dict['stage']
        iter = state_dict['iter']

        if stage == 2:

            df_table = get_data_table(children, n_clicks)

            # compute cuts

            cuts = get_cuts_df(df_table, iter+1)

            cuts_plot = [None] * iter
            for i in np.arange(iter):
                cuts_plot[i] = compute_cut_plot(cuts[i])

            list_out = [{'x': c[0], 'y': c[1], 'type': 'lines+markers', 'name': 'cut {}'.format(i+1)}
                        for i, c in enumerate(cuts_plot)]

            dict_out = {'data': list_out, 'layout': {'title': 'Future Cost Function'}}

    return dict_out


@app.callback(Output('example-graph-2', 'figure'),
              [Input('example-graph', 'figure')],
              [State('example-graph-2', 'figure'),
               State('table-out', 'children'),
               State('submit-button', 'n_clicks'),
               State('hidden-value', 'children')])
def update_plot_2(figure1, figure2, tab, n_clicks, jsonvalue):

    list_out = []
    dict_out = figure2

    if n_clicks == 0:
        dict_out = {'data': list_out, 'layout': {'title': 'Gap'}}
    else:
        # get CURRENT stage and iteration
        state_dict = json.loads(jsonvalue)

        stage = state_dict['stage']
        iter = state_dict['iter']

        if stage == 2:

            df_table = get_data_table(tab, iter)

            # lower bound
            lb = df_table.iloc[17, 1:iter+1].values

            # upper bound
            ub = df_table.iloc[18, 1:iter+1].values

            # iter
            iterations = np.arange(iter+1)

            list_out = [{'x': iterations, 'y': lb, 'type': 'lines+markers', 'name': 'Lower Bound'},
                        {'x': iterations, 'y': ub, 'type': 'lines+markers', 'name': 'Upper Bound'}]

            dict_out = {'data': list_out, 'layout': {'title': 'Gap'}}

    return dict_out


@app.callback(Output('example-graph-3', 'figure'),
              [Input('hidden-value', 'children')],
              [State('table-out', 'children'),
               State('submit-button', 'n_clicks')])
def update_decision_tree(jsonvalue, tab, n_clicks):

    c = CaseConfig()
    c.read_config()

    if n_clicks == 0:
        nodes = pd.DataFrame({'idx': [1], 'x': [1], 'y': [1], 'iter': [1], 'stage': [1], 'vini': [0], 'active': [True]})
        edges = None
        stagevalue = 1
        itervalue = 1

    else:
        df_table = get_data_table(tab, n_clicks)
        state_dict = json.loads(jsonvalue)

        stagevalue = state_dict['stage'] - 1
        itervalue = state_dict['iter']

        # populate nodes
        idx=[]
        x=[]
        y=[]
        iter=[]
        stage=[]
        vini=[]
        ii = 1
        for i in range(1, itervalue + 1):
            for s in range(c.nper):
                if s == 0 and i == 1:
                    idx = idx + [ii]
                    x = x + [1]
                    y = y + [1]
                    iter = iter + [i]
                    stage = stage + [s + 1]
                    vini = vini + [0]

                    ii = ii + 1
                elif s > 0 and i < itervalue:
                    idx = idx + [ii]
                    x = x + [s + 1]
                    y = y + [i]
                    iter = iter + [i]
                    stage = stage + [s + 1]
                    vini = vini + [float(df_table.iloc[4, i])]
                    ii = ii + 1

                elif 0 < s <= stagevalue and i == itervalue:
                    idx = idx + [ii]
                    x = x + [s + 1]
                    y = y + [i]
                    iter = iter + [i]
                    stage = stage + [s + 1]
                    vini = vini + [float(df_table.iloc[4, i])]

                    ii = ii + 1

        # define active node
        if stagevalue == 0:
            # root node active
            active = [True] + [False]*(len(stage)-1)
        else:
            # other node active
            active = [True if stage[i] == stagevalue+1 and iter[i] == itervalue else False for i in range(len(stage))]

        nodes = pd.DataFrame({'idx': idx, 'x': x, 'y': y, 'iter': iter, 'stage': stage, 'vini': vini,
                               'active': active})

        # populate edges
        fromnode=[]
        tonode=[]

        ii = 1
        for i in range(1, itervalue + 1):
            for s in range(c.nper):
                if s == 0 and i < itervalue:
                    fromnode = fromnode + [1]
                    tonode = tonode + [ii + 1]
                    ii = ii + 1
                elif s == 0 and i == itervalue and s <= stagevalue - 1:
                    fromnode = fromnode + [1]
                    tonode = tonode + [ii + 1]
                    ii = ii + 1
                elif i < itervalue and s < c.nper-1:
                    fromnode = fromnode + [ii]
                    tonode = tonode + [ii + 1]
                    ii = ii + 1
                elif i == itervalue and s < stagevalue - 1:
                    fromnode = fromnode + [ii]
                    tonode = tonode + [ii + 1]
                    ii = ii + 1

        edges = pd.DataFrame({'from': fromnode, 'to': tonode})

    print(nodes)

    print(edges)

    # set layout
    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)',
                       margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
                       xaxis={'tickmode': 'array', 'tickvals': [1, 2, 3], 'ticktext': ['1', '2', '3']})

    # initiate plot
    fig = go.Figure(layout=layout)

    # add edges
    if edges is not None:
        for r in edges.iterrows():
            # from
            p1 = nodes.loc[nodes['idx'] == r[1]['from']]
            # to
            p2 = nodes.loc[nodes['idx'] == r[1]['to']]

            fig.add_trace(go.Scatter(x=[p1['x'].values[0], p2['x'].values[0]], y=[p1['y'].values[0], p2['y'].values[0]],
                                     mode='lines', showlegend=False, opacity=1, line=dict(color='black')))

    # then add inactive nodes
    df = pd.DataFrame(nodes[nodes['active'] == False])
    if df.shape[0] > 0:
        df['text'] = df.apply(lambda row: 't={0:d}<br>iter={1:d}<br>v={2:.2f}'.format(row['stage'], row['iter'], row['vini']),
                              axis=1)
        fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', text=df['text'], hoverinfo='text',
                                 marker=dict(size=20, opacity=1, color='white',
                                             line=dict(color='black', width=1)),
                                 showlegend=False))

    # then add active node (single node)
    df = pd.DataFrame(nodes[nodes['active'] == True])
    df['text'] = 't={0:d}<br>iter={1:d}<br>v={2:.2f}'.format(df['stage'].iloc[0], itervalue, df['vini'].iloc[0])

    fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='markers', text=df['text'], hoverinfo='text',
                             marker=dict(size=20, opacity=1, color='red'), showlegend=False))

    # remove yaxis
    fig.update_yaxes(showticklabels=False)

    # show plot!
    #fig.write_html('first_figure.html', auto_open=True)

    return fig


#with (open('./save.p', 'rb')) as f:
#    dataframe = pck.load(f)

if __name__ == '__main__':
    app.run_server(debug=True)