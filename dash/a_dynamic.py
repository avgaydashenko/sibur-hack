# coding: utf-8

import dash
import pickle
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import plotly.graph_objs as go
import pandas as pd
import os
import datetime
from plotly.graph_objs import *
from flask import send_from_directory, redirect
import dash
import plotly
import numpy as np
from dash.dependencies import Input, Output, State

def update_csv(df):
    df.columns = ['date', 'Ток', 'Давление расплава до сит ЕХ-21401', 'Давление расплава до сит ЕХ-21402',
                  'Температура полипропилена перед сеткой фильтром',
                  'Частота P-25001A',
                  'Расход гранул полипропилена',
                  'Давление впереди вала', 'Частота вращения мотора EX21401-M03', 'Частота вращения мотора EX21401-M04',
                  'Температура охлаждающей воды', 'Температура  воды',
                  'Положение ножа', 'Положение диска',
                  'Расход в EX-21401', 'Расход воды в E-21402',
                  'Регулятор частоты врещения RFM-21304', 'Специальная энергия',
                  'Температура горячего масла EX21401-HE82',
                  'Температура горячего масла  EX21401-HE81', 'Температура фильеры экструдера',
                  'Температура поверхности диска', 'Температура в EX-21401',
                  'Температура оборотной воды',
                  'Ход штока P-25001A', 'Уровень в V-25001A',
                  'Расход подачи порошка пп от RF-21304', 'Фактический расход пропилена',
                  'Давление расплава после сит', 'Давление расплава после сит',
                  'Температура полипропилена перед фильерой', 'Температура в цилиндре 2 EX-21401',
                  'Внутренняя температура цилиндра №3', 'Внешняя температура цилиндра №2',
                  'Внешняя температура цилиндра №3', 'Внешняя температура цилиндра №4',
                  'Внешняя температура цилиндра №5', 'Внешняя температура цилиндра №6',
                  'Внешняя температура цилиндра №7', 'Внешняя температура цилиндра №8',
                  'Внешняя температура цилиндра №9', 'ЭКСТР.ДВИГ.ВЛАСТЬ...214JI200A',
                  'Мощность мотора EX21401-M01', 'Ток ЕХ-21401-М01', 'name']
    e = ['Василий Иванов'] + ['Радимир Иванов'] + ['Сергей Иванов'] + ['Иван Иванов'] + ['Павел Иванов'] + [
        'Андрей Павлович'] + ['Василий Павлович'] + ['Артем Павлович'] + ['Василий Радимирович'] + ['Николай Павлович'] \
        + ['Василий Евгеньевич'] + ['Андрей Евгеньевич'] + ['Радимир Станиславович'] + ['Евгений Станиславович'] + [
            'Иван Сергеевич'] + ['Павел Сергеевич'] + ['Василий Сергеевич'] + ['Андрей Сергеевич'] + [
            'Василий Николаевич'] + ['Андрей Николаевич'] \
        + ['Федор Евгеньевич'] + ['Роман Евгеньевич'] + ['Федор Станиславович'] + ['Роман Станиславович'] + [
            'Юрий Сергеевич'] + ['Константин Сергеевич'] + ['Федор Сергеевич'] + ['Роман Сергеевич'] + [
            'Федор Николаевич'] + ['Роман Николаевич'] \
        + ['Николай Евгеньевич'] + ['Артем Евгеньевич'] + ['Николай Станиславович'] + ['Артем Станиславович'] + [
            'Николай Сергеевич'] + ['Артем Сергеевич'] + ['Михаил Сергеевич'] + ['Ренат Сергеевич'] + [
            'Николай Николаевич'] + ['Артем Николаевич'] \
        + ['Артур Евгеньевич'] + ['Всеволод Евгеньевич'] + ['Егор Станиславович'] + ['Артур Станиславович'] + [
            'Женя Сергеевич'] + ['Владислав Сергеевич'] + ['Станислав Сергеевич'] + ['Всеволод Сергеевич'] + [
            'Станислав Николаевич'] + ['Всеволод Николаевич']
    length = len(df[df.name == 1])
    l = []
    for item in e[:int(df.shape[0] / length)]:
        l += [item] * length
    l = pd.Series(l)
    df.name = l
    falls = df.iloc[range(0, df.shape[0], length), :]
    for i in range(falls.shape[0]):
        df.iloc[i * length:(i + 1) * length, 0] = df.iloc[i * length:(i + 1) * length, 0] - falls.iloc[i, 0]
    options = []
    for c in falls.columns[2:]:
        options.append({'label': c, 'value': c})

    return df, falls, options

ALL, FALLS, options = update_csv(pd.read_csv('frames.csv', parse_dates=['date']))

app = dash.Dash(__name__)
app.config['suppress_callback_exceptions'] = True
app.scripts.config.serve_locally = True
server = app.server


column_name = pickle.load(open('column_name', 'rb'))
related_columns = pickle.load(open('related_columns', 'rb'))
DF = pd.read_csv('xxxsasha.csv', encoding='cp1251', parse_dates=[])
DF.columns = column_name

prediction_df = pd.read_csv('pppsasha.csv')
top_features = pd.read_csv('sasha.csv')

# reusable componenets
def make_dash_table(df):
    ''' Return a dash definitio of an HTML table for a Pandas dataframe '''
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table



def get_common_header():
    return [get_logo(), get_header(), html.Br([]), get_menu()]

# includes page/full view
def get_logo():
    logo = html.Div([

        html.Div([
            html.Img(src='https://stsgeo.ru/images/img/brands/sibur/sibur-logo-new.jpg', height='40', width='160')
        ], className="ten columns padded"),

        html.Div([
            # dcc.Link('Full View   ', href='/full-view')
        ], className="two columns page-view no-print")

    ], className="row gs-header")
    return logo


def get_header():
    header = html.Div([

        html.Div([
            html.H5('AUsome')
        ], className="twelve columns padded", style={'background-color': '#397F7E'})

    ], className="row gs-header gs-text-header")
    return header


def get_menu():
    menu = html.Div([

        dcc.Link('Личный кабинет   ', href='/profile', className="tab first four columns"),

        dcc.Link('Главная   ', href='/main', className="tab four columns"),

        dcc.Link('Накопленный опыт   ', href='/advice', className="tab four columns"),

    ], className="row ")
    return menu

@app.callback(
    Output('datatable-gapminder', 'selected_row_indices'),
    [Input('graph-gapminder', 'clickData')],
    [State('datatable-gapminder', 'selected_row_indices')])
def update_selected_row_indices(clickData, selected_row_indices):
    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])
    return selected_row_indices

@app.callback(
    Output('graph-gapminder', 'figure'),
    [Input('datatable-gapminder', 'rows'),
     Input('datatable-gapminder', 'selected_row_indices'),
     Input('my-dropdown', 'value')])
def update_figure(rows, selected_row_indices, selected_dropdown_value):
    dff = ALL
    fig = plotly.tools.make_subplots(
        rows=1, cols=1,
        subplot_titles=([selected_dropdown_value]),
        shared_xaxes=True)
    colors = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, FALLS.shape[0])]

    selected_row_names = list(set(list(FALLS.iloc[selected_row_indices,:].name)))

    for i, name in enumerate(selected_row_names):
        sub_dff = dff[dff.name == name]
        fig.append_trace({
            'x': sub_dff['date'],
            'y': sub_dff[selected_dropdown_value],
            'type' : 'scatter',
            'line' : dict(
                shape="spline",
                smoothing=2,
                width=1,
                color=colors[i]
            ),
            'name':name
        }, 1, 1)

    fig['layout']['showlegend'] = False
    fig['layout']['height'] = 800
    fig['layout']['margin'] = {
        'l': 40,
        'r': 10,
        't': 60,
        'b': 200
    }

    return fig



def get_profile_page():
    return [
        dcc.Dropdown(
            id='profile-dropdown',
            options=[{'label': c, 'value': c} for c in DF.columns if c != 'date'],
            value=DF.columns.values[1],
        ),
        html.Div([
            html.Div([
                dcc.Graph(id='profile-graph'),
            ], className='nine columns', style={'height': '300px', 'margin': 'auto'}),
            dcc.Interval(id='profile-update', interval=1000, n_intervals=0),

            html.Div([
                html.Div([
                    html.Div(['Отчет'], style={'font-size': '30', 'text-align': 'center'}),
                    html.Img(src='/static/img/report.jpg', height=400, width=300),
                    html.Div([html.Button('Отправить', type='button', className='btn btn-primary btn-block')], className='btn-group-verical btn-block'),
                    # html.Div([html.Button('НЕ ОК', type='button', className='btn btn-primary btn-block')], className='btn-group-verical btn-block'),
                    # html.Div([html.Button('КЕК', type='button', className='btn btn-primary btn-block')], className='btn-group-verical  btn-block'),
                ],  className='three columns')
            ], className='two columns')
        ], className='row'),
        html.Div([
            dcc.Graph(id='profile-subgraph1', className='three columns', style={'height': '250px', 'margin': '0.5%'}),
            dcc.Graph(id='profile-subgraph2', className='three columns', style={'height': '250px', 'margin': '0.5%'}),
            dcc.Graph(id='profile-subgraph3', className='three columns', style={'height': '250px', 'margin': '0.5%'}),
        ]),

    ]

PROFILE_ITER_NUM = [0]

def profile_plot_for_column(c, is_main=False):
    dff = DF.iloc[PROFILE_ITER_NUM[0]: PROFILE_ITER_NUM[0] + 200]

    fig = plotly.tools.make_subplots(
        rows=1, cols=1,
        subplot_titles=([c]),
        shared_xaxes=True)

    colors = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, FALLS.shape[0])]

    fig.append_trace({
        'x': dff['date'],
        'y': dff[c],
        'type' : 'scatter',
        'line' : dict(
            shape="spline",
            smoothing=2,
            width=1,
            color=colors[0]
        ),
        'name':c
    }, 1, 1)

    fig['layout']['showlegend'] = False
    # fig['layout']['height'] = 200
    if not is_main:
        fig['layout']['titlefont'] = {
            'size': 4
        }
    fig['layout']['margin'] = {
        'l': 15,
        'r': 15,
        't': 20,
        'b': 15
    }

    return fig

@app.callback(
    Output('profile-graph', 'figure'),
    [Input('profile-dropdown', 'value'), Input('profile-update', 'n_intervals')]
)
def update_profile_figure(selected_dropdown_value, intervals):
    return profile_plot_for_column(selected_dropdown_value)

@app.callback(
    Output('profile-subgraph1', 'figure'),
    [Input('profile-dropdown', 'value'), Input('profile-update', 'n_intervals')]
)
def update_profile_figure1(selected_dropdown_value, intervals):
    return profile_plot_for_column(column_name[related_columns[column_name.index(selected_dropdown_value) - 1][0]])

@app.callback(
    Output('profile-subgraph2', 'figure'),
    [Input('profile-dropdown', 'value'), Input('profile-update', 'n_intervals')]
)
def update_profile_figure2(selected_dropdown_value, intervals):
    return profile_plot_for_column(column_name[related_columns[column_name.index(selected_dropdown_value) - 1][1]])

@app.callback(
    Output('profile-subgraph3', 'figure'),
    [Input('profile-dropdown', 'value'), Input('profile-update', 'n_intervals')]
)
def update_profile_figure3(selected_dropdown_value, intervals):
    PROFILE_ITER_NUM[0] += 1
    return profile_plot_for_column(column_name[related_columns[column_name.index(selected_dropdown_value) - 1][2]])


def get_main_page():
    return [
        dcc.Dropdown(
            id='main-dropdown',
            options=[{'label': c, 'value': c} for c in DF.columns if c != 'date'],
            value=DF.columns.values[1]
        ),
        html.Div([
            html.Div([
                dcc.Graph(id='main-graph'),
            ], className='nine columns'),
            html.Div([
                dcc.Graph(
                    id='example-graph',
                    style={'height': '300px', 'margin': 'auto'}
                ),
                'На эти параметры стоит ориентироваться больше, чем на остальные',
                dcc.RadioItems(
                    id='radio-features',
                    options=[{'label': i, 'value': i} for i in ['f1', 'f2', 'f3']],
                    value='f1',
                    #style={'margin': 'auto'}
                ),

                html.Div(id='output-a'),

                html.Div(children='''
                Пожалуйста, оцените качество предсказания
                '''),

                dcc.Slider(
                    id='my-slider',
                    min=0,
                    max=10,
                    step=1,
                    value=5,
                    marks={
                        0: '0',
                        3: '3',
                        5: '5',
                        7: '7',
                        10: '10'
                    },
                )
            ], className='three columns', style={'margin-left': '0.5%', 'vertical-align': 'middle'})
        ], className='row'),
        html.Div([
            dcc.Graph(id='main-subgraph1', className='three columns', style={'height': '250px', 'margin': '0.5%'}),
            dcc.Graph(id='main-subgraph2', className='three columns', style={'height': '250px', 'margin': '0.5%'}),
            dcc.Graph(id='main-subgraph3', className='three columns', style={'height': '250px', 'margin': '0.5%'})

            # html.Img(src='/static/img/alisa.png', className='three columns', width=100, height=100, style={
            #     'margin-left': '80'
            # })
        ]),
        html.P(id='placeholder'),
        dcc.Interval(id='main-update', interval=1000, n_intervals=0),
        dcc.Interval(id='radio-update', interval=5000, n_intervals=0),

    ]

MAIN_ITER_NUM = [0]
MAIN_COLS = [DF.columns.values[1] for i in range(4)]

def main_plot_for_column(c, is_main=False):
    dff = DF.iloc[MAIN_ITER_NUM[0]: MAIN_ITER_NUM[0] + 200]
    fig = plotly.tools.make_subplots(
        rows=1, cols=1,
        subplot_titles=([c]),
        shared_xaxes=True)

    colors = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, FALLS.shape[0])]
    fig.append_trace({
        'x': dff['date'],
        'y': dff[c],
        'type' : 'scatter',
        'line' : dict(
            shape="spline",
            smoothing=2,
            width=1,
            color=colors[0]
        ),
        'name':c
    }, 1, 1)

    fig['layout']['showlegend'] = False
    # fig['layout']['height'] = 200
    if not is_main:
        fig['layout']['titlefont'] = {
            'size': 4
        }
    fig['layout']['margin'] = {
        'l': 15,
        'r': 15,
        't': 20,
        'b': 15
    }
    return fig

@app.callback(
    Output('main-graph', 'figure'),
    [Input('main-dropdown', 'value'), Input('main-update', 'n_intervals')]
)
def update_main_figure(selected_dropdown_value, intervals):
    if selected_dropdown_value is not None:
        MAIN_COLS[0] = selected_dropdown_value
    return main_plot_for_column(MAIN_COLS[0], True)

@app.callback(
    Output('main-subgraph1', 'figure'),
    [Input('main-dropdown', 'value'), Input('main-update', 'n_intervals')]
)
def update_main_figure1(selected_dropdown_value, intervals):
    if selected_dropdown_value is not None:
        MAIN_COLS[1] = selected_dropdown_value
    return main_plot_for_column(column_name[related_columns[column_name.index(MAIN_COLS[1]) - 1][0]])

@app.callback(
    Output('main-subgraph2', 'figure'),
    [Input('main-dropdown', 'value'), Input('main-update', 'n_intervals')]
)
def update_main_figure2(selected_dropdown_value, intervals):
    if selected_dropdown_value is not None:
        MAIN_COLS[2] = selected_dropdown_value
    return main_plot_for_column(column_name[related_columns[column_name.index(MAIN_COLS[2]) - 1][1]])

@app.callback(
    Output('main-subgraph3', 'figure'),
    [Input('main-dropdown', 'value'), Input('main-update', 'n_intervals')]
)
def update_main_figure3(selected_dropdown_value, intervals):
    if selected_dropdown_value is not None:
        MAIN_COLS[2] = selected_dropdown_value
    MAIN_ITER_NUM[0] += 1
    return main_plot_for_column(column_name[related_columns[column_name.index(MAIN_COLS[3]) - 1][2]])

RADIO_ITER_NUM = [0]

@app.callback(
    Output('example-graph', 'figure'),
    [Input('radio-update', 'n_intervals')]
)
def update_prediction(intervals):
    pred = str(int(prediction_df.iloc[RADIO_ITER_NUM[0], 0] * 100))

    return {
        "data": [{

            # update time
            "values": [80, 20],
            "labels": [
                "Passed time",
                "Remaning time"
            ],

            # adjust timer placement
            "domain": {"x": [0, 1]},
            "hoverinfo": "label",
            "hole": .9,
            "type": "pie",
            "textinfo": "none"
        }],
        "layout": {
            # "autosize": False,
            # "width": 270,
            "height": 280,

            "title": "Предсказание уроня риска",
            "showlegend": False,
            "annotations": [{
                "font": {
                    "size": 60
                },
                "showarrow": False,

                # replace here with prediction
                "text": pred,

                # adjust text placement
                "x": 0.5,
                "y": 0.5
            }]
        }
    }


@app.callback(
    Output('radio-features', 'options'),
    [Input('radio-update', 'n_intervals')]
)
def update_radio_features(intervals):
    d = []
    RADIO_ITER_NUM[0] += 1
    for i in range(3):
        c = column_name[top_features.iloc[RADIO_ITER_NUM[0], i]]
        d.append({'label': c, 'value': c})
    return d


@app.callback(
    Output('main-dropdown', 'value'),
    [Input('radio-features', 'value')]
)
def update_radio_select(val):
    if val == 'f1':
        return MAIN_COLS[0]
    return val



def get_advice_page():
    tmp_df = FALLS.copy()
    print(tmp_df.columns)
    tmp_df.columns = np.hstack([['Дата'], tmp_df.columns[1:-1], ['Имя']])
    cols =  tmp_df.columns[1:-1]
    return [
        html.Div([
            html.Div(children=[
                dt.DataTable(
                    rows=tmp_df.to_dict('records'),

                    # optional - sets the order of columns
                    columns=['Дата', 'Имя'] + list(sorted( tmp_df.columns[1:-1])),

                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[2],
                    id='datatable-gapminder'
                ),
            ], className='six columns'),

            html.Div(children=[
                dcc.Dropdown(
                    id='my-dropdown',
                    options=options,
                    value=ALL.columns[12]
                ),
                dcc.Graph(
                    id='graph-gapminder',
                    style={'height': '450px'}
                )
            ], className='six columns')
        ], className='row'),
        html.Div([
            # html.Img(src='/static/img/Microphone-icon.png', height=80, width=80, style={'vertical-align': 'middle'}),
            'В аналогичной ситуации Сергей Иванов увеличил расстояние ножей до 13 мм'
        ], className='row', style={'font-size': 20})
    ]

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/main' or pathname == '/':
        return main_page
    elif pathname == '/profile':
        return profile_page
    elif pathname == '/advice':
        return advice_page
    else:
        return html.Div([  # 404
            html.P(["404 Page not found"])
        ], className="no-page")



main_page = html.Div([html.Div(get_common_header() + get_main_page())])
profile_page = html.Div([html.Div(get_common_header() + get_profile_page())])
advice_page = html.Div([html.Div(get_common_header() + get_advice_page())])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
dt.DataTable(
        rows=FALLS.to_dict('records'),

        # optional - sets the order of columns
        columns=sorted(FALLS.columns),

        row_selectable=True,
        filterable=True,
        sortable=True,
        selected_row_indices=[],
        id='datatable-gapminder'
    )], hidden=True),
    html.Div(id='page-content')
])


@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "https://codepen.io/bcd/pen/KQrXdb.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

app.css.append_css({"external_url": "/static/custom.css"})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})


if __name__ == '__main__':
    app.run_server(debug=True)
