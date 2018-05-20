import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div(children=[
    dcc.Graph(
        id='example-graph',
        figure={
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
                "autosize": False,

                "title": "Prediction and time",
                "showlegend": False,
                "annotations": [{
                        "font": {
                            "size": 70
                        },
                        "showarrow": False,

                        # replace here with prediction
                        "text": "85",

                        # adjust text placement
                        "x": 0.5,
                        "y": 0.5
                }]
            }
        }
    ),

    dcc.RadioItems(
        id='dropdown-a',
        options=[{'label': i, 'value': i} for i in ['Canada', 'USA', 'Mexico']],
        value='Canada'
    ),

    html.Div(id='output-a'),

    html.Div(children='''
    Please, evaluate prediction.
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
])

@app.callback(
    dash.dependencies.Output('output-a', 'children'),
    [dash.dependencies.Input('dropdown-a', 'value')])
def callback_a(dropdown_value):
    return 'You\'ve selected "{}"'.format(dropdown_value)


if __name__ == '__main__':
    app.run_server(debug=True)