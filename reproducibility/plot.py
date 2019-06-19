import os
import pandas as pd
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
from typing import List
from track.persistence.storage import LocalStorage, load_database


# Differen Init

BASE = os.path.dirname(os.path.realpath(__file__)) + '/../'


def plot_lines(lines: List[List[go.Scatter]], graph_name='test'):
    """ Aggregate the the problem in `problem_ids` """
    all_lines = []
    for line in lines:
        all_lines.extend(line)

    font = 'Courier New, monospace'

    layout = go.Layout(
        title=go.layout.Title(
            text=graph_name,
            xref='paper',
            font=dict(
                family=font,
                size=14,
                color='#7f7f7f'
            )
        ),
        xaxis=go.layout.XAxis(
            type='log',
            title=go.layout.xaxis.Title(
                text='Duration (s)',
                font=dict(
                    family=font,
                    size=12,
                    color='#7f7f7f'
                )
            )
        ),
        yaxis=go.layout.YAxis(
            type='log',
            title=go.layout.yaxis.Title(
                text='Objective',
                font=dict(
                    family=font,
                    size=12,
                    color='#7f7f7f'
                )
            )
        ),
        legend=dict(
            orientation="h",
            font=dict(
                family=font,
                size=12,
                color='#7f7f7f'
            ),
        )
    )
    fig = go.Figure(data=all_lines, layout=layout)
    # py.plot(fig, filename=f'results/graphs/{graph_name}.html', auto_open=False)
    pio.write_image(fig, f'{graph_name}.png')


def get_curve_with_error(db, name, metric='train_loss'):
    amd = load_database(db)

    loss = {}

    # train_loss, test_loss, test_acc

    for i, trial_id in enumerate(amd.trials):
        trial = amd.objects[trial_id]

        loss[i] = {int(k): v for k, v in trial.metrics[metric].items()}

    df = pd.DataFrame(loss)
    max_metric = df.max(axis=1)
    min_metric = df.min(axis=1)
    avg_metric = df.mean(axis=1)

    df = pd.DataFrame({
        'avg': avg_metric,
        'min': min_metric,
        'max': max_metric
    })

    color = (0, 176, 246)
    color_full = ','.join(map(lambda x: str(x), color))
    color_dim = f'rgba({color_full},0.1)'
    color_full = f'rgb({color_full})'

    upper_bound = go.Scatter(
        name='upper',
        x=df.index,
        y=max_metric,  # y_max,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor=color_dim,
        fill='tonexty',
        showlegend=False
    )
    mean = go.Scatter(
        x=df.index,
        y=avg_metric,
        line=dict(color=color_full),
        mode='lines',
        name=name,
        fillcolor=color_dim,
        fill='tonexty'
    )
    lower_bound = go.Scatter(
        name='lower',
        x=df.index,
        y=min_metric,  # y_min,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        showlegend=False
    )
    return [lower_bound, mean, upper_bound]


amd_1 = get_curve_with_error(f'{BASE}/amd_2.json', name='(2) AMD - Cost', metric='test_acc')
plot_lines([amd_1], graph_name='amd_2')


amd_2 = get_curve_with_error(f'{BASE}/amd_1.json', name='(1) AMD - Cost', metric='test_acc')
plot_lines([amd_2], graph_name='amd_1')

