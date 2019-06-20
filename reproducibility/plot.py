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


def plot_lines(lines: List[List[go.Scatter]], graph_name='test', y_name='Objective', folder=''):
    """ Aggregate the the problem in `problem_ids` """
    all_lines = []
    for line in lines:
        all_lines.extend(line)

    font = 'Courier New, monospace'

    h = 640
    w = h * 10 / 9

    layout = go.Layout(
        width=640,
        height=h,
        margin=go.layout.Margin(
            l=0.1 * h,
            r=0.1 * h,
            b=0.1 * w,
            t=0.1 * w,
            pad=4
        ),
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
                text='Epoch',
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
                text=y_name,
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
                size=10,
                color='#7f7f7f'
            )
        )
    )
    fig = go.Figure(data=all_lines, layout=layout)
    # py.plot(fig, filename=f'results/graphs/{graph_name}.html', auto_open=False)
    pio.write_image(fig, f'{folder}/{graph_name}.png')


def get_curve_with_error(db, name, metric='train_loss', color=(0, 176, 246)):
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

color_palette = [
    (0,  176, 246),
    (246, 176, 0),

    (176, 0, 246),
    (176, 246, 0),

    (0, 256, 176),
    (256, 0, 176),
]

metric = 'test_acc'

for m, name in [('test_acc', 'Test Accuracy'), ('train_loss', 'Train Loss'), ('test_loss', 'Test Loss')]:

    amd_2 = get_curve_with_error(f'{BASE}/results/amd_2.json', name=f'AMD', metric=m, color=color_palette[0])
    cpu_2 = get_curve_with_error(f'{BASE}/results/cpu_2.json', name=f'CPU', metric=m, color=color_palette[1])

    plot_lines([amd_2, cpu_2], graph_name=f'different_initialization_{m}', y_name=name, folder=f'{BASE}/graphs')

    amd_1 = get_curve_with_error(f'{BASE}/results/amd_1.json', name=f'AMD', metric=m, color=color_palette[0])
    cpu_1 = get_curve_with_error(f'{BASE}/results/cpu_1.json', name=f'CPU', metric=m, color=color_palette[1])

    plot_lines([amd_1, cpu_1], graph_name=f'same_initialization_{m}', y_name=name, folder=f'{BASE}/graphs')



