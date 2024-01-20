import numpy as np
import pandas as pd
from dowhy import gcm
import networkx as nx
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from scmtools.utils import get_dot_graph
from scmtools.model import build_ground_truth_causal_model
from scmtools.backtracking import sample_factual_centered_backtracking_counterfactuals

pd.set_option('display.max_columns', None)


dataset_size = 1000
num_backtracking_samples = 500

node_function_dict = {
    'A': (None, gcm.ScipyDistribution(stats.bernoulli, p=0.5)),
    'X': (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1)),
    'Y': (lambda x: (100 * x[:, 0] + x[:, 1]) > 0, gcm.ScipyDistribution(stats.norm, loc=0, scale=0))
}
causal_graph = nx.DiGraph([('X', 'Y'), ('A', 'Y')])

outcome_name = 'Y'

np.random.seed(1)
get_dot_graph(graph=causal_graph).view()
causal_model = build_ground_truth_causal_model(causal_graph=causal_graph, node_function_dict=node_function_dict)
data_df = gcm.draw_samples(causal_model=causal_model, num_samples=dataset_size)
print(data_df)

backtracking_conditional_dict = {
    'A': False,
    'X': True
}
print(f'Sampling {num_backtracking_samples} backtracking counterfactuals per observation...')
joint_data_df = sample_factual_centered_backtracking_counterfactuals(
    outcome_name=outcome_name,
    causal_model=causal_model,
    backtracking_conditional_dict=backtracking_conditional_dict,
    observed_data=data_df,
    num_backtracking_samples=num_backtracking_samples,
    backtracking_variance=1
)
print('Done.')
print(joint_data_df)

np.random.seed(123)
point_color = 'red'
other_point_color = 'dodgerblue'
color_0 = 'black'
color_1 = 'black'
color_2 = 'black'
color_3 = 'black'
color_4 = 'black'
color_fill = True
color_alpha = 0

hatches = ['//', '\\\\', '||', '..', '--', 'XX', 'oo']

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(8.5, 5.5), dpi=300)
axes = axes.ravel()

sns.kdeplot(
    data=data_df.query(expr='A == 0'),
    x='X',
    fill=color_fill,
    alpha=color_alpha,
    bw_adjust=1,
    ax=axes[0],
    color=color_1,
    hatch='\\'
)

sns.kdeplot(
    data=data_df.query(expr='A == 1'),
    x='X',
    fill=color_fill,
    alpha=color_alpha,
    bw_adjust=1,
    ax=axes[0],
    color=color_2,
    hatch='/'
)

sns.kdeplot(
    data=data_df.query(expr='A == 0 & Y == 0'),
    x='X',
    fill=color_fill,
    alpha=color_alpha,
    bw_adjust=1,
    clip=(None, 0),
    ax=axes[0],
    color=color_0,
    hatch='||'
)

unit_index = np.argmin(abs(1 + joint_data_df.X) + joint_data_df.A)
print(unit_index)
unit_data = joint_data_df.iloc[unit_index]
print(unit_data)
assert(unit_data.X < 0)
assert(unit_data.A == 0)
assert(unit_data.Y == 0)

axes[0].scatter(x=unit_data.X, y=0, color=point_color, clip_on=False, edgecolor='black')

other_unit_index = np.argmin(abs(1.5 + joint_data_df.X) - joint_data_df.A)
print(other_unit_index)
other_unit_data = joint_data_df.iloc[other_unit_index]
print(other_unit_data)
assert(other_unit_data.X < 0)
assert(other_unit_data.A == 1)
assert(other_unit_data.Y == 1)

axes[0].scatter(x=other_unit_data.X, y=0, color=other_point_color, clip_on=False, edgecolor='black', marker='s')

axes[0].legend(['$P(X|A=0)$', '$P(X|A=1)$', '$P(X|\hat{Y}=0, A=0)$', '$(\hat{Y}=0, A=0, X=-1)$', '$(\hat{Y}=1, A=1, X=-1.5)$'], loc='upper left')
axes[0].axvline(x=0, ymin=0, ymax=1, color='black', linestyle='dashed')

query = f'X == {unit_data.X} & A == 0 & Y == 0 & Y_star == 1'
unit_query_df = joint_data_df.query(expr=query)
sns.kdeplot(
    data=unit_query_df,
    x='X_star',
    fill=color_fill,
    alpha=color_alpha,
    label='$P(X^*|\hat{Y}^*=1, \hat{Y}=0, A=0, X=-1)$',
    bw_adjust=1,
    clip=(0, None),
    ax=axes[1],
    color=point_color,
    hatch='//'
)

query = f'X == {other_unit_data.X} & A == 1 & Y == 1 & Y_star == 1'
other_unit_query_df = joint_data_df.query(expr=query)
sns.kdeplot(
    data=other_unit_query_df,
    x='X_star',
    fill=color_fill,
    alpha=color_alpha,
    label='$P(X^*|\hat{Y}^*=1, \hat{Y}=1, A=1, X=-1.5)$',
    bw_adjust=1,
    ax=axes[1],
    color=other_point_color,
    hatch='\\\\'
)

query = 'A == 0 & Y == 0 & Y_star == 1'
query_df = joint_data_df.query(expr=query)
sns.kdeplot(
    data=query_df,
    x='X_star',
    fill=color_fill,
    alpha=color_alpha,
    label='$P(X^*|\hat{Y}^*=1, \hat{Y}=0, A=0)$',
    bw_adjust=1,
    clip=(0, None),
    ax=axes[1],
    color=color_3,
    hatch='..'
)

query = 'A == 1 & Y_star == 1'
query_df = joint_data_df.query(expr=query)
sns.kdeplot(
    data=query_df,
    x='X_star',
    fill=color_fill,
    alpha=color_alpha,
    label='$P(X^*|\hat{Y}^*=1, A=1)$',
    bw_adjust=1,
    ax=axes[1],
    color=color_4,
    hatch='OO'
)

axes[1].axvline(x=0, ymin=0, ymax=1, color='black', linestyle='dashed')
axes[1].legend(loc='upper left')
axes[1].scatter(x=unit_data.X, y=0, color=point_color, clip_on=False, edgecolor='black')
axes[1].scatter(x=other_unit_data.X, y=0, color=other_point_color, clip_on=False, edgecolor='black', marker='s')
plt.xlabel('X')
axes[0].set_ylabel('Density Estimate')
axes[1].set_ylabel('Density Estimate')
plt.show()
