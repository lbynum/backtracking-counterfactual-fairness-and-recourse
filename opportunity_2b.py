import torch
import numpy as np
import pandas as pd
from dowhy import gcm
import networkx as nx
import seaborn as sns
from scipy import stats
from sklearn import base
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from scmtools.utils import get_dot_graph
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scmtools.model import build_ground_truth_causal_model
from scmtools.backtracking import sample_noninformative_backtracking_counterfactuals

pd.set_option('display.max_columns', None)


dataset_size = 500
num_backtracking_samples = 1000
num_loss_function_samples = 1000

true_outcome_function = lambda x: x[:, 0] + x[:, 1] + 2 * x[:, 2] + x[:, 3] - 2
node_function_dict = {
    'A': (None, gcm.ScipyDistribution(stats.bernoulli, p=0.5)),
    'Z_A': (lambda x: 0.5 * x, gcm.ScipyDistribution(stats.norm, loc=0, scale=1)),
    'Z_A_prime': (None, gcm.ScipyDistribution(stats.norm, loc=0, scale=1)),
    'X_1': (lambda x: 2 * x[:, 0] + x[:, 1], gcm.ScipyDistribution(stats.norm, loc=0, scale=1)),
    'X_2': (lambda x: 3 * x, gcm.ScipyDistribution(stats.norm, loc=0, scale=1)),
    'Y': (true_outcome_function, gcm.ScipyDistribution(stats.norm, loc=0, scale=1))
}
causal_graph = nx.DiGraph([('A', 'Z_A'), ('Z_A', 'X_1'), ('Z_A_prime', 'X_1'), ('Z_A_prime', 'X_2'), ('Z_A_prime', 'Y'), ('X_1', 'Y'), ('X_2', 'Y'), ('Z_A', 'Y')])
get_dot_graph(graph=causal_graph).view()

outcome_name = 'Y'
protected_attribute = 'A'

np.random.seed(1)
causal_model = build_ground_truth_causal_model(causal_graph=causal_graph, node_function_dict=node_function_dict)
data_df = gcm.draw_samples(causal_model=causal_model, num_samples=dataset_size)

train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=2)
print(f'Train examples: {len(train_df)}\nTest examples: {len(test_df)}')
X_train, X_test = train_df.drop(columns=[outcome_name]), test_df.drop(columns=[outcome_name])
y_train, y_test = train_df[outcome_name], test_df[outcome_name]
print(train_df)

backtracking_conditional_dict = {
    'A': False,
    'Z_A': False,
    'Z_A_prime': True,
    'X_1': True,
    'X_2': True,
    'Y': True
}
print(f'Sampling {num_backtracking_samples} backtracking counterfactuals per observation...')
joint_data_df = sample_noninformative_backtracking_counterfactuals(
    causal_model=causal_model,
    backtracking_conditional_dict=backtracking_conditional_dict,
    observed_data=train_df,
    num_backtracking_samples=num_backtracking_samples
)
print('Done.')

original_columns = X_train.columns
original_X = joint_data_df[original_columns]

star_columns = [f'{name}_star' for name in original_columns]
star_X_renamed = joint_data_df[star_columns].rename(columns=dict(zip(star_columns, original_columns)))

print(joint_data_df)

class RandomModel(base.BaseEstimator):
    def fit(self, X, y):
        pass
    def predict(self, X):
        y_pred = np.random.binomial(n=1, size=len(X), p=0.5)
        return y_pred

models_to_test = [
    ("$Z_{A'}$", LinearRegression(), ['Z_A_prime']),
    ('$X_2$', LinearRegression(), ['X_2']),
    ("$X_1, X_2$", LinearRegression(), ['X_1', 'X_2']),
    ("$X_1, Z_{A'}$", LinearRegression(), ['X_1', 'Z_A_prime']),
    ('$X_2, Z_A$', LinearRegression(), ['X_2', 'Z_A']),
    ("$X_2, Z_{A'}$", LinearRegression(), ['X_2', 'Z_A_prime']),
    ("$X_1, X_2,$\n$Z_A$", LinearRegression(), ['X_1', 'X_2', 'Z_A']),
    ("$X_1, X_2,$\n$Z_{A'}$", LinearRegression(), ['X_1', 'X_2', 'Z_A_prime']),
    ("$X_1, X_2,$\n$Z_{A'}, Z_A$", LinearRegression(), ['X_1', 'X_2', 'Z_A', 'Z_A_prime']),
    ('Random', RandomModel(), original_columns)
]


for model_description, black_box_model, black_box_feature_names in models_to_test:
    black_box_model.fit(X=X_train[black_box_feature_names], y=y_train)

    y_pred = (black_box_model.predict(X=X_test[black_box_feature_names]) > 0).astype(int)
    y_true = (y_test > 0).astype(int)
    print(f'{model_description} test performance:')
    print(classification_report(y_pred=y_pred, y_true=y_true))
    print('-' * 80)

opportunity_set = ['X_2', 'Z_A_prime']
opportunity_set_star = [f'{variable_name}_star' for variable_name in opportunity_set]

def individual_equal_opportunity_lhs_condition(observed_row, protected_attribute, outcome_name):
    observed_outcome = observed_row[outcome_name]
    if observed_outcome > 0:
        flipped_outcome_query = '<= 0'
        observed_outcome_query = '> 0'
    else:
        flipped_outcome_query = '> 0'
        observed_outcome_query = '<= 0'

    covariate_names = observed_row.index.tolist()
    covariate_names.remove(outcome_name)
    observed_covariates_query = ' & '.join(f'{variable} == {observed_row[variable]}' for variable in covariate_names)

    query_string = f'{observed_covariates_query} & {outcome_name} {observed_outcome_query} & {outcome_name}_star {flipped_outcome_query}'
    return query_string

def individual_equal_opportunity_rhs_condition(observed_row, protected_attribute, outcome_name):
    a_observed = observed_row[protected_attribute]
    a_prime = 1 - a_observed

    observed_outcome = observed_row[outcome_name]
    if observed_outcome > 0:
        flipped_outcome_query = '<= 0'
    else:
        flipped_outcome_query = '> 0'

    query_string = f'{protected_attribute} == {a_prime} & {outcome_name}_star {flipped_outcome_query}'
    return query_string


def compute_loss_between_lhs_vs_rhs(protected_attribute, outcome_name, joint_data_df, train_df, X_train, y_train,
                                    original_X, star_X_renamed, loss_function, lhs_condition_function,
                                    rhs_condition_function, lhs_variables, rhs_variables, num_loss_function_samples,
                                    models_to_test, plot_marginals=False):
    distances_dict = {}
    for model_description, black_box_model, black_box_feature_names in models_to_test:
        black_box_model.fit(X=X_train[black_box_feature_names], y=y_train)

        cf_data_with_predictions = joint_data_df.copy(deep=True)
        cf_data_with_predictions[outcome_name] = black_box_model.predict(original_X[black_box_feature_names])
        cf_data_with_predictions[f'{outcome_name}_star'] = black_box_model.predict(star_X_renamed[black_box_feature_names])

        train_df_with_predictions = train_df.copy(deep=True)
        train_df_with_predictions[outcome_name] = black_box_model.predict(train_df[black_box_feature_names])

        distances_list = []
        for index in tqdm(list(range(len(train_df_with_predictions))), desc=f"Computing MMD for {model_description}"):
            observed_row = train_df_with_predictions.iloc[index]
            lhs_condition = lhs_condition_function(observed_row=observed_row, protected_attribute=protected_attribute, outcome_name=outcome_name)
            rhs_condition = rhs_condition_function(observed_row=observed_row, protected_attribute=protected_attribute, outcome_name=outcome_name)
            lhs_data = cf_data_with_predictions.query(expr=lhs_condition)
            rhs_data = cf_data_with_predictions.query(expr=rhs_condition)

            lhs_samples = lhs_data[lhs_variables]
            rhs_samples = rhs_data[rhs_variables]

            if plot_marginals:
                fig, axes = plt.subplots(
                    nrows=2,
                    ncols=len(lhs_variables),
                    dpi=150,
                    figsize=(12, 4),
                    sharex=True
                )

                axes = axes.ravel()
                for i, (lhs_name, rhs_name) in enumerate(zip(lhs_variables, rhs_variables)):
                    axes[i].hist(lhs_samples[lhs_name])
                    axes[i].set_title(f'LHS {lhs_name}')
                    axes[i + len(lhs_variables)].hist(rhs_samples[rhs_name])
                    axes[i + len(lhs_variables)].set_title(f'RHS {rhs_name}')
                plt.suptitle(f'LHS {lhs_variables} vs. RHS {rhs_variables}')
                plt.tight_layout()
                plt.show(block=False)

            if len(lhs_samples) < 2:
                raise RuntimeError(f'Not enough LHS samples for observation:\n\t{observed_row}\nGiven query:\n\t{lhs_condition}')

            if len(rhs_samples) < 2:
                raise RuntimeError(f'Not enough RHS samples for observation:\n\t{observed_row}\nGiven query:\n\t{rhs_condition}')

            if len(lhs_samples) > num_loss_function_samples:
                lhs_samples = lhs_samples.sample(num_loss_function_samples)
            if len(rhs_samples) > num_loss_function_samples:
                rhs_samples = rhs_samples.sample(num_loss_function_samples)

            lhs_dist = torch.tensor(
                np.concatenate(
                    [np.expand_dims(lhs_samples[name].values, axis=1)
                    for name in lhs_variables],
                    axis=1
                )
            )
            rhs_dist = torch.tensor(
                np.concatenate(
                    [np.expand_dims(rhs_samples[name].values, axis=1)
                    for name in rhs_variables],
                    axis=1
                )
            )
            distance_between_distributions = loss_function(lhs_dist, rhs_dist).item()
            distances_list.append(distance_between_distributions)
        distances_dict[model_description] = distances_list
        print(model_description)
        print(f'Mean MMD opportunity distance: {np.mean(distances_list)}')

    distances_df = pd.DataFrame(distances_dict)
    return distances_df

distances_df = compute_loss_between_lhs_vs_rhs(
    protected_attribute=protected_attribute,
    outcome_name=outcome_name,
    joint_data_df=joint_data_df,
    train_df=train_df,
    X_train=X_train,
    y_train=y_train,
    original_X=original_X,
    star_X_renamed=star_X_renamed,
    loss_function=SamplesLoss(loss='energy'),
    lhs_condition_function=individual_equal_opportunity_lhs_condition,
    rhs_condition_function=individual_equal_opportunity_rhs_condition,
    lhs_variables=opportunity_set_star,
    rhs_variables=opportunity_set_star,
    num_loss_function_samples=num_loss_function_samples,
    models_to_test=models_to_test
)
print('Mean MMD distances:')
print(distances_df.mean(axis=0))

plt.figure(figsize=(6, 8))
sns.set(rc={'text.usetex' : True}, font_scale=2, context='paper')
sns.set_style('whitegrid')
sns.boxplot(distances_df.abs(), orient='h')
sns.despine(right = True)
plt.axvline(x=0, linestyle='dashed', color='r')
plt.xlabel('Absolute Energy Distance MMD')
plt.gca().xaxis.grid(True)
plt.tight_layout()

filename = f"results/opportunity_s2b_{','.join(opportunity_set)},n={dataset_size},nS={num_backtracking_samples},nL={num_loss_function_samples}"
distances_df.to_csv(f'{filename}.csv')
print(f'Saved to: {filename}.csv')
plt.show()
