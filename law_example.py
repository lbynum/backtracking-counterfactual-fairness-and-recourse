import torch
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from dowhy import gcm
from scipy import stats
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from scmtools.backtracking import sample_noninformative_backtracking_counterfactuals

pd.set_option('display.max_columns', None)


np.random.seed(1)
dataset_size = 5000
law_df = pd.read_csv(filepath_or_buffer='data/law_data.csv', index_col=0).sample(dataset_size, replace=False)
law_df = law_df[law_df.region_first != 'PO']

law_df.sex = law_df.sex.replace({2: 0})
columns_to_keep = ['race', 'sex', 'LSAT', 'UGPA', 'ZFYA']
law_df = law_df[columns_to_keep]
outcome_column = 'Y'
law_df.rename(columns={'UGPA': 'GPA', 'ZFYA': outcome_column}, inplace=True)

majority_group = law_df['race'].mode()[0]
law_df['race'] = (law_df['race'] == majority_group).astype(int)

law_df[outcome_column] = (law_df[outcome_column] > 0).astype(int)

law_train_df, law_test_df = train_test_split(law_df, test_size=0.2, random_state=2)
print(f'Train examples: {len(law_train_df)}\nTest examples: {len(law_test_df)}')
X_train, X_test = law_train_df.drop(columns=[outcome_column]), law_test_df.drop(columns=[outcome_column])
y_train, y_test = law_train_df[outcome_column], law_test_df[outcome_column]

numeric_columns = ['LSAT', 'GPA']
scaler = StandardScaler()
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

print(X_train)

edge_list = [
    ('sex', 'LSAT'), ('sex', 'GPA'), ('sex', outcome_column),
    ('race', 'LSAT'), ('race', 'GPA'), ('race', outcome_column)
]
causal_graph = nx.DiGraph(incoming_graph_data=edge_list)
causal_model = gcm.InvertibleStructuralCausalModel(graph=causal_graph)

outcome_node = outcome_column
root_nodes = ['sex', 'race']
for node in causal_model.graph.nodes:
    if node in root_nodes:
        causal_model.set_causal_mechanism(node, gcm.EmpiricalDistribution())
    elif node == outcome_node:
        causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(
            prediction_model=gcm.ml.create_logistic_regression_classifier(),
            noise_model=gcm.ScipyDistribution(stats.norm)
        ))
    else:
        causal_model.set_causal_mechanism(node, gcm.AdditiveNoiseModel(
            prediction_model=gcm.ml.create_linear_regressor(),
            noise_model=gcm.ScipyDistribution(stats.norm)
        ))

causal_df = pd.concat([X_train, y_train.astype(int)], axis=1)
gcm.fit(causal_model=causal_model, data=causal_df)

full_linmod = LogisticRegression().fit(X=X_train, y=y_train)

unaware_columns = ['LSAT', 'GPA']
unaware_linmod = LogisticRegression().fit(X=X_train[unaware_columns], y=y_train)

X_train_lsat = X_train[['race', 'sex']]
y_train_lsat = X_train['LSAT']
lsat_linmod = LinearRegression().fit(X=X_train_lsat, y=y_train_lsat)

X_train_gpa = X_train[['race', 'sex']]
y_train_gpa = X_train['GPA']
gpa_linmod = LinearRegression().fit(X=X_train_gpa, y=y_train_gpa)

def compute_X_residuals(X):
    '''Use previously-fit linear regressions to compute residuals for LSAT and GPA.'''
    X_test_columns = ['race', 'sex']
    X_test = X[X_test_columns]

    y_test_lsat = X['LSAT']
    predicted_lsat = lsat_linmod.predict(X=X_test)
    epsilon_lsat = y_test_lsat - predicted_lsat

    y_test_gpa = X['GPA']
    predicted_gpa = gpa_linmod.predict(X=X_test)
    epsilon_gpa = y_test_gpa - predicted_gpa

    return pd.DataFrame({'epsilon_gpa': epsilon_gpa, 'epsilon_lsat': epsilon_lsat})


fair_linmod = LogisticRegression().fit(X=compute_X_residuals(X_train), y=y_train)

y_pred_full = full_linmod.predict(X=X_test)
accuracy_full = (y_pred_full == y_test).mean()
y_pred_unaware = unaware_linmod.predict(X=X_test[unaware_columns])
accuracy_unaware = (y_pred_unaware == y_test).mean()
y_pred_fair = fair_linmod.predict(X=compute_X_residuals(X=X_test))
accuracy_fair = (y_pred_fair == y_test).mean()
y_pred_dummy = np.array([y_train.mode()[0]] * len(y_test))
accuracy_dummy = (y_pred_dummy == y_test).mean()
print(f'\nFull model accuracy:\t{accuracy_full}')
print(f'Unaware model accuracy:\t{accuracy_unaware}')
print(f'Fair model accuracy:\t{accuracy_fair}')
print(f'Constant clf accuracy:\t{accuracy_dummy}\n')

counterfactual_data = {}
for racial_category in law_df.race.unique():
    intervention_dict = {'race': lambda x: racial_category}
    causal_test_df = pd.concat([X_train, y_train.astype(int)], axis=1)
    X_cf = gcm.counterfactual_samples(
        causal_model=causal_model,
        interventions=intervention_dict,
        observed_data=causal_test_df
    )
    counterfactual_data[racial_category] = X_cf
print(f'Counterfactual data for intervention: R <- {racial_category}')
print(counterfactual_data[racial_category].head())

y_pred_fair_dict = {}
y_pred_full_dict = {}
y_pred_unaware_dict = {}
for racial_category, X_cf in counterfactual_data.items():
    X_cf = X_cf[X_train.columns]
    y_pred_full = full_linmod.predict_proba(X=X_cf)[:, 1]
    y_pred_full_dict.update({f'R <— {racial_category}': y_pred_full})

    y_pred_unaware = unaware_linmod.predict_proba(X=X_cf[unaware_columns])[:, 1]
    y_pred_unaware_dict.update({f'R <— {racial_category}': y_pred_unaware})

    y_pred_fair = fair_linmod.predict_proba(X=compute_X_residuals(X=X_cf))[:, 1]
    y_pred_fair_dict.update({f'R <— {racial_category}': y_pred_fair})

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)
axes = axes.ravel()
pd.DataFrame(y_pred_full_dict).boxplot(rot=0, ylabel='P(Y=1)', ax=axes[0])
axes[0].set_title('Full Model')
pd.DataFrame(y_pred_unaware_dict).boxplot(rot=0, ylabel='P(Y=1)', ax=axes[1])
axes[1].set_title('Unaware Model')
pd.DataFrame(y_pred_fair_dict).boxplot(rot=0, ylabel='P(Y=1)', ax=axes[2])
axes[2].set_title('Level 3 Model')
plt.suptitle('Distrbution of P(Y=1) across counterfactual R settings')


num_backtracking_samples = 100
train_df = pd.concat([X_train, y_train.astype(int)], axis=1)
backtracking_conditional_dict = {
    'LSAT': True,
    'GPA': True,
    'race': False,
    'sex': False,
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

def compute_distance_given_observation(observed_row, data_df_with_predictions, outcome, balancing_variables,
                                       protected_attribute, num_loss_function_samples=1000):
    a_observed = observed_row[protected_attribute]
    a_prime = 1 - a_observed

    query_for_observed_X = ' & '.join(f'{variable}=={observed_row[variable]}' for variable in observed_row.index)
    data_df_for_observed_row = data_df_with_predictions.query(expr=query_for_observed_X)
    y_observed = data_df_for_observed_row[outcome].values[0]
    y_flipped = 1 - y_observed

    factual_query = f'{protected_attribute} == {a_prime} & {outcome}_star == {y_flipped}'
    counterfactual_query = f'{outcome}_star == {y_flipped} & {outcome} == {y_observed} & {query_for_observed_X}'

    factual_samples = data_df_with_predictions.query(expr=factual_query)
    counterfactual_samples = data_df_with_predictions.query(expr=counterfactual_query)

    if len(factual_samples) > num_loss_function_samples:
        factual_samples = factual_samples.sample(num_loss_function_samples)
    if len(counterfactual_samples) > num_loss_function_samples:
        counterfactual_samples = counterfactual_samples.sample(num_loss_function_samples)

    factual_balancing_variables = balancing_variables
    counterfactual_balancing_variables = [f'{variable_name}_star' for variable_name in factual_balancing_variables]

    loss_function = SamplesLoss(loss='energy')
    factual_dist = torch.tensor(
        np.concatenate(
            [np.expand_dims(factual_samples[name].values, axis=1)
            for name in counterfactual_balancing_variables],
            axis=1
        )
    )
    counterfactual_dist = torch.tensor(
        np.concatenate(
            [np.expand_dims(counterfactual_samples[name].values, axis=1)
            for name in counterfactual_balancing_variables],
            axis=1
        )
    )
    distance_between_distributions = loss_function(factual_dist, counterfactual_dist).item()

    return distance_between_distributions


np.random.seed(3)
num_rows = len(joint_data_df)

original_columns = X_train.columns
original_X = joint_data_df[original_columns]

star_columns = [f'{name}_star' for name in original_columns]
star_X = joint_data_df[star_columns].rename(columns=dict(zip(star_columns, original_columns)))

random_model_df = joint_data_df.copy(deep=True)
random_model_df['Y'] = stats.bernoulli.rvs(p=0.5, size=num_rows)
random_model_df['Y_star'] = stats.bernoulli.rvs(p=0.5, size=num_rows)

full_model_df = joint_data_df.copy(deep=True)
full_model_df['Y'] = full_linmod.predict(X=original_X)
full_model_df['Y_prob'] = full_linmod.predict_proba(X=original_X)[:, 1]
full_model_df['Y_star'] = full_linmod.predict(X=star_X)
full_model_df['Y_star_prob'] = full_linmod.predict_proba(X=star_X)[:, 1]

unaware_model_df = joint_data_df.copy(deep=True)
unaware_model_df['Y'] = unaware_linmod.predict(X=original_X[unaware_columns])
unaware_model_df['Y_prob'] = unaware_linmod.predict_proba(X=original_X[unaware_columns])[:, 1]
unaware_model_df['Y_star'] = unaware_linmod.predict(X=star_X[unaware_columns])
unaware_model_df['Y_star_prob'] = unaware_linmod.predict_proba(X=star_X[unaware_columns])[:, 1]

fair_model_df = joint_data_df.copy(deep=True)
fair_model_df['Y'] = fair_linmod.predict(X=compute_X_residuals(X=original_X))
fair_model_df['Y_prob'] = fair_linmod.predict_proba(X=compute_X_residuals(X=original_X))[:, 1]
fair_model_df['Y_star'] = fair_linmod.predict(X=compute_X_residuals(X=star_X))
fair_model_df['Y_star_prob'] = fair_linmod.predict_proba(X=compute_X_residuals(X=star_X))[:, 1]


outcome = 'Y'
balancing_variables = ['U_LSAT', 'U_GPA']
protected_attribute = 'race'
num_loss_function_samples = 1000

random_distances = [
    compute_distance_given_observation(
        X_train.iloc[index],
        data_df_with_predictions=random_model_df,
        outcome=outcome,
        balancing_variables=balancing_variables,
        protected_attribute=protected_attribute,
        num_loss_function_samples=num_loss_function_samples
    )
    for index in tqdm(list(range(len(X_train))), desc='Computing MMD for random model')
]

full_distances = [
    compute_distance_given_observation(
        X_train.iloc[index],
        data_df_with_predictions=full_model_df,
        outcome=outcome,
        balancing_variables=balancing_variables,
        protected_attribute=protected_attribute,
        num_loss_function_samples=num_loss_function_samples
    )
    for index in tqdm(list(range(len(X_train))), desc='Computing MMD for full model')
]

unaware_distances = [
    compute_distance_given_observation(
        X_train.iloc[index],
        data_df_with_predictions=unaware_model_df,
        outcome=outcome,
        balancing_variables=balancing_variables,
        protected_attribute=protected_attribute,
        num_loss_function_samples=num_loss_function_samples
    )
    for index in tqdm(list(range(len(X_train))), desc='Computing MMD for unaware model')
]

fair_distances = [
    compute_distance_given_observation(
        X_train.iloc[index],
        data_df_with_predictions=fair_model_df,
        outcome=outcome,
        balancing_variables=balancing_variables,
        protected_attribute=protected_attribute,
        num_loss_function_samples=num_loss_function_samples
    )
    for index in tqdm(list(range(len(X_train))), desc='Computing MMD for fair model')
]

distances_df = pd.DataFrame({
    'Random':random_distances,
    'Full': full_distances,
    'Unaware': unaware_distances,
    'ICF Fair': fair_distances
})

print('Mean MMD distances:')
print(distances_df.mean(axis=0))

plt.figure()
sns.set(rc={'text.usetex' : True}, font_scale=2, context='paper')
sns.set_style('whitegrid')
sns.boxplot(distances_df.abs(), orient='h')
sns.despine(right = True)
plt.axvline(x=0, linestyle='dashed', color='r')
plt.xlabel('Absolute Energy Distance MMD')
plt.gca().xaxis.grid(True)
plt.tight_layout()
filename = f"results/law_{','.join(balancing_variables)},n={dataset_size},nS={num_backtracking_samples},nL={num_loss_function_samples}"
distances_df.to_csv(f'{filename}.csv')
print(f'Saved to: {filename}.csv')

for balancing_variables in [['LSAT', 'U_LSAT', 'U_GPA']]:
    random_distances = [
        compute_distance_given_observation(
            X_train.iloc[index],
            data_df_with_predictions=random_model_df,
            outcome=outcome,
            balancing_variables=balancing_variables,
            protected_attribute=protected_attribute,
            num_loss_function_samples=num_loss_function_samples
        )
        for index in tqdm(list(range(len(X_train))), desc='Computing MMD for random model')
    ]

    full_distances = [
        compute_distance_given_observation(
            X_train.iloc[index],
            data_df_with_predictions=full_model_df,
            outcome=outcome,
            balancing_variables=balancing_variables,
            protected_attribute=protected_attribute,
            num_loss_function_samples=num_loss_function_samples
        )
        for index in tqdm(list(range(len(X_train))), desc='Computing MMD for full model')
    ]

    unaware_distances = [
        compute_distance_given_observation(
            X_train.iloc[index],
            data_df_with_predictions=unaware_model_df,
            outcome=outcome,
            balancing_variables=balancing_variables,
            protected_attribute=protected_attribute,
            num_loss_function_samples=num_loss_function_samples
        )
        for index in tqdm(list(range(len(X_train))), desc='Computing MMD for unaware model')
    ]

    fair_distances = [
        compute_distance_given_observation(
            X_train.iloc[index],
            data_df_with_predictions=fair_model_df,
            outcome=outcome,
            balancing_variables=balancing_variables,
            protected_attribute=protected_attribute,
            num_loss_function_samples=num_loss_function_samples
        )
    for index in tqdm(list(range(len(X_train))), desc='Computing MMD for fair model')
    ]

    distances_df = pd.DataFrame({
        'Random':random_distances,
        'Full': full_distances,
        'Unaware': unaware_distances,
        'ICF Fair': fair_distances
    })

    print('Mean MMD distances:')
    print(distances_df.mean(axis=0))

    plt.figure()
    sns.set(rc={'text.usetex' : True}, font_scale=2, context='paper')
    sns.set_style('whitegrid')
    sns.boxplot(distances_df.abs(), orient='h')
    sns.despine(right = True)
    plt.axvline(x=0, linestyle='dashed', color='r')
    plt.xlabel('Absolute Energy Distance MMD')
    plt.gca().xaxis.grid(True)
    plt.tight_layout()
    filename = f"results/law_{','.join(balancing_variables)},n={dataset_size},nS={num_backtracking_samples},nL={num_loss_function_samples}"
    distances_df.to_csv(f'{filename}.csv')
    print(f'Saved to: {filename}.csv')

plt.show()
