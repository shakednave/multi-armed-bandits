import pandas as pd
import numpy as np
from cmab_classes import ContextualBandit, encode_action, prepare_for_vw

params = {
        'num_of_rows': 1000,
        'prob_a': 0.5,
        'prob_b': 0.3,
        'prob_c': 0.2,
        'reward_a': 0.6,
        'reward_b': 0.8,
        'reward_c': 0.3
    }


def generate_data(**kwargs):
    """
    generates data for the course of the simulation.
    :param num_of_rows:
    :param prob_a:
    :param prob_b:
    :param prob_c:
    :return:
    """
    df = pd.DataFrame(columns=['x', 'action', 'reward'])
    x = {'a':1, 'b':2, 'c':3}
    for i in range(params['num_of_rows']):
        current_action = np.random.choice(['a', 'b', 'c'], p=[params['prob_a'], params['prob_b'], params['prob_c']])
        x_local = x[current_action]
        reward = params[f'reward_{current_action}']
        df.loc[i] = [x_local, current_action, reward]
    return df


if __name__ == '__main__':
    permute = {
                "off_policy_eval": "dr",
                "explore": True,
                "method": "cover"
    }
    n_arms = int((len(params) -1) / 2)
    df = generate_data(**params)
    df, action_mappings = encode_action(df)
    df, df_parsed = prepare_for_vw(df)
    cb = ContextualBandit(n_arms=n_arms)
    summary_metrics, base_def = cb.learn(df_parsed, **permute)
    cb.plot_learning()
    test = df.x.to_frame()
    test, test_vw = prepare_for_vw(test, is_train=False)
    preds = cb.predict(test_vw)
    cb.plot_arm_selection(preds, action_mappings)