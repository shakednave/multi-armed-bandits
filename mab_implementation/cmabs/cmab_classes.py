import re
import pandas as pd
import vowpalwabbit
import json
import os
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections
from datetime import datetime
from vowpalwabbit.dftovw import DFtoVW,  Feature, ContextualbanditLabel

logging.basicConfig(format='%(asctime)s     %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    stream=sys.stdout,
                    level='INFO')

def encode_action(df: pd.DataFrame):
    """
    encode action name to action_id (str --> int)
    """

    action_mappings = df.action.value_counts().to_dict()
    # encode actions from 1 to k
    action_mappings = {k: i+1 for (i, k) in enumerate(action_mappings.keys())}
    # apply encoder
    df.action = df.action.map(lambda x: action_mappings.get(x))

    return df, action_mappings

def prepare_for_vw(df: pd.DataFrame, is_train: bool = True):
    """
    converts data from pandas format to the needed VW format.
    This means turning rewards to costs, adding probability to the actions, renaming columns,
    removing redundant cols, syntax parsing, etc.
    """

    if not is_train:
        try:
            df.drop(columns=['action', 'probability', 'cost'], inplace=True)
        except:
            pass
    if is_train:
        features = [col for col in df.columns if col not in ('action', 'probability', 'cost')]
        # vw works with costs not rewards
        df['cost'] = df['reward'] * -1

        # probability of taken action is estimated by it's frequency
        probas = (df.action.value_counts() / df.shape[0]).to_dict()
        df.loc[:, 'probability'] = df.action.map(lambda x: probas.get(x))
        converter = DFtoVW(df=df,
                                 label=ContextualbanditLabel(action="action", cost="cost", probability="probability"),
                                 features=[Feature(col) for col in features])

        df_parsed = converter.convert_df()
    else:
        features = ['x']
        converter = DFtoVW(df=df,
                                features=[Feature(col) for col in features])
        df_parsed = converter.convert_df()
    return df, df_parsed

class ContextualBandit():
    """
    https://arxiv.org/pdf/1802.04064.pdf
    """

    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.df_iter = None
        pass

    def learn(self, train_vw: list, explore: bool,
              method: str = 'epsilon_greedy',
              off_policy_eval: str = 'dr', bag: int = 5, cover: int = 3,
              epsilon: float = 0.2, **kwargs):
        assert off_policy_eval in ('dr', 'ips', 'mtr')

        base_def = None
        if not explore:
            # no exploration, obtain best policy using ips/dr/mtr alone, this is a deterministic approach
            base_def = "--cb {} --cb_type {}".format(self.n_arms, off_policy_eval)

        else:
            # exploration algorithms add a probability distribution to each action in the policy.
            # It adds some randomness
            assert method in ('epsilon_greedy', 'bag', 'cover')
            if method == 'bag':
                base_def = "--cb_explore {} --cb_type {} --bag {}".format(self.n_arms, off_policy_eval, bag)
            if method == 'cover':
                base_def = "--cb_explore {} --cb_type {} --cover {}".format(self.n_arms, off_policy_eval, cover)
            if method == 'epsilon_greedy':
                base_def = "--cb_explore {} --cb_type {} --epsilon {}".format(self.n_arms, off_policy_eval, epsilon)


        logging.info(f'Training with: {base_def}')

        vw = vowpalwabbit.Workspace(base_def, enable_logging=True, P=1)
        for i in range(len(train_vw)):
            vw.learn(train_vw[i])
        vw.finish()

        log_parser = VWLogParser(vw.get_log())
        params, df_iter, summary_metrics = log_parser.parse()

        self.df_iter = df_iter

        now = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
        model_stats = {'method': method, 'off_policy_eval': off_policy_eval,
                       'time': now}  # , 'average_loss': average_loss}
        logging.info(model_stats)
        self.model = vw
        return summary_metrics, base_def

    def predict(self, test_vw: list):
        predictions = []
        for i in range(len(test_vw)):
            pred = self.model.predict(test_vw[i])
            predictions.append(pred)
        return predictions

    def plot_learning(self):
        df_iter = self.df_iter
        sns.lineplot(x=df_iter['example_counter'], y=df_iter['average_loss'])
        plt.xlabel("Iteration")
        plt.ylabel("Average Loss")
        plt.title("Learning Curve")
        plt.show()
        plt.close()

        sns.scatterplot(x=df_iter['example_counter'], y=df_iter['since_last'])
        plt.xlabel("Iteration")
        plt.ylabel("Current Loss")
        plt.title("Loss Dispersion")
        plt.show()
        plt.close()

    def plot_arm_selection(self, preds, mappings):
        """
        plots the distribution of arm selection
        """
        # inverse  action_name -> code to code-> action_name
        inv_map = {v: k for k, v in mappings.items()}

        preds = np.array(preds)
        if preds.ndim == 1: # deterministic arms
            selected_actions = preds
        else: # stochastic arms
            # make sure all rows sum to 1 (due to numeric instability)
            epsilon = 10 ** -6
            np.nan_to_num(preds, nan=epsilon, posinf=epsilon, neginf=epsilon)
            preds /= preds.sum(axis=1, keepdims=1)
            # we are calculating rp.random.choice(p=predictions) but doing it in a vectorized way in 2D
            _, n_arms = preds.shape

            def f(row):
                return np.random.choice(n_arms, p=row)
            selected_actions = np.apply_along_axis(f, axis=1, arr=preds)
            selected_actions += 1  # since the returned indexes start with zero, and vw actions start with 1

        counters = collections.Counter(selected_actions)
        n_arms = len(mappings)
        for arm in range(1, n_arms + 1):
            if arm not in counters:
                counters[arm] = 0
        counters = {k: v / len(selected_actions) for k, v in counters.items()}

        x_s, y_s = [inv_map[x] for x in counters.keys()], list(counters.values())
        sns.barplot(x=x_s, y=y_s)
        plt.xlabel("Arm")
        plt.ylabel("Proportion")
        plt.title("Arm Selection On Test Set")
        plt.show()
        plt.close()

class VWLogParser:
    """Parser for Vowpal Wabbit output log"""

    def __init__(self, file_path_or_list):
        """The file name or list of lines to parse"""
        if isinstance(file_path_or_list, (list, str)):
            self.file_path_or_list = file_path_or_list
        else:
            raise TypeError("Argument `fname` should be a str (for file path) or a list of log lines")

    def parse(self):
        """Parse the output from `vw` command, return dataframe/dictionnaries with the associated data."""
        # Init containers
        self.table_lst = []
        self.params = {}
        self.metrics = {}

        self.inside_table = False
        self.after_table = False

        if isinstance(self.file_path_or_list, list):
            for row in self.file_path_or_list:
                self._parse_vw_row(row)
        else:
            with open(self.file_path_or_list, "r") as f:
                for row in f:
                    self._parse_vw_row(row)

        self.df = self._make_output_df(self.table_lst)

        return self.params, self.df, self.metrics

    def _cast_string(self, s):
        """Cast to float or int if possible"""
        try:
            out = float(s)
        except ValueError:
            out = s
        else:
            if out.is_integer():
                out = int(out)

        return out

    def _make_output_df(self, lst):
        """Make dataframe from the list"""
        # Make columns from first and second elements of the list
        columns = [f"{first_row}_{second_row}" for (first_row, second_row) in zip(*lst[:2])]

        df = pd.DataFrame(data=lst[2:], columns=columns)
        df = df.iloc[:-1, :]
        # Cast cols to appropriate types
        int_cols = ["example_counter", "current_features"]
        for col in int_cols:
            df[col] = df[col].astype(int)

        float_cols = df.columns.drop(int_cols)
        for col in float_cols:
            try :
                df[col] = df[col].astype(float)
            except ValueError:
                df[col] = df[col].astype(str)

        return df

    def _parse_vw_row(self, row):
        if len(row) == 0:
            return
        """Parse row and add parsed elements to instance attributes params, metrics and table_lst"""
        if "=" in row:
            param_name, value = [element.strip() for element in row.split("=", maxsplit=1)]
            if self.after_table:
                self.metrics[param_name] = self._cast_string(value)
            else:
                self.params[param_name] = self._cast_string(value)
        elif ":" in row and not (row[0].isdigit() or row[0] == '-'):

            param_name, value = [element.strip() for element in row.split(":", maxsplit=1)]
            self.params[param_name] = self._cast_string(value)

        elif row[0].isdigit() or row[0] == '-':
            parsed_row = ' '.join(row.split())
            self.table_lst += [parsed_row.split(' ')]

        elif not self.after_table:
            if re.match("average\s+since", row):
                self.inside_table = True
            if row == "\n":
                self.inside_table = False
                self.after_table = True
            if self.inside_table:
                self.table_lst += [row.split()]
