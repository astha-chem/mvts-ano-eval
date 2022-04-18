from tfsnippet.scaffold import CheckpointSaver
from tfsnippet.utils import get_variables_as_dict, Config
from src.algorithms.omni_anomaly.model import OmniAnomaly
from src.algorithms.omni_anomaly.prediction import Predictor
from src.algorithms.omni_anomaly.training import Trainer
import pandas as pd
from src.algorithms.algorithm_utils import *
"""OmniAnomaly adapted from https://github.com/NetManAIOps/OmniAnomaly (MIT License)"""

class OmniAnoAlgo(Algorithm, TensorflowUtils):
    def __init__(self, name: str='OmniAnoAlgo', num_epochs: int=30, batch_size: int=50, sequence_length: int=100,
                 seed: int=0, gpu: int=None, details=False, out_dir=None,
                 train_val_percentage: float=0.25, z_dim: int=3, stride: int=1, rnn_num_hidden: int=500,
                 dense_dim: int=500, nf_layers: int=20):
        """

        :param name:
        :param num_epochs:
        :param batch_size:
        :param sequence_length:
        :param seed: is initialized in tensorflowutils and algorithm
        :param gpu:
        :param details:
        :param out_dir:
        :param train_val_percentage:
        :param z_dim:
        """
        Algorithm.__init__(self, __name__, name, seed, details=details, out_dir=out_dir)
        TensorflowUtils.__init__(self, seed, gpu)
        self.train_val_pc = train_val_percentage
        self.batch_size = batch_size

        self.config = ExpConfig()
        self.config.max_epoch = num_epochs
        self.config.window_length = sequence_length

        self.config.batch_size = batch_size
        self.config.test_batch_size = batch_size
        self.config.z_dim = z_dim
        self.config.stride = stride
        self.config.rnn_num_hidden = rnn_num_hidden
        self.config.dense_dim = dense_dim
        self.config.nf_layers = nf_layers
        self.model = None
        self.trainer = None
        self.predictor = None
        self.normal_scores = None
        self.tf_session = None
        self.best_val_loss = None

    def fit(self, X: pd.DataFrame):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        self.config.x_dim = data.shape[1]
        tf.reset_default_graph()
        with tf.variable_scope('model') as model_vs:
            self.model = OmniAnomaly(config=self.config, name="model")
            # construct the trainer
            self.trainer = Trainer(model=self.model,
                                   model_vs=model_vs,
                                   max_epoch=self.config.max_epoch,
                                   batch_size=self.config.batch_size,
                                   valid_batch_size=self.config.test_batch_size,
                                   initial_lr=self.config.initial_lr,
                                   lr_anneal_epochs=self.config.lr_anneal_epoch_freq,
                                   lr_anneal_factor=self.config.lr_anneal_factor,
                                   grad_clip_norm=self.config.gradient_clip_norm,
                                   valid_step_freq=self.config.valid_step_freq,
                                   stride=self.config.stride)

            # construct the predictor
            self.predictor = Predictor(self.model, batch_size=self.config.batch_size, n_z=self.config.test_n_z,
                                  last_point_only=True)
            self.tf_session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            with self.tf_session.as_default():
                if self.config.restore_dir is not None:
                    # Restore variables from `save_dir`.
                    saver = CheckpointSaver(get_variables_as_dict(model_vs), self.config.restore_dir)
                    saver.restore()

                metrics_dict = self.trainer.fit(data, valid_portion=self.train_val_pc)
                self.best_val_loss = metrics_dict['best_valid_loss']
                train_score, train_z, train_pred_speed = self.predictor.get_score(data)
                self.normal_scores = -np.sum(train_score, axis=1)

    def get_val_loss(self):
        return self.best_val_loss

    def predict(self, X: pd.DataFrame, starts=np.array([])):
        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        with self.tf_session.as_default():
            score_tc, test_z, pred_speed = self.predictor.get_score(data)
            # Add padding for first window_length time points
            padding = np.max(score_tc) * np.ones((self.config.window_length - 1, score_tc.shape[1]))
            score_tc = -np.concatenate((padding, score_tc), axis=0)
            score_t = np.sum(score_tc, axis=1)
        predictions_dic = {'score_t': score_t,
                           'score_tc': score_tc,
                           'error_t': None,
                           'error_tc': None,
                           'recons_tc': None,
                           }
        return predictions_dic

    def close_session(self):
        self.tf_session.close()

class ExpConfig(Config):
    # dataset configuration
    dataset = None
    x_dim = None
    stride = 1
    # model architecture configuration
    use_connected_z_q = True
    use_connected_z_p = True

    # model parameters
    z_dim = 3
    rnn_cell = 'GRU'  # 'GRU', 'LSTM' or 'Basic'
    rnn_num_hidden = 500
    window_length = 100
    dense_dim = 500
    posterior_flow_type = 'nf'  # 'nf' or None
    nf_layers = 20  # for nf
    max_epoch = 20
    train_start = 0
    max_train_size = None  # `None` means full train set
    batch_size = 50
    l2_reg = 0.0001
    initial_lr = 0.001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = 40
    lr_anneal_step_freq = None
    std_epsilon = 1e-4

    # evaluation parameters
    test_n_z = 1
    test_batch_size = 50
    test_start = 0
    max_test_size = None  # `None` means full test set

    # the range and step-size for score for searching best-f1
    # may vary for different dataset
    bf_search_min = -400.
    bf_search_max = 400.
    bf_search_step_size = 1.

    valid_step_freq = 100
    gradient_clip_norm = 10.

    early_stop = True  # whether to apply early stop method

    # pot parameters
    # recommend values for `level`:
    # SMAP: 0.07
    # MSL: 0.01
    # SMD group 1: 0.0050
    # SMD group 2: 0.0075
    # SMD group 3: 0.0001
    # level = 0.01

    # outputs config
    save_z = False  # whether to save sampled z in hidden space
    get_score_on_dim = True  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
    save_dir = 'model'
    restore_dir = None  # If not None, restore variables from this dir
    result_dir = 'result'  # Where to save the result file
    train_score_filename = 'train_score.pkl'
    test_score_filename = 'test_score.pkl'


def main():
    from src.datasets.skab import Skab
    seed = 0
    print("Running main")
    ds = Skab(seed=seed)
    x_train, y_train, x_test, y_test = ds.data()
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]
    algo = OmniAnoAlgo(num_epochs=4)
    algo.fit(x_train)
    results = algo.predict(x_test)
    print(results["score_tc"].shape)
    print(results["score_tc"][:10])



if __name__ == "__main__":

    main()
