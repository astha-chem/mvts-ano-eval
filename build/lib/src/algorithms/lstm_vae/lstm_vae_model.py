from __future__ import print_function
import numpy as np
import tensorflow as tf
import six
from timeit import default_timer as timer

"""Variational LSTM Autoencoder adapted from https://github.com/Danyleb/Variational-Lstm-Autoencoder"""

class LSTM_Var_Autoencoder(object):

    def __init__(self, intermediate_dim=None, z_dim=None, n_dim=None, kulback_coef=0.1,
                 stateful=False, model_id=0):
        """
        Args:
        intermediate_dim : LSTM cells dimension.
        z_dim : dimension of latent space.
        n_dim : dimension of input data.
        statefull : if true, keep cell state through batches.
        """
        
        if not intermediate_dim or not z_dim or not n_dim:
            raise ValueError("You should set intermediate_dim, z_dim"
                             "(latent space) dimension and your input"
                             "third dimension, n_dim."
                             " \n            ")

        tf.reset_default_graph()

        self.model_id = model_id
        self.z_dim = z_dim
        self.n_dim = n_dim
        self.intermediate_dim = intermediate_dim
        self.stateful = stateful
        self.input = tf.placeholder(tf.float32, shape=[None, None, self.n_dim])
        self.batch_size = tf.placeholder(tf.int64)
        self.kulback_coef = kulback_coef
        dataset = tf.data.Dataset.from_tensor_slices(self.input).repeat() \
            .batch(self.batch_size)
        self.batch_ = tf.placeholder(tf.int32, shape=[])
        self.ite = dataset.make_initializable_iterator()
        self.x = self.ite.get_next()
        self.repeat = tf.placeholder(tf.int32)

        def gauss_sampling(mean, sigma):
            with tf.name_scope("sample_gaussian"):
                eps = tf.random_normal(tf.shape(sigma), 0, 1, dtype=tf.float32)
                z = tf.add(mean, tf.exp(0.5*sigma) * eps)
                return z

        def get_state_variables(batch_size, cell):
            # For each layer, get the initial state and make a variable out of it
            # to enable updating its value.
            state_variables = []
            for state_c, state_h in cell.zero_state(batch_size, tf.float32):
                state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                    (state_c), (state_h)))
            # Return as a tuple, so that it can be fed to dynamic_rnn as an initial
            # state
            return tuple(state_variables)

        # Add an operation to update the train states with the last state
        # tensors
        def get_state_update_op(state_variables, new_states):
            update_ops = []
            for state_variable, new_state in zip(state_variables, new_states):
                update_ops.extend([state_variable[0] == new_state[0],
                                   state_variable[1] == new_state[1]])
            return tf.tuple(update_ops)

        # Return an operation to set each variable in a list of LSTMStateTuples
        # to zero
        def get_state_reset_op(state_variables, cell, batch_size):
            zero_states = cell.zero_state(batch_size, tf.float32)
            return get_state_update_op(state_variables, zero_states)

        weights = {
            'z_mean': tf.get_variable(
                "z_mean",
                shape=[
                    self.intermediate_dim,
                    self.z_dim],
                initializer=tf.contrib.layers.xavier_initializer()),
            'log_sigma': tf.get_variable(
                "log_sigma",
                shape=[
                    self.intermediate_dim,
                    self.z_dim],
                initializer=tf.zeros_initializer())}
        biases = {
            'z_mean_b': tf.get_variable("b_mean", shape=[self.z_dim],
                                        initializer=tf.zeros_initializer()),
            'z_std_b': tf.get_variable("b_log_sigma", shape=[self.z_dim],
                                       initializer=tf.zeros_initializer())
        }

        with tf.variable_scope("encoder"):
            with tf.variable_scope("LSTM_encoder"):
                lstm_layer = tf.nn.rnn_cell.LSTMCell(
                    self.intermediate_dim,
                    forget_bias=1,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    activation=tf.nn.relu)

        if self.stateful:
            self.batch_ = tf.placeholder(tf.int32, shape=[])
            layer = tf.nn.rnn_cell.MultiRNNCell([lstm_layer])
            states = get_state_variables(self.batch_, layer)
            outputs, new_states = tf.nn.dynamic_rnn(
                layer, self.x, initial_state=states, dtype=tf.float32)
            self.update_op = get_state_update_op(states, new_states)
            self.reset_state_op = get_state_reset_op(
                states, lstm_layer, self.batch_)
        else:
            outputs, _ = tf.nn.dynamic_rnn(lstm_layer, self.x, dtype="float32")

        # For each layer, get the initial state. states will be a tuple of
        # LSTMStateTuples.
        self.z_mean = tf.add(tf.matmul(
            outputs[:, -1, :], weights['z_mean']), biases['z_mean_b'])
        self.z_sigma = tf.nn.softplus(tf.add(tf.matmul(
            outputs[:, -1, :], weights['log_sigma']), biases['z_std_b']))
        self.z = gauss_sampling(self.z_mean, self.z_sigma)

        # from [batch_size,z_dim] to [batch_size, TIMESTEPS, z_dim]
        repeated_z = tf.keras.layers.RepeatVector(
            self.repeat, dtype="float32")(self.z)

        with tf.variable_scope("decoder"):
            if self.stateful:
                with tf.variable_scope('lstm_decoder_stateful'):
                    rnn_layers_ = [
                        tf.nn.rnn_cell.LSTMCell(
                            size,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            forget_bias=1) for size in [
                            self.intermediate_dim,
                            n_dim]]
                    multi_rnn_cell_ = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_)
                    states_ = get_state_variables(self.batch_, multi_rnn_cell_)
                self.x_reconstr_mean, new_states_ = tf.nn.dynamic_rnn(
                    cell=multi_rnn_cell_, inputs=repeated_z, initial_state=states_, dtype=tf.float32)
                self.update_op_ = get_state_update_op(states_, new_states_)
                self.reset_state_op_ = get_state_reset_op(
                    states_, multi_rnn_cell_, self.batch_)
            else:
                with tf.variable_scope('lstm_decoder_stateless'):
                    rnn_layers = [
                        tf.nn.rnn_cell.LSTMCell(
                            size,
                            initializer=tf.contrib.layers.xavier_initializer(),
                            forget_bias=1) for size in [
                            self.intermediate_dim,
                            n_dim]]
                    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
                self.x_reconstr_mean, _ = tf.nn.dynamic_rnn(
                    cell=multi_rnn_cell, inputs=repeated_z, dtype=tf.float32)

    def _create_loss_optimizer(self, opt, **param):
        with tf.name_scope("MSE"):
            reconstr_loss = tf.reduce_sum(
                tf.losses.mean_squared_error(
                    self.x, self.x_reconstr_mean))
        with tf.name_scope("KL_divergence"):
            latent_loss = - 0.5 * tf.reduce_sum(1 + self.z_sigma
                                               - self.z_mean**2
                                               - tf.exp(self.z_sigma), 1)
            self._cost = tf.reduce_mean(reconstr_loss + self.kulback_coef*latent_loss)
        # apply gradient clipping
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), 10)
        self.train_op = opt(**param).apply_gradients(zip(grads, tvars))

    def fit(
            self,
            X_train,
            X_val,
            learning_rate=0.001,
            batch_size=100,
            num_epochs=200,
            opt=tf.train.GradientDescentOptimizer,
            REG_LAMBDA=0,
            grad_clip_norm=10,
            optimizer_params=None,
            verbose=True,
            patience=10):      
        
        if len(np.shape(X_train)) != 3:
            raise ValueError(
                'Input must be a 3-D array. I could reshape it for you, but I am too lazy.'
                ' \n            Use input.reshape(-1,timesteps,1).')
        if optimizer_params is None:
            optimizer_params = {}
            optimizer_params['learning_rate'] = learning_rate
        else:
            optimizer_params = dict(six.iteritems(optimizer_params))

        self._create_loss_optimizer(opt, **optimizer_params)
        lstm_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope='LSTM_encoder')
        self._cost += REG_LAMBDA * tf.reduce_mean(tf.nn.l2_loss(lstm_var))

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        saver = tf.train.Saver()
        
        self.sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(
            self.ite.initializer,
            feed_dict={
                self.input: X_train,
                self.batch_size: batch_size})
        batches_train_per_epoch = int(np.ceil(len(X_train) / batch_size))
        batches_val_per_epoch = int(np.ceil(len(X_val) / batch_size))
        num_samps_train_per_batch = [batch_size]*(len(X_train)//batch_size)
        if (len(X_train)%batch_size>0):
            num_samps_train_per_batch.append(len(X_train)%batch_size)
        num_samps_val_per_batch = [batch_size]*(len(X_val)//batch_size)
        if (len(X_val)%batch_size>0):
            num_samps_val_per_batch.append(len(X_val)%batch_size)

        print("\n")
        print("Training...")
        print("\n")
        start = timer()

        train_loss_by_epoch = []
        epoch_wo_improv = 0
        val_loss_by_epoch = []
        save_path = "/tmp/vaemodel"+str(self.model_id)+".ckpt"
        for epoch in range(num_epochs):
            if epoch_wo_improv < patience:

                train_loss = []
                for train_step in range(batches_train_per_epoch):
                    if self.stateful:
                        loss, _, s, _ = self.sess.run([self._cost, self.train_op, self.update_op, self.update_op_],
                                                      feed_dict={self.repeat: np.shape(X_train)[1], self.batch_: batch_size})
                    else:
                        loss, _ = self.sess.run([self._cost, self.train_op], feed_dict={
                                                self.repeat: np.shape(X_train)[1]})
                    train_loss.append(loss*num_samps_train_per_batch[train_step])
                train_loss_by_epoch.append(np.sum(train_loss)/len(X_train))

                # validation loss
                val_loss = []
                for val_step in range(batches_val_per_epoch):
                    if self.stateful:
                        loss = self.sess.run([self._cost],
                            feed_dict={self.repeat: np.shape(X_val)[1], self.batch_: batch_size})
                    else:
                        loss = self.sess.run([self._cost], 
                            feed_dict={self.repeat: np.shape(X_val)[1]})
                    val_loss.append(loss[0]*num_samps_val_per_batch[val_step])
                val_loss_avg = np.sum(val_loss)/len(X_val)
                val_loss_by_epoch.append(val_loss_avg)

                best_val_loss_epoch = np.argmin(val_loss_by_epoch)
                if best_val_loss_epoch == epoch:
                    # any time a new best is encountered, the best_params will get replaced
                    saver.save(self.sess, save_path)
                    best_val_loss = val_loss_avg
                # Check for early stopping by counting the number of epochs since val loss improved
                if epoch > 0 and val_loss_by_epoch[-1] >= val_loss_by_epoch[-2]:
                    epoch_wo_improv += 1
                else:
                    epoch_wo_improv = 0
                # update and print training error per patch
                if train_step == (batches_train_per_epoch - 1):
                    mean_loss = np.mean(train_loss)

                    if self.stateful:  # reset cell & hidden states between epochs
                        self.sess.run([self.reset_state_op],
                                      feed_dict={self.batch_: batch_size})
                        self.sess.run([self.reset_state_op_],
                                      feed_dict={self.batch_: batch_size})
                if epoch % 10 == 0 & verbose:
                    print(
                        "Epoch {:^6} Loss {:0.5f}"  .format(
                            epoch + 1, mean_loss))

            else:
                saver.restore(self.sess, save_path)
                # print("Model restored.")
                break

        end = timer()
        print("\n")
        print("Training time {:0.2f} minutes".format((end - start) / (60)))
        return train_loss_by_epoch, val_loss_by_epoch, best_val_loss

    def reconstruct(self, X, get_error=False):
        self.sess.run(
            self.ite.initializer,
            feed_dict={
                self.input: X,
                self.batch_size: np.shape(X)[0]})
        if self.stateful:
            _, _ = self.sess.run([self.reset_state_op, self.reset_state_op_], feed_dict={
                                 self.batch_: np.shape(X)[0]})
            x_rec, _, _ = self.sess.run([self.x_reconstr_mean, self.update_op, self.update_op_], feed_dict={
                                        self.batch_: np.shape(X)[0], self.repeat: np.shape(X)[1]})
        else:
            x_rec = self.sess.run(self.x_reconstr_mean,
                                  feed_dict={self.repeat: np.shape(X)[1]})
        if get_error:
            squared_error = (x_rec - X)**2
            return x_rec, squared_error
        else:
            return x_rec

    def reduce(self, X):
        self.sess.run(
            self.ite.initializer,
            feed_dict={
                self.input: X,
                self.batch_size: np.shape(X)[0]})
        if self.stateful:
            _ = self.sess.run([self.reset_state_op], feed_dict={
                              self.batch_: np.shape(X)[0]})
            x, _ = self.sess.run([self.z, self.update_op], feed_dict={
                                 self.batch_: np.shape(X)[0], self.repeat: np.shape(X)[1]})
        else:
            x = self.sess.run(self.z)
        return x
