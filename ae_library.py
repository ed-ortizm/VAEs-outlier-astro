import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
################################################################################
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
###############################################################################
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
# https://github.com/tensorflow/tensorflow/issues/47311
# The main issue here is that you are using a custom loss callback that takes an
# argument advantage (from your data generator, most likely numpy arrays).
# In Tensorflow 2 eager execution, the advantage argument will be numpy,
# whereas y_true, y_pred are symbolic.
#
# The way to solve this is to turn off eager execution
###############################################################################
class VariationalAE:
    """ VAE for outlier detection using tf.keras """
    ############################################################################
    def __init__(self,
        input_dimensions:'int',
        encoder_units:'list', latent_dimensions:'int', decoder_units:'list',
        batch_size:'int', epochs:'int', learning_rate:'float')->'None':
        """
        Creates a variational auto encoder

        INPUTS
            input_dimensions:
            encoder_units: python list containing the number of units in
                each layer of the encoder
            latent_dimensions: number of  dimensions for the latent
                representation
            decoder_units: python list containing the number of units in
                each layer of the decoder
            batch_size: number of batches for the training set
            epochs: maximum number of epochs to train the algorithm
            learning_rate: value for the learning rate
        """

        self.input_dimensions = input_dimensions

        self.input_layer = Input(
            shape=(self.input_dimensions,),
            name='input_layer')

        self.encoder_units = encoder_units
        self.latent_dimensions = latent_dimensions
        self.decoder_units = decoder_units

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.encoder = None
        self.z = None
        self.decoder = self.build_decoder()

        # self.z_mean = None
        # self.z_log_sigma = None

        self.vae = self.build_vae()
    ############################################################################
    def build_vae(self)->'keras.model':
        """
        Builds and returns a compiled variational auto encoder using keras API
        """

        # get z_mean and z_log_sigma keras.tensors
        self.encoder = self.build_encoder()
        z_mean, z_log_sigma, z  = self.encoder(self.input_layer)
        ########################################################################
        self.decoder = self.build_decoder()
        ########################################################################
        vae_output = self.decoder(z)

        vae = Model(self.input_layer, vae_output,
            name='variational_auto_encoder')

        adam_optimizer = Adam(learning_rate=self.learning_rate)
        ########################################################################
        loss = self.standard_loss(z_mean, z_log_sigma)
        # loss = self.standard_loss()
        vae.compile(loss=loss , optimizer=adam_optimizer,
            metrics=['accuracy'])

        return vae
    ############################################################################
    # def standard_loss(self, y_true, y_pred)->'keras.custom_loss':
    def standard_loss(self, z_mean, z_log_sigma)->'keras.custom_loss':
        """
        Standard loss function for the variational auto encoder, that is,
        mean squared error for de reconstruction loss and the KL divergence.

        INPUTS
            y_true : input datapoint
            y_pred : predicted value by the network

        OUTPUT
            loss function with the keras format that is compatible with the
            .compile API
        """

        def loss(y_true, y_pred):

            mse = K.sum(K.square(y_true - y_pred), axis=-1)
            kl = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
            kl = (-0.5) * K.sum(kl, axis=-1)

            return mse + kl

        return loss
    ############################################################################
    def build_encoder(self)-> 'keras.model':
        """
        Builds and returns a compiled variational encoder using keras API
        """
        X = self.input_layer
        standard_deviation = np.sqrt(2. / self.input_dimensions)

        ########################################################################
        initial_weights = None
        for idx, units in enumerate(self.encoder_units):

            initial_weights = tf.keras.initializers.RandomNormal(
                mean=0.,
                stddev=standard_deviation)

            layer = Dense(units, activation='relu',
                kernel_initializer=initial_weights,
                name=f'encoder_{idx+1}')(X)

            X = layer
            standard_deviation = np.sqrt(2. / units)
            ####################################################################
        z_mean = Dense(self.latent_dimensions, name='z_mean',
                kernel_initializer=initial_weights)(X)

        z_log_sigma = Dense(self.latent_dimensions,
                name='z_log_sigma',
                kernel_initializer=initial_weights)(X)

        z = self.sampling(z_mean, z_log_sigma)
        ########################################################################
        encoder = Model(self.input_layer, [z_mean, z_log_sigma, z],
            name='variational_encoder')

        # encoder = Model(self.input_layer, X, name='variational_encoder')

        return encoder
    ############################################################################
    def sampling(self, z_mean, z_log_sigma):

        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.)
        z = z_mean + K.exp(z_log_sigma / 2) * epsilon

        return z
    ############################################################################
    def build_decoder(self)->'keras.model':
        """
        Builds and returns a compiled decoder using keras API
        """
        # An imput layer rather than self.z tensor for backpropagation
        decoder_input = Input(shape=(self.latent_dimensions,),
            name='z_sampling')

        standard_deviation = np.sqrt(2. / self.latent_dimensions)

        X = decoder_input
        initial_weights = None
        ########################################################################
        for idx, units in enumerate(self.decoder_units):

            initial_weights = tf.keras.initializers.RandomNormal(
                mean=0., stddev=standard_deviation)

            layer = Dense(units, activation='relu',
                kernel_initializer=initial_weights,
                name=f'decoder_{idx+1}')(X)

            X = layer
            standard_deviation = np.sqrt(2./units)

            # if units == self.decoder_units[-1]:
        ########################################################################
        decoder_output = Dense(self.input_dimensions,
            kernel_initializer=initial_weights,
            name='decoder_output')(X)

        decoder = Model(decoder_input, decoder_output,
            name='variational_decoder')

        return decoder
    ############################################################################
    def custom_loss(self):
        pass
    ############################################################################
    def fit(self, spectra:'2D np.array')-> 'None':

        self.vae.fit(x=spectra, y=spectra, epochs=self.epochs,
            batch_size=self.batch_size, verbose=2)
    ############################################################################
    def predict(self, spectra:'2D np.array')-> '2D np.array':

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        return self.vae.predict(spectra)
    ############################################################################
    def encode(self, spectra:'2D np.array')-> '2D np.array':

        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
        return self.encoder(spectra)
    ############################################################################
    def decode(self, coding:'2D np.array')->'2D np.aray':

        if coding.ndim==1:
            coding = coding.reshape(1,-1)

        return self.decoder(coding)
    ############################################################################
    def save_vae(self, fpath:'str'):

        self.vae.save(f'{fpath}')
    # ############################################################################
    def save_encoder(self, fpath:'str'):

        self.encoder.save(f'{fpath}')
    ############################################################################
    def save_decoder(self, fpath:'str'):

        self.decoder.save(f'{fpath}')
    ############################################################################
    def plot_model(self):

        plot_model(self.vae, to_file='DenseVAE.png', show_shapes='True')
        plot_model(self.encoder, to_file='DenseEncoder.png', show_shapes='True')
        plot_model(self.decoder, to_file='DenseDecoder.png', show_shapes='True')
    # ############################################################################
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.vae.summary()
###############################################################################
