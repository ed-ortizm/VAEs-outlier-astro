import os

################################################################################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

################################################################################
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Activation, BatchNormalization, ReLU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

################################################################################
tf.compat.v1.disable_eager_execution()
# this is necessary because I use custom loss function
################################################################################
class VAE:

    def __init__(self,
        input_dimensions:'int',
        encoder_units:'list',
        latent_dimensions:'int',
        decoder_units:'list',
        batch_size:'int',
        epochs:'int',
        learning_rate:'float',
        reconstruction_loss_weight:'float',
        output_activation:'str'='linear'
        )->'tf.keras.model':

        """
            PARAMETERS

            input_dimensions:

            encoder_units: python list containing the number of units
                in each layer of the encoder

            latent_dimensions: number of  dimensions for the latent
                representation

            decoder_units: python list containing the number of units
                in each layer of the decoder

            batch_size: number of batches for the training set

            epochs: maximum number of epochs to train the algorithm

            learning_rate: value for the learning rate

            output_activation:

            reconstruction_loss_weight: weighting factor for the
                reconstruction loss
        """

        self.input_dimensions = input_dimensions

        self.encoder_units = encoder_units
        self.latent_dimensions = latent_dimensions
        self.decoder_units = decoder_units

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.out_activation = output_activation

        self.reconstruction_loss_weight = reconstruction_loss_weight

        self.encoder = None
        self.decoder = None
        self.model = None

        self._build()
    ############################################################################
    def _build(self)->'':
        """
        Builds and returns a compiled variational auto encoder using
        keras API
        """

        self._build_encoder()
        self._build_decoder()
        # self._build_vae()
        # return Input(shape=self.latent_space_dim, name="decoder_input")
    ############################################################################
    def _build_decoder(self):
        pass
    ############################################################################
    def _build_encoder(self):

        encoder_input = Input(
            shape=self.input_dimensions,
            name="encoder_input"
        )

        encoder_block = self._encoder_block(encoder_input)

        latent_layer = self._latent_layer(encoder_block)

        self._model_input = encoder_input
        self.encoder = Model(encoder_input, latent_layer, name="encoder")

    ############################################################################
    def _encoder_block(self, encoder_input):

        x = encoder_input
        standard_deviation = np.sqrt(2. / self.input_dimensions)

        for layer_index, number_units in enumerate(self.encoder_units):

            x, standard_deviation = self._add_layer(
                    x,
                    layer_index,
                    number_units,
                    # initial_weights,
                    standard_deviation
                )

        return x
    ############################################################################
    def _add_layer(self,
        x:'',
        layer_index:'int',
        number_units:'int',
        # initial_weights,
        standard_deviation:''
        ):

        initial_weights = tf.keras.initializers.RandomNormal(
            mean=0.,
            stddev=standard_deviation
            )

        layer = Dense(
            number_units,
            kernel_initializer=initial_weights,
            name=f'encoder_layer_{layer_index + 1}'
            )

        x = layer(x)

        x = ReLU(name=f'relu_encoder_layer_{layer_index + 1}')(x)

        x = BatchNormalization(
            name=f'batch_normaliztionencoder_layer_{layer_index + 1}'
            )(x)

        standard_deviation = np.sqrt(2. / number_units)

        return x, standard_deviation
    ############################################################################
    def _latent_layer(self, x:''):

        self.mu = Dense(self.latent_dimensions, name="mu")(x)

        self.log_variance = Dense(self.latent_dimensions,
                                  name="log_variance")(x)


        ########################################################################
        def sample_normal_distribution(args):

            mu, log_variance = args
            epsilon = K.random_normal(
                                        shape=K.shape(self.mu),
                                        mean=0.,
                                        stddev=1.
                                      )

            point = mu + K.exp(log_variance / 2) * epsilon

            return point
        ########################################################################
        x = Lambda(sample_normal_distribution,
                    name='encoder_outputs')([self.mu, self.log_variance])

        return x
    ############################################################################
    ############################################################################

    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################
#         # get z_mean and z_log_sigma keras.tensors
#         self.encoder = self.build_encoder()
#         z_mean, z_log_sigma, z  = self.encoder(self.input_layer)
#         ########################################################################
#         self.decoder = self.build_decoder()
#         ########################################################################
#         vae_output = self.decoder(z)
#
#         vae = Model(self.input_layer, vae_output,
#             name='variational_auto_encoder')
#
#         adam_optimizer = Adam(learning_rate=self.learning_rate)
#         ########################################################################
#         loss = self.standard_loss(z_mean, z_log_sigma)
#         # loss = self.standard_loss()
#         vae.compile(loss=loss , optimizer=adam_optimizer,
#             metrics=['accuracy'])
#
#         return vae
#     ############################################################################
#     # y_target, y_predicted: keep consistency with keras API
#     # a loss function expects this two parameters
#     def _loss(self, y_target, y_predicted):
#         """
#         Standard loss function for the variational auto encoder, that is,
#         mean squared error for de reconstruction loss and the KL divergence.
#
#         INPUTS
#             y_true : input datapoint
#             y_pred : predicted value by the network
#
#         OUTPUT
#             loss function with the keras format that is compatible with the
#             .compile API
#         """
#
#         reconstruction_loss = self._reconstruction_loss(y_target, y_predicted)
#         kl_loss = self._kl_loss(y_target, y_predicted)
#
#         [loss] = [
#             self.reconstruction_loss_weight * reconstruction_loss + kl_loss
#         ]
#
#         return loss
#
#     ############################################################################
#     def _reconstruction_loss(self, y_target, y_predicted):
#
#         error = y_target - y_predicted
#
#         reconstruction_loss = K.mean(K.square(error), axis=1)
#
#         return reconstruction_loss
#
#     ############################################################################
#     def _kl_loss(self, y_target, y_predicted):
#
#         kl_loss = -0.5 * K.sum(1 +
#             self.log_variance - K.square(self.mu) - K.exp(self.log_variance),
#             axis=1
#         )
#
#         return kl_loss
#     ############################################################################
#     def build_encoder(self)-> 'keras.model':
#         """
#         Builds and returns a compiled variational encoder using keras API
#         """
#         X = self.input_layer
#         standard_deviation = np.sqrt(2. / self.input_dimensions)
#
#         ########################################################################
#         initial_weights = None
#         for idx, units in enumerate(self.encoder_units):
#
#             initial_weights = tf.keras.initializers.RandomNormal(
#                 mean=0.,
#                 stddev=standard_deviation)
#
#             layer = Dense(units,
#                 activation='relu',
#                 kernel_initializer=initial_weights,
#                 name=f'encoder_{idx+1}')(X)
#
#             X = layer
#             standard_deviation = np.sqrt(2. / units)
#             ####################################################################
#         z_mean = Dense(self.latent_dimensions,
#             name='z_mean',
#             kernel_initializer=initial_weights
#             )(X)
#
#         z_log_sigma = Dense(self.latent_dimensions,
#             name='z_log_sigma',
#             kernel_initializer=initial_weights
#             )(X)
#
#         z = self.sampling(z_mean, z_log_sigma)
#         ########################################################################
#         encoder = Model(self.input_layer, [z_mean, z_log_sigma, z],
#             name='variational_encoder')
#
#         return encoder
#     ############################################################################
#     def sampling(self, z_mean, z_log_sigma):
#
#         epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1.)
#         z = z_mean + K.exp(z_log_sigma / 2) * epsilon
#
#         return z
#     ############################################################################
#     def build_decoder(self)->'keras.model':
#         """
#         Builds and returns a compiled decoder using keras API
#         """
#         # An imput layer rather than self.z tensor for backpropagation
#         decoder_input = Input(shape=(self.latent_dimensions,),
#             name='z_sampling')
#
#         standard_deviation = np.sqrt(2. / self.latent_dimensions)
#
#         X = decoder_input
#         initial_weights = None
#         ########################################################################
#         for idx, units in enumerate(self.decoder_units):
#
#             initial_weights = tf.keras.initializers.RandomNormal(
#                 mean=0.,
#                 stddev=standard_deviation)
#
#             layer = Dense(units,
#                 activation='relu',
#                 kernel_initializer=initial_weights,
#                 name=f'decoder_{idx+1}'
#                 )(X)
#
#             X = layer
#             standard_deviation = np.sqrt(2./units)
#
#             # if units == self.decoder_units[-1]:
#         ########################################################################
#         decoder_output = Dense(self.input_dimensions,
#             activation=self.out_activation,
#             kernel_initializer=initial_weights,
#             name='decoder_output')(X)
#
#         decoder = Model(decoder_input, decoder_output,
#             name='variational_decoder')
#
#         return decoder
#     ############################################################################
#     def custom_loss(self):
#         pass
#     ############################################################################
#     def fit(self, spectra:'2D np.array')-> 'None':
#
#         self.vae.fit(x=spectra, y=spectra, epochs=self.epochs,
#             batch_size=self.batch_size, verbose=2)
#     ############################################################################
#     def predict(self, spectra:'2D np.array')-> '2D np.array':
#
#         if spectra.ndim == 1:
#             spectra = spectra.reshape(1, -1)
#
#         return self.vae.predict(spectra)
#     ############################################################################
#     def encode(self, spectra:'2D np.array')-> '2D np.array':
#
#         if spectra.ndim == 1:
#             spectra = spectra.reshape(1, -1)
#         return self.encoder(spectra)
#     ############################################################################
#     def decode(self, coding:'2D np.array')->'2D np.aray':
#
#         if coding.ndim==1:
#             coding = coding.reshape(1,-1)
#
#         return self.decoder(coding)
#     ############################################################################
#     def save_vae(self, fpath:'str'):
#
#         self.vae.save(f'{fpath}')
#     # ############################################################################
#     def save_encoder(self, fpath:'str'):
#
#         self.encoder.save(f'{fpath}')
#     ############################################################################
#     def save_decoder(self, fpath:'str'):
#
#         self.decoder.save(f'{fpath}')
#     ############################################################################
#     def plot_model(self):
#
#         plot_model(self.vae, to_file='DenseVAE.png', show_shapes='True')
#         plot_model(self.encoder, to_file='DenseEncoder.png', show_shapes='True')
#         plot_model(self.decoder, to_file='DenseDecoder.png', show_shapes='True')
#     # ############################################################################
#     def summary(self):
#         self.encoder.summary()
#         self.decoder.summary()
#         self.vae.summary()
# ###############################################################################
