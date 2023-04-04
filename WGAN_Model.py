
import time
import tensorflow as tf
import keras
from network import build_critic, build_generator
from sklearn.preprocessing import *
class WGAN():
    def __init__(self, n_features,latent_space=3,BATCH_SIZE=100,n_var = 2,use_bias=False, lr = 0.0001, activation1 = 'relu', activation2='tanh',stddev = 0.1):
        """
        Initialise the WGAN-GP model.
        """
        self.n_features = n_features
        self.n_var = n_var
        self.BATCH_SIZE = BATCH_SIZE
        self.latent_space = latent_space
        self.n_critic = 8
        self.stddev = stddev
        # building the components of the WGAN-GP
        self.generator = build_generator(self.latent_space, n_var,self.n_features,use_bias=use_bias,activation1=activation1,activation2=activation2)
        self.critic = build_critic(n_var,use_bias,self.n_features)
        self.wgan = keras.models.Sequential([self.generator, self.critic])

        # setting hyperparemeters of the WGAN-GP
        self.generator_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.critic_mean_loss = tf.keras.metrics.Mean(dtype=tf.float32)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)

        # saving the events of the generator and critic
        generator_log_dir = './content/generator'
        critic_log_dir = './content/critic'
        self.generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
        self.critic_summary_writer = tf.summary.create_file_writer(critic_log_dir)

        # for prediction function
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(1e-2)


    ############################################################################
    ############################################################################
    # preprocessing

    def preproc(self, training_data, feature_range = [0,1]):
        """
        Prepares the data for the WGAN-GP by splitting the data set
        into batches and normalizing it between -1 and 1 or 0 and 1.
        """

        scaler = MinMaxScaler(feature_range=(feature_range[0],feature_range[1]))
        X_train_scaled = scaler.fit_transform(training_data)
        train_dataset = X_train_scaled.reshape(-1, self.n_features).astype('float32')
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.shuffle(len(X_train_scaled))
        train_dataset = train_dataset.batch(self.BATCH_SIZE)

        return train_dataset, scaler


    ############################################################################
    ############################################################################
    # training

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def critic_loss(self, real_output,fake_output):
        return tf.reduce_mean(fake_output)-tf.reduce_mean(real_output)

    def gradient_penalty(self, real, fake):
        """
        WGAN-GP uses gradient penalty instead of the weight
        clipping to enforce the Lipschitz constraint.
        """

        alpha = tf.random.uniform(shape=[real.shape[0], self.n_features], minval=-1., maxval=1.)
        interpolated = real + alpha * (fake - real)

        with tf.GradientTape() as t:
            t.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        grad = t.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grad)) + 1e-12)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp


    @tf.function
    def train_G(self, batch):
        """
        The training routine for the generator
        """
        if batch.shape[0]==self.BATCH_SIZE:
            noise = tf.random.normal([self.BATCH_SIZE, self.latent_space], mean=0.0, stddev=self.stddev)
        else:
            noise = tf.random.normal([batch.shape[0]%self.BATCH_SIZE, self.latent_space], mean=0.0, stddev=self.stddev)

        with tf.GradientTape() as gen_tape:
            generated_data = self.generator(noise, training=True)
            fake_output = self.critic(generated_data, training=True)
            gen_loss = self.generator_loss(fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return gen_loss

    @tf.function
    def train_D(self, batch):
        """
        The training routine for the critic
        """
        if batch.shape[0]==self.BATCH_SIZE:
            noise = tf.random.normal([self.BATCH_SIZE, self.latent_space], mean=0.0, stddev=self.stddev)
        else:
            noise = tf.random.normal([batch.shape[0]%self.BATCH_SIZE, self.latent_space], mean=0.0, stddev=self.stddev)

        with tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)

            real_output = self.critic(batch, training=True)
            fake_output = self.critic(generated_data, training=True)

            disc_loss = self.critic_loss(real_output, fake_output)
            gp = self.gradient_penalty(batch, generated_data)
            disc_loss += gp*10.0

        gradients_of_critic = disc_tape.gradient(disc_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))

        return disc_loss
    # @tf.function
    def train(self, dataset, epochs):
        """
        Training the WGAN-GP
        """

        hist = []
        for epoch in range(epochs):
            tf.keras.backend.clear_session()
            start = time.time()
            print("Epoch {}/{}".format(epoch, epochs))

            for batch in dataset:

                for _ in range(self.n_critic):
                    self.train_D(batch)
                    disc_loss = self.train_D(batch)
                    self.critic_mean_loss(disc_loss)


                gen_loss = self.train_G(batch)
                self.generator_mean_loss(gen_loss)
                self.train_G(batch)

            with self.generator_summary_writer.as_default():
                tf.summary.scalar('generator_loss', self.generator_mean_loss.result(), step=epoch)

            with self.critic_summary_writer.as_default():
                tf.summary.scalar('critic_loss', self.critic_mean_loss.result(), step=epoch)

            hist.append([self.generator_mean_loss.result().numpy(), self.critic_mean_loss.result().numpy()])

            self.generator_mean_loss.reset_states()
            self.critic_mean_loss.reset_states()

            # outputting loss information
            print("critic: {:.6f}".format(hist[-1][1]), end=' - ')
            print("generator: {:.6f}".format(hist[-1][0]), end=' - ')
            print('{:.0f}s'.format( time.time()-start))


        return hist
    def load_models(self, name):
        self.generator = keras.models.load_model(name, compile=False)

    ############################################################################
    ############################################################################
    # prediction

    def mse_loss(self, inp, outp):
        """
        Calculates the MSE loss for the first value in the output (change for other values)
        """
        inp = tf.reshape(inp, [-1, self.n_features])
        outp = tf.reshape(outp, [-1, self.n_features])
        return self.mse(inp[:,0], outp[:,0])


    def opt_step(self, latent_values, real_coding):
        """
        Minimizes the loss between generated point and inputted point
        """
        with tf.GradientTape() as tape:
            tape.watch(latent_values)
            gen_output = self.generator(latent_values, training=False)
            loss = self.mse_loss(real_coding, gen_output)

        gradient = tape.gradient(loss, latent_values)
        self.optimizer.apply_gradients(zip([gradient], [latent_values]))

        return loss

    def optimize_coding(self, real_coding,limit=None,niter=100):
        """
        Optimizes the latent space values, limit contains the upper and lower limits of the latent space.
        """
        latent_values = tf.random.normal([len(real_coding), self.latent_space], mean=0.0, stddev=0.1)
        if limit==None:
            None
        else:
            for j in range(len(real_coding)):
                for i in range(latent_values.shape[1]):
                    if latent_values[j,i] > limit[0]:
                        latent_values[j,i].assign(limit[0])
                    elif latent_values[j,i] < limit[1]:
                        latent_values[j,i].assign(limit[1])
                    else:
                        None
        latent_values = tf.Variable(latent_values)

        loss = []

        for epoch in range(100):
            loss.append(self.opt_step(latent_values, real_coding).numpy())
        if limit==None:
            None
        else:
            for j in range(len(real_coding)):
                for i in range(latent_values.shape[1]):
                    if latent_values[j, i] > limit[0]:
                        latent_values[j,i].assign(limit[0])
                    elif latent_values[j, i] < limit[1]:
                        latent_values[j,i].assign(limit[1])
                    else:
                        None
        return latent_values


    def predict(self, input_data, scaler, limit=None):
        """
        Optimizes the latent space of the input then produces a prediction from
        the generator.
        """

        latent_values = self.optimize_coding(input_data,limit=limit)

        predicted_vals_1 = scaler.inverse_transform((self.generator.predict(tf.convert_to_tensor(latent_values)).reshape(len(input_data), self.n_features)))
        predicted_vals_1 = predicted_vals_1.reshape(len(input_data), self.n_features)

        predicted_vals = predicted_vals_1[1:,:]
        return predicted_vals



