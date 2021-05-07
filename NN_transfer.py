import tensorflow as tf
import numpy as np
from scipy import integrate
from system_dynamics import System
from networks import NN_Dense
from dynamics_methods import *
from embedding_methods import *
from embed_dynamics import *

## 0917 Started this file to attempt using transfer learning for trajectory loss functions
class NN_ODE_traj(NN_Delay):
    ## New argument compared to other similar methods like NN_ODE_diff:
    # ode_model: This argument gives the trained NN_ODE_diff model (or other model) that we base the transfer on.
    def __init__(self, dynamics, input_mask, ode_model, seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', 
                 optimizer='adam', opt_args=(), loss='mse', pred=-1, lr_sched=None,
                 de=3, delay_int=5, no_normalize=True,
                 pred_frame_len=1, pred_frame_weight_func=None, pred_frame_interv=1):
        
        super().__init__(dynamics, input_mask, seed, log_dir, tensorboard,
                         Nlayer, Nneuron, learning_rate, activation, output_activation, 
                         optimizer, opt_args=opt_args, loss=loss, pred=pred, lr_sched=lr_sched, 
                         de=de, delay_int=delay_int, sym=False)
        self.ode_model = ode_model
        self.no_normalize = no_normalize
        self.pred_frame_len = pred_frame_len # How long should the prediction training trajectory be
        self.pred_frame_interv = pred_frame_interv # How often to evaluate a chunk of trajectory
        
        # Assign a weighting function for the cumulative trajectory
        if hasattr(pred_frame_weight_func, '__call__'):
            self.pred_frame_weight_func = pred_frame_weight_func
        else:
            def exp_weight_func(array, time_array=None, init_weight=0.001):
                # Takes in an array (1D), assumed to be the loss at different times, and optionally a 
                # time array assumed to be cumulative.
                # We assign the last value a weight = 1, and the first value a weight = init_weight, and interpolate in-between.
                if time_array is None:
                    time_array = np.arange(len(array))
                # Weight(t) = exp( -gamma * (tf - t) ) with tf being the final time, and gamma > 0, satisfying
                # the conditions Weight(0) = 0.001, Weight(tf) = 1. I.e. we have ln(0.001)/(0-tf) = gamma
                if init_weight >= 1:
                    init_weight = 0.001
                gamma = np.log( init_weight ) / (time_array[0] - time_array[-1]) 
#                 print('exp func - gamma = ', gamma, ' = log(',init_weight,'/',time_array[0] - time_array[-1],')')
#                 print('input is ', array)
#                 print('weights are ', np.exp(-gamma * (time_array[-1] - time_array)))
#                 print('exp func - result is ', np.dot( array, np.exp(-gamma * (time_array[-1] - time_array)) ))
                return np.dot( array, np.exp(-gamma * (time_array[-1] - time_array)) )
                
            self.pred_frame_weight_func = exp_weight_func
        
    ## 0917: Overwrite the construct method in the parent classes, because we'll need to build our own.
    # This method will be called by the code that builds networks when needed, or when set_input_mask() is called. 
    # Doesn't work with non-default activations like LeakyReLU for now.
    def construct(self):
        self.train_prep()
        
        # Construct model
        Regularizer = tf.keras.regularizers.l1(0.001)
        base_model = tf.keras.models.clone_model(self.ode_model)
        
        # Copied from tutorial; not sure if it will work
        # https://www.tensorflow.org/guide/keras/transfer_learning#the_typical_transfer-learning_workflow
        # Alternatively I should consider building it layer by layer except for the outpout layer of the base model.
        base_model.trainable = False
        
        self.model = base_model
        self.model.add( tf.keras.layers.Reshape((np.prod(self.output_shape),)) ) # Squish previous model's output to 1D
        self.model.add( tf.keras.layers.Dense(
                    self.Nneuron, activation = self.activation, kernel_regularizer = Regularizer
                ) )
        self.model.add( tf.keras.layers.Dense(np.prod(self.output_shape), activation = self.output_activation) )
        self.model.add( tf.keras.layers.Reshape(self.output_shape) )
        
#         inputs = tf.keras.Input(shape=self.input_shape)
#         x = base_model(inputs, training=False)
#         x = tf.keras.layers.Dense(self.Nneuron, activation = self.activation, kernel_regularizer = Regularizer)(x)
#         x = tf.keras.layers.Dense(np.prod(self.output_shape), activation = self.output_activation)(x)
#         outputs = tf.keras.layers.Reshape(self.output_shape)(x)
#         self.model = tf.keras.Model(inputs, outputs)
        
        self.model.compile(optimizer = self.optimizer, loss = self.loss)
        
    # 0917 Define customized loss function.
    # If we were to make it compatible with tensorflow default methods, then the requirements would be that the arguments satisfy:
    # 1) Input has 2 arguments - actual value and model prediction - of the type "tensor"; 2) Output is scalar.
    # But if we're writing our own, then we can do what we like here. Currently I let input_traj be a frame of the true trajectory
    # in the shape of (Nfeatures+2, frame_len), and traj_pred be the predicted derivatives of the same size.
    # We don't want to regenerate the true trajectory from true derivatives; we want to take in longer true trajectory arguments.
    # Currently we try:
    # 1. Take in the actual trajectory record as the Inputset (including time and input) with shape (Nfeatures+2, frame_len),
    #    hopefully cut into pieces that we want to fucus on. Also take in deriv_pred with shape (Nfeatures, frame_len).
    # 2. Use the time and input from Inputset and deriv_pred to 
    #    1) predict derivatives using model, and 2) approximate trajectory using cumtrapz().
    # 3. Find the SoS between real trajectory (excluding time and input) and appoximated trajectory. Return this scalar.
    #    One possible method is to use self.loss determined in initialization for this step.
    # However, keep in mind that the frame length in training inputs for model.fit() has to be 1.
    def find_loss(self, input_traj, deriv_pred):
        timeset = input_traj[self.dynamics.timeind]
        x0 = input_traj[self.dynamics.stateind:self.dynamics.inputind, [0]]
        state_traj = input_traj[self.dynamics.stateind:self.dynamics.inputind]
        u_traj = input_traj[self.dynamics.inputind:]
        
        traj_pred = self.deriv2traj(deriv_pred, timeset, x0)
        
        sos_traj = np.linalg.norm(traj_pred - state_traj, axis=0)
#         print('find_loss, sos_traj and timeset - ', sos_traj, timeset)
        return self.pred_frame_weight_func( sos_traj, timeset )
        
#         return self.loss(traj_pred, state_traj)

    # Might have to do this...
    class TrajLoss(tf.keras.losses.Loss):
        def call(self, y_true, y_pred):
#             return tf.convert_to_tensor(y_pred)
            print('TrajLoss call - ', y_true, y_pred)
            y_pred = tf.convert_to_tensor(y_pred)
            y_true = tf.cast(y_true, y_pred.dtype)
            return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=0)

#             return self.pred_frame_weightless_func( y_true, y_pred )
#             return self.find_loss(y_true, y_pred)
        
#         def find_loss(self, input_traj, deriv_pred):
#             timeset = input_traj[self.dynamics.timeind]
#             x0 = input_traj[self.dynamics.stateind:self.dynamics.inputind, [0]]
#             state_traj = input_traj[self.dynamics.stateind:self.dynamics.inputind]
#             u_traj = input_traj[self.dynamics.inputind:]

#             traj_pred = self.deriv2traj(deriv_pred, timeset, x0)

#             sos_traj = np.linalg.norm(traj_pred - state_traj, axis=0)
#             return self.pred_frame_weight_func( sos_traj, timeset )
        
    def find_loss_with_Loss(self, input_traj, deriv_pred, lossObj):
        timeset = input_traj[self.dynamics.timeind]
        x0 = input_traj[self.dynamics.stateind:self.dynamics.inputind, [0]]
        state_traj = input_traj[self.dynamics.stateind:self.dynamics.inputind]
        u_traj = input_traj[self.dynamics.inputind:]
        
        traj_pred = self.deriv2traj(deriv_pred, timeset, x0)
        
#         sos_traj = np.linalg.norm(traj_pred - state_traj, axis=0)
        return lossObj( self.pred_frame_weight_func( state_traj, timeset ), self.pred_frame_weight_func( traj_pred, timeset ) )
    
    def deriv2traj(self, deriv, time, x0):
        # This method uses cumtrapz() to automatically obtain the trajectory prediction.
        # time is expected to be a 1D vector.
        if len(time) > deriv.shape[1]:
            time = time[-res_deriv.shape[1]:]
        print('deriv2traj - ', deriv.shape, time.shape, x0.shape)
        traj_result = integrate.cumtrapz( deriv, x=time, axis=1 ) + x0
        traj_result = np.hstack((x0, traj_result))
        return traj_result
    
    # Takes gradient without ruining the time data. Assumes that data is in the shape of (Nfeatures+1+Ninputs, Nsamples).
    def take_deriv_without_time(self, data, time_ind=0, input_ind=-1):
        # https://numpy.org/doc/stable/reference/generated/numpy.delete.html
        return self.dynamics.dynamics( np.delete(np.delete(data, time_ind, axis=0), input_ind, axis=0),
                                                  data[time_ind], data[input_ind] )
    # Generation method copied from NN_ODE_diff
    # input is trajectory. Output is trajectory's correct derivatives
    def train_data_generation_helper(self, inds=[]):
        
        if self.frame_size_changed:
            output_deriv = [self.take_deriv_without_time(inp) for inp in self.dynamics.Inputset]
            
            # Store this stuff inside to avoid potential repeated calculation
            self.output_deriv = output_deriv
            
            # Assume there's no prediction task in this case, so pred = 0
            (self.Inputset, self.Outputset) = delay_embed(
                  self.delay_int, self.de, self.dynamics.Inputset, output_deriv, 0, symmetric=self.sym)
            self.frame_size_changed = False
        
        # Call the parent method. Because we don't have any active flags, the parent method won't do
        # anything extra before finally calling the matriarch's generation method.
        super().train_data_generation_helper(inds=inds)
        
        # For this class, we don't want concatenated data. We want to be able to treat different sections separately.
        # Thus we do some extra work by copying some of the code from the NN.train_..._helper() method.
        if len(inds) > 0:
            (Inputset, Outputset) = ( [self.Inputset[i] for i in inds], [self.Outputset[i] for i in inds] )
        else:
            Inputset = self.Inputset
            Outputset = self.Outputset
        # Then normalize data directly. Note that the normalization parameters were already obtained from the concatenated
        # data from the parent method call. For this one we're just normalizing them individually for training usage.
        Inputset = [normalize_frame(inp, no_normalize=self.no_normalize)[0] for inp in Inputset]
        Outputset = [normalize_frame(oup, no_normalize=self.no_normalize)[0] for oup in Outputset]
        return Inputset, Outputset
    
    ## 0917
    # If we have to write our own training loop, then we'll have to redefine the train() method.    
    def train(self, epoch=50, inds=[]):
        # Generate framed data after normalization. Note that this Inputset is not masked, comparing to other classes.
        (Inputset, Outputset) = self.train_data_generation_helper(inds)
        # Train, and store all relevant data logs
        self.history.append( self.model_fit(
            Inputset, Outputset, inds, epochs=epoch, callbacks=self.train_callback
        ) )
        self.current_iter += 1
    # And we'll implement our own model_fit() method instead of using model.fit():
    # Reference: https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
    # Note: The arguments Inputset and Outputset are expected to be normalized lists of different sets of training data, 
    #       instead of concatenations like in all other classes.
    def model_fit(self, Inputset, Outputset, inds=[], epochs=50, callbacks=None):
        # Note: I don't know how to do callbacks here, so I'll just ignore them.
        
        # Use the model to generate a prediction on derivatives.
        # Note that Inputset is not masked. When we run test() or predict() in other classes, the Inputset is usually
        # already embedded/framed, masked, normalized, and then finally being passed to predict(). 
        # Here it has been embedded and individually normalized, but still hasn't been masked. Thus:
        # * The squeeze and T applied at the end helps it return to the correct shape of (Nvariables, Nsamples).
#         print('', self.input_mask, len(Inputset))
#         print(Inputset[0][:,self.input_mask,:].shape, Inputset[0].shape)
        derivs = [normalize_frame(
                    self.model.predict(inp[:,self.input_mask,:]), 
                    params=self.output_norm_params, reverse=True, no_normalize=self.no_normalize
                )[0].squeeze().T for inp in Inputset]
        
        # Shapes:
        # derivs = [ (Nobservables, Nsamples) ], each depending on the data section
        # Inputset = [ (Nframes, Nfeatures+2, delay_dim) ]
        
        # Next we need to generate chunks of data for training long-term predictions. 
        # This is when pred_frame_interv and pred_frame_len come into play.
        # Note that to generate this kind of prediction data, we would need non-normalized, non-framed input data,
        # so we would need to resort to the inputset in dynamics. I hate myself. 
        true_trajs = []
        time_offsets = []
        if len(inds) <= 0:
            inds = [i for i in range(len(Inputset))]
        for i,j in enumerate(inds):
            true_trajs.append( 
                # Our loss method accepts full Inputset, so we don't have to index it
                framing_helper( 
                    self.dynamics.Inputset[j][:, -derivs[i].shape[1]:],
#                     self.dynamics.Inputset[j][self.dynamics.stateind:self.dynamics.inputind, -derivs[i].shape[1]:],
                    self.pred_frame_len, interv=self.pred_frame_interv
                )
            )
            time_offsets.append( self.dynamics.Inputset[j].shape[1] - derivs[i].shape[1] )
        
#         true_traj_joined = np.concatenate(true_trajs)
        
        # Maybe we could store all possible indices into a single list, and refer to them when needed
        ind_collection = []
        for i,j in enumerate(inds):
            for k in range(true_trajs[i].shape[0]):
                ind_collection.append((i,j,k))
                
        # The predicted trajectories, as well as updated derivative predictions, would need to be obtained inside the loop.
        
        # Define a loss method that tries to fool the tape by really performing a MSE on prediction
        loss_object = tf.keras.losses.MSE
        self_loss_object = self.TrajLoss()
        def loss(model, x, y, training=True):
            y_ = model(x, training=training) # Is this line enough to store the forward pass?
#             print('loss - ',x.shape, y.shape, y_.shape)
            return loss_object(y_true=y, y_pred=y_)
        
        # Define a wrapper method that calculates a scalar loss, similar to the loss_object in the tutorial, 
        # and directly provides the gradient information
        def grad_loss(true_traj, pred_traj, inputs, targets):
            with tf.GradientTape() as tape:
                loss_value = loss(self.model, inputs, targets, training=True) # loss is a predefined function above (tutorial)
                loss_value = self.find_loss_with_Loss(true_traj, pred_traj, self_loss_object)
                print('grad_loss procedure - ', loss_value, self.find_loss(true_traj, pred_traj))
#                 loss_value = self_loss_object(y_true=true_traj, y_pred=pred_traj)
#                 loss_value = tf.convert_to_tensor(self.find_loss(true_traj, pred_traj), np.float64)
            return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
        
        # Define a gradient learning method
        # Copied from the tutorial... but not sure if we really need it 
#         def grad(model, inputs, targets):
#             with tf.GradientTape() as tape:
#                 loss_value = loss(model, inputs, targets, training=True) # loss is a predefined function in the tutorial
#             return loss_value, tape.gradient(loss_value, model.trainable_variables)
        
        # Start the main train loop
        train_loss_results = []
        train_accuracy_results = []

        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean() # Keep it?
            
            # Generate trajectory predictions
            '''
            traj_preds = []
            for i,j in enumerate(inds):
                traj_preds.append( np.zeros_like(true_trajs[i]) )
                for frame_ind in range(true_trajs[i].shape[0]):
                    traj_preds[-1][frame_ind] = self.deriv2traj(
                        derivs[i][:, frame_ind*(self.pred_frame_interv)], # derivatives
                        self.dynamics.Inputset[j][self.dynamics.timeind, 
                                                  (frame_ind*(self.pred_frame_interv)+time_offsets[i]):(frame_ind*(self.pred_frame_interv+self.pred_frame_len)+time_offsets[i])], # time values
                        self.dynamics.Inputset[j][self.dynamics.stateind:self.dynamics.inputind, [frame_ind*(self.pred_frame_interv)+time_offsets[i]]] # initial condition, 2D
                    )
                
                traj_preds.append( 
                    framing_helper( 
                        self.deriv2traj(
                            self.dynamics.Inputset[j][self.dynamics.stateind:self.dynamics.inputind, -derivs[i].shape[1]:], time, x0)
                        ,
                        self.pred_frame_len, interv=self.pred_frame_interv
                    )
                )
            '''
            # Training loop
            ind_perm = np.random.permutation(ind_collection)
            for (i,j,k) in ind_perm:
                traj_pred = self.deriv2traj(
                        derivs[i][:, (k*self.pred_frame_interv):(k*self.pred_frame_interv+self.pred_frame_len)], # derivatives
                        self.dynamics.Inputset[j][self.dynamics.timeind, 
        (k*self.pred_frame_interv+time_offsets[i]):(k*self.pred_frame_interv+self.pred_frame_len+time_offsets[i])], # time values
                        self.dynamics.Inputset[j][self.dynamics.stateind:self.dynamics.inputind, 
                                                  [k*(self.pred_frame_interv)+time_offsets[i]]  ] # 2D initial conditions
                    )
                true_traj = true_trajs[i][k]
                
                # I'm not sure if I'm putting the correct input and target values below, tbh.
                # If all go according to plan, then those two arguments shouldn't matter.
#                 print( Inputset[i][[k],self.input_mask,:].shape, Outputset[i][k].shape, Inputset[i].shape, Outputset[i].shape,
#                        type(Inputset[i]))
                print('training loop - ', (i,j,k), traj_pred.shape, true_traj.shape)
                print('training loop - Inputset and Outputset size - ',Inputset[i].shape, Outputset[i].shape)
                loss_value, grads = grad_loss(true_traj, traj_pred, 
                                              Inputset[i][k,self.input_mask,:][np.newaxis,:,:], Outputset[i][k][np.newaxis,:,:])
#                                               Outputset[i][k,self.dynamics.stateind:self.dynamics.inputind,:])
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss

            # End epoch
            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result() ))
                
            # Generate new derivative 
            derivs = [normalize_frame(
                            self.model.predict(inp[:,self.input_mask,:]), 
                            params=self.output_norm_params, reverse=True, no_normalize=self.no_normalize
                        )[0].squeeze().T for inp in Inputset]
#             derivs = [self.model.predict(inp[:,self.input_mask,:]).squeeze().T for inp in Inputset]


        
    ## Copied from NN_ODE_diff because the network is still learning derivatives:
    # To test, we also want to output the trajectory calculated from prediction.
    # Note: If Inputset and Outputset are custom arguments, then make sure they comply with the input mask thing.
    #       This method would assume that Inputset is trajectory.
    # Outputset is not needed in this method, because it would be calculated from Inputset (full trajectory inluding t and u).
    def test(self, Inputset=None, dt=0, inds=[], squeeze=True):
        if dt <= 0:
            dt = self.dynamics.dt
        
        if Inputset is not None:
            # Take the gradient and put them into parent call, assuming the inputsets are of the expected shape
#             input_deriv = [self.take_deriv_without_time(inp, dt, 0) for inp in Inputset]
#             output_deriv = [self.take_deriv_without_time(oup, dt, 0) for oup in Outputset]
            output_deriv = [self.take_deriv_without_time(inp) for inp in Inputset]
        elif len(inds) > 0:
#             input_deriv = [self.input_deriv[i] for i in inds]
            Inputset = [self.dynamics.Inputset[i] for i in inds]
            output_deriv = [self.output_deriv[i] for i in inds]
        else:
            Inputset = self.dynamics.Inputset #self.input_deriv
            output_deriv = self.output_deriv
            
        # The parent method would: 1) Delay embed the provided Inputset and Outputset;
        # 2) Obtain Timeset, normalize the Inputset, and apply input_mask;
        # 3) Call the matriach method, where it would run the prediction, de-normalize result, squeeze, and return.
        test_results = super().test(Inputset, output_deriv, inds=[], squeeze=squeeze)
        # Note that Inputset in this scope is the full state trajectory with time and input history.
        # The Inputset in test_results is the masked / stripped down version with only the observed states' history. 
        # The prediction that happened in the parent call is based on the masked Inputset, not the full state one.
        
        # Then we run odeint on the returned result and see what happens.
        # To translate the code below: We 
        # 1) Find x0 by referring to the first column of test data
        # 2) Use cumsum() or similar integration methods to approximate integration 
        integrate_results = []
        correct_results = [] # Stores correct trajectory; time step matches the ones in integrate_results.
        for i,res in enumerate(test_results[0]):
            # Note that "res" is the framed result of the shape (Nframes, Nfeatures, 1).
            x0 = Inputset[i][self.dynamics.stateind:self.dynamics.inputind,[0]]
            res_deriv = res.squeeze().T # DO NOT use reshape!!! It doesn't preserve the spatial relationship
            res_output = Inputset[i][self.dynamics.stateind:self.dynamics.inputind, -res_deriv.shape[1]:]
            timeset = test_results[3][i]
            # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumtrapz.html
            traj_result = integrate.cumtrapz( res_deriv, x=Inputset[i][0,-res_deriv.shape[1]:], axis=1 ) + x0
            traj_result = np.hstack((x0, traj_result))
            integrate_results.append(traj_result)
            correct_results.append(res_output)
            
        # return: 1) traditional test return package, 2) trajectory based on prediction, 3) real trajectory info
        return test_results, integrate_results, correct_results