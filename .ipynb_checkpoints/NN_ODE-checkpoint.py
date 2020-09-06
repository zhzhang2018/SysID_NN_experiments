import tensorflow as tf
import numpy as np
from scipy import integrate
from system_dynamics import System
from networks import NN_Dense
from dynamics_methods import *
from embedding_methods import *
from embed_dynamics import *
 
# 0831 Try to let NN learn ODE instead

# # Default normalization method to override the one in dynamics_methods.
# # Note that the functions of the same names from dynamics_methos are still here, but we're not referencing them.
# def normalize(data, axis, params=None, reverse=False):
#     # Arguments:
#     # data - the input data; could be of any shape
#     # axis - the only axis that should be preserved; i.e. the axis that represents Nfeatures
#     # params - a tuple (mean, variance) to use for data, where each element should have Nfeatures values.
#     #          If params=None, then the system finds the mean and variance by itself.
#     # reverse - whether we want normalization or de-normalization. When reverse is True, params should not be None.
#     Nfeats = data.shape[axis]
#     if params is None:
#         # https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html#numpy.moveaxis
#         X = np.moveaxis(data, axis, 0).reshape(Nfeats, -1)
#         params = ( np.mean(X, axis=1), np.std(X, axis=1) )
#         if reverse:
#             print('Normalize() method warning: Nonsensical input combination. If you want reverse, you should provide params.')
#     # Does nothing to the data. We don't normalize stuff for ODEs.
#     print('Successfully called the normalization method that does nothing.')
#     return data, params

# # New method 0810: Uses normalize() to normalize data with frames
# def normalize_frame(data, params=None, reverse=False):
#     return normalize(data, axis=1, params=params, reverse=reverse)

# This class tries to learn ODE by defining loss as the difference between goal (input) trajectory and
# the predicted trajectory calculated from its derivative predictions using a solver.
class NN_ODE_traj(NN_Delay):
    def __init__(self, dynamics, input_mask, seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', 
                 optimizer='adam', opt_args=(), loss='mse', pred=-1, lr_sched=None,
                 de=3, delay_int=5):
#         loss = 
        super().__init__(dynamics, input_mask, seed, log_dir, tensorboard,
                         Nlayer, Nneuron, learning_rate, activation, output_activation, 
                         optimizer, opt_args=opt_args, loss=loss, pred=pred, lr_sched=lr_sched, 
                         de=de, delay_int=delay_int, sym=False)
        self.no_normalize = True

# This more clever class differentiates the input data first, and then tries to learn the ODE.
# Note: This class treats trajectory observation as input, and the corresponding state derivatives as output.
#       I.e. This class doesn't treat treating output as a function of the states. Its Outputset would always
#            be the derivatives of the entire Inputset. If you give it a dynamics system where the Outputset
#            isn't the same as Inputset, then unexpected behavior could happen...? Maybe. 
class NN_ODE_diff(NN_Delay):
    def __init__(self, dynamics, input_mask, seed=2020, log_dir=None, tensorboard=False,
                 Nlayer=2, Nneuron=5, learning_rate=0.001, activation='relu', output_activation='none', 
                 optimizer='adam', opt_args=(), loss='mse', pred=-1, lr_sched=None,
                 de=3, delay_int=5):
        super().__init__(dynamics, input_mask, seed, log_dir, tensorboard,
                         Nlayer, Nneuron, learning_rate, activation, output_activation, 
                         optimizer, opt_args=opt_args, loss=loss, pred=0, lr_sched=lr_sched, 
                         de=de, delay_int=delay_int, sym=False)
        self.no_normalize = True
    
    # This class's method for taking gradient without ruining the time data.
    # Assumes that data is in the shape of (Nfeatures+1+Ninputs, Nsamples), in the shape of 
    # [a row of time; trajectory history; input history]
    def take_deriv_without_time(self, data, time_ind=0, input_ind=-1):
        # https://numpy.org/doc/stable/reference/generated/numpy.delete.html
        return self.dynamics.dynamics( np.delete(np.delete(data, time_ind, axis=0), input_ind, axis=0),
                                                  data[time_ind], data[input_ind] )
#         return np.vstack( data[time_ind], 
#                           self.dynamics.dynamics( np.delete(np.delete(data, time_ind, axis=0), input_ind, axis=0),
#                                                   data[time_ind], data[input_ind] ),
#                           data[input_ind])
#                           np.gradient( np.delete(data, time_ind, axis=0), dt, axis=1 ) )
    
    # Generates time derivative from training trajectory data.
    # Would've been more complicated if we use PDEs...
    # input is trajectory. Output is trajectory's correct derivatives
    def train_data_generation_helper(self, inds=[]):
        
        if self.frame_size_changed:
#             input_deriv = [self.take_deriv_without_time(inp, 
#                                        self.dynamics.dt, # Spacing between data points. Uniform for now.
#                                        time_ind=0
#                                        #axis=1 # Axis 0 are the different variables. Axis 1 is over time.
#                                       ) for inp in self.dynamics.Inputset]
#             output_deriv = [self.take_deriv_without_time(oup, self.dynamics.dt, 0) for oup in self.dynamics.Outputset]
#             output_deriv = []
#             for i,inp in self.dynamics.Inputset:
#                 # Use the dynamics method inside to find the true dynamics as the Outputset.
#                 # inp[1:-1] is the full state, inp[0] is time, inp[-1] is input u.
#                 output_deriv.append( self.dynamics.dynamics(inp[1:-1], inp[0], inp[-1]) )
            output_deriv = [self.take_deriv_without_time(inp) for inp in self.dynamics.Inputset]
            
            # Store this stuff inside to avoid potential repeated calculation
            self.output_deriv = output_deriv
#             self.input_deriv, self.output_deriv = (input_deriv, output_deriv)
            
            # Assume there's no prediction task in this case, so pred = 0
            (self.Inputset, self.Outputset) = delay_embed(
                  self.delay_int, self.de, self.dynamics.Inputset, output_deriv, 0, symmetric=self.sym)
#                 self.delay_int, self.de, input_deriv, output_deriv, 0, symmetric=self.sym)
            self.frame_size_changed = False
        
        # Call the parent method. Because we don't have any active flags, the parent method won't do
        # anything extra before finally calling the matriarch's generation method.
        return super().train_data_generation_helper(inds=inds)
    
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
        ### TODO: Find a better / more accurate way to integrate
        integrate_results = []
        correct_results = [] # Stores correct trajectory; time step matches the ones in integrate_results.
        for i,res in enumerate(test_results[0]):
            # Note that "res" is the framed result of the shape (Nframes, Nfeatures, 1).
            x0 = Inputset[i][self.dynamics.stateind:self.dynamics.inputind,[0]]
            res_deriv = res.squeeze().reshape(self.dynamics.output_d,-1)
            res_output = Inputset[i][self.dynamics.stateind:self.dynamics.inputind, -res_deriv.shape[1]:]
#             print( res_deriv.shape, res_output.shape, x0.shape )
            # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumtrapz.html
            traj_result = integrate.cumtrapz( res_deriv, x=Inputset[i][0,-res_deriv.shape[1]:], axis=1 ) + x0
            traj_result = np.hstack((x0, traj_result))
            integrate_results.append(traj_result)
            correct_results.append(res_output)
#             x0 = res[2][0,:,0] # The first col (:,0) of the first frame (0) of Inputset (2)
            
        # return: 1) traditional test return package, 2) trajectory based on prediction, 3) real trajectory info
        return test_results, integrate_results, correct_results
