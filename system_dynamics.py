import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dynamics_methods import *

# This file contains classes for system dynamics data synthesis.
# Parent class for system dynamics modules.
# This module should store the dynamics equations, and be
# able to generate data as needed.
class System():
    def __init__(self, d, init=None, t0=0, tf=1, dt=0.01, noise=0, pred=0):
        self.d = d
        if init is None or init.shape != (d,):
            self.init_default = np.zeros((d, ))
        else:
            self.init_default = init
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.noise = noise
        self.pred = pred
        
        # Stores a list of input functions
        self.u_func_list = []
        # Stores a list of init values
        self.init = []
        # Stores a list of time information
        self.t0s = []
        self.tfs = []
        self.dts = []
        
        # Stores synthesized data
        self.data_generated = []
        self.Inputset = []
        self.Outputset = []
        self.Timeset = []
        # This variable below is deprecated
        self.seg_ind_list = [] # All 3 should be turned into arrays later
        
        self.output_d = 1
    
    # Get-set methods (only ones that I'll frequently use)
    def clear_u_func_list(self):
        self.u_func_list = []
    
    def add_u_func(self, u_func, init=None, times=None):
        # Argument "init" should be a list of arrays if u_func is a list and each array matches a different u_func item, 
        # or be an array otherwise.
        # Argument "times" should be in the form of [(t0,tf,dt), (t0,tf,dt), ...] or (t0, tf, dt), like "init".
        
        #self.u_func_list.append(u_func)
        # Had to change the code above... otherwise python says u_func is not an iterable even though
        # I only want to add it in a list as an object like before. 
        
        # Added on 0707: Allows setting different init values
        # Added on 0804: Allows specifying custom time range.
        if init is None:
            init = self.init_default
        if times is None:
            times = (self.t0, self.tf, self.dt)
        if type(u_func) is list:
            self.u_func_list += u_func
            # Check if "init" is a corresponding list satisfying our requirements
            if type(init) is list and len(init) == len(u_func):
                init = [self.init_default if initial is None else initial   for initial in init]
                self.init += init
            else:
                # Note: This place will scream error in the future if the lengths don't match.
                self.init += [init] * len(u_func)
            # Do the same thing for "times"
            if type(times) is list and len(times) == len(u_func):
                self.t0s += [t[0] for t in times]
                self.tfs += [t[1] for t in times]
                self.dts += [t[2] for t in times]
            else:
                # Note: This place will scream error in the future if the lengths don't match.
                self.t0s += [times[0]] * len(u_func)
                self.tfs += [times[1]] * len(u_func)
                self.dts += [times[2]] * len(u_func)
            self.data_generated += [False] * len(u_func)
        else:
            self.u_func_list += [u_func]
            self.init += [init] # If u_func is not list, then treat init as non-list by default.
            self.t0s += [times[0]]
            self.tfs += [times[1]]
            self.dts += [times[2]]
            self.data_generated += [False]
    
    # Data generation methods
    def dynamics(self):
        # Contains system dynamics; to be filled individually
        pass
    
    def output(self):
        # Contains how to generate output
        pass
    
    def data_generator(self):
        # Combines data_generator() with other code procedures.
        # Synthesizes system trajectory by feeding input u(t) into specified dynamics function y().
        # Uses odeint solver.
        # Update 0804: Added functionality for different time information for different simulation datasets
        # Update 0804: Enabled it to check if existing datasets have already been previously generaged when being called. 
#         t = np.arange(self.t0, self.tf, self.dt)
#         Nd_seg = np.ceil((self.tf - self.t0) / self.dt)

        # Trajectory storage variables
        data_list = []
        input_list = []
        output_list = []
        
        # Generate data for each segment
#         for i in range(Nseg):
        Nseg = 0
        for i, ufunc in enumerate(self.u_func_list):
            # If the i-th input function already had its corresponding dataset generated, then we skip it in this loop.
            # Later we'll combine all newly-generated datasets (stored in data_list etc.) with existing datasets (in 
            # self.Inputset etc.)
            if self.data_generated[i]:
                continue
            else:
                self.data_generated[i] = True
                Nseg += 1
            t = np.arange(self.t0s[i], self.tfs[i], self.dts[i])
            Nd_seg = np.ceil((self.tfs[i] - self.t0s[i]) / self.dts[i])

            x = odeint(func=self.dynamics, y0=self.init[i], t=t, args=(self.u_func_list[i],))
            data = np.vstack((t, x.T))
            data_list.append( data )
            # Regenerate input time history
            input_list.append( [self.u_func_list[i](t) for t in data_list[i][0,:]] )
            # Regenerate the oscillator output g in the paper
            output_list.append( self.output(data) )
            #output_list.append( np.array([self.output(x) for x in np.swapaxes(data_list[i],0,1)]) )
            
        # Put together as one matrix with time-input data, and one with output data.
        # Don't concatenate them into one full array yet - don't want to introduce complication to indexing.
        # The replication doesn't use time input, but we include it anyways.
        # The input consists of [velocity, position]. Still kept the input in the last row for good measure.
        # The first one in the list is the training data; the rest are test data.
        # Noise was added to output data as well.
#         Nseg = len(self.u_func_list)
        InputDataset = [ np.vstack((data_list[i], input_list[i])) for i in range(Nseg) ]
        # Check output dimension, because this can become tricky...
        if len( output_list[i].shape ) == 1:
            # If it's 1D (N,), then likely we need to reshape it into 2D (1,N)
            OutputDataset = [ opt.reshape(1,-1) + opt * self.noise * np.random.randn(1, opt.shape[-1]) \
                              for opt in output_list ] # -1 stands for the dim size to be inferred
#             OutputDataset = [ output_list[i].reshape(1,-1) + \
#                               output_list[i] * self.noise * np.random.randn(1, output_list[i].shape[-1]) \
#                               for i in range(Nseg) ] # -1 stands for the dim size to be inferred
        else:
            OutputDataset = [ opt + opt * self.noise * np.random.randn(opt.shape[0], opt.shape[1]) \
                              for opt in output_list ] # -1 stands for the dim size to be inferred

        # Index list for train and test data
#         seg_ind_list = [0, data_list[0].shape[1]]
        
        #print(data_list[0].shape)
        #print(len(input_list[0]))
        #print(output_list[0].shape)
        #print(InputDataset[0].shape)
        #print(OutputDataset[0].shape)

        self.Inputset += InputDataset
        self.Outputset += OutputDataset
        self.Timeset += [InputData[[0]] for InputData in InputDataset]
#         self.seg_ind_list = seg_ind_list
        
        self.timeind = 0
        self.stateind = 1
        self.inputind = data.shape[0]
    
    # Returns a certain group of input data
    def take_dataset(self, inds):
        # Input: "inds" is a list of indices. 
        # Returns the synthesized data at those indices.
        Inputset = [self.Inputset[i] for i in inds]
        Outputset = [self.Outputset[i] for i in inds]
        #seg_ind_list = [0] + [self.seg_ind_list[i+1] - self.seg_ind_list[i] for i in inds]
        #seg_ind_list = np.cumsum(seg_ind_list)
        return Inputset, Outputset #, seg_ind_list
    
    # Plotting function
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        pass
        
    ## 0714 I'm phasing out this framing() method. Refer to dynamics_methods.py instead.
    # Process data inside this module into frames
    def framing(self, input_data=None, output_data=None, data_ind=None, frame_size=1, pred_size=0):
        # Segments input data into numerous training timeframes
        # Arguments:
        # input_data: The training input data that is to be made into overlapping frames. Should be 2D.
        # output_data: The labels of corresponding input data. Will be made into matching frames.
        # data_ind: The indices where data from a new segment starts.
        # Outputs:
        # input_frames: A list of arrays of frames, each in the shape of (Nframes, <2D frame shape>)
        # output_frames: A list of array of outputs corresponding to each frame in input_frames
        # pred_frames: <Deprecated> An array of future frames corresponding to each frame in input_frames
        if input_data is None:
            input_data = self.Inputset
        if output_data is None:
            output_data = self.Outputset
        if data_ind is None:
            data_ind = self.seg_ind_list # But... do we really use it at all???

        Ninputrow = input_data[0].shape[0]
        Noutputrow = output_data[0].shape[0]

        # Generate data by splitting it into successive overlapping frames.
        # Its first dimension is going to be samples. Each 2D sample occupies the 2nd and 3rd.
        # Empty lists to hold the results:
        input_frames = []
        output_frames = []
        #pred_frames = []
        #frames_ind = [0]
        # Process each segment
        ### Consider using framing_helper later
        for i in range(len(input_data)):
        #for i in range(len(data_ind)-1):
            #Nframes = data_ind[i+1]-data_ind[i] - pred_size - frame_size
            Nframes = input_data[i].shape[1] - pred_size - frame_size
            inframes = np.zeros((Nframes, Ninputrow, frame_size))
            outframes = np.zeros((Nframes, Noutputrow, 1)) # Assuming scalar output
            #predframes = np.zeros((Nframes, Ninputrow, pred_size))

            # Put into frames
            for j in range(Nframes):
                inframes[j,:,:] = input_data[i][:,j:j+frame_size]
                outframes[j,:,:] = output_data[i][:,[j+frame_size-1]] # Output at the final timestep.
                
                #!!!#!!!# Added additinoal square bracket to preserve dimensionality #!!!#!!!#

            # Also offer complementary future state if you're into predictions
            #if pred_size > 0:
            #    for j in range(Nframes):
            #        predframes[j,:,:] = input_data[i][:, j+frame_size:j+frame_size+pred_size]

            input_frames.append(inframes)
            output_frames.append(outframes)
            #pred_frames.append(predframes)
            #frames_ind.append(Nframes)

        # Combine everything into arrays... this step is transferred to networks.train
        #input_frames = np.concatenate(input_frames)
        #output_frames = np.concatenate(output_frames)
        #pred_frames = np.concatenate(pred_frames)
        #frames_ind = np.cumsum(frames_ind)

        return input_frames, output_frames #, pred_frames, frames_ind

    ### 0714 I'm phasing out these two methods below. See dynamics_methods and embedding_methods instead.
    ## Addition during the week of 0619-0626: Functions for delay embedding
    # This method is preserved here for compatibility considerations. It should've been removed
    # from this class if the design style were to be followed. Find it at embed_dynamics.py instead.
    def delay_embed(self, dt_int, de, inds=[], Inputset=None, Outputset=None, symmetric=False):
        # As of now, we assume that:
        # 1) dt_int is an integer specifying the delay as the number of dts.
        #    Currently don't support intervals that aren't integer multiples of dt.
        # 2) de is a provided embedding dimension.
        #    Might consider implementing FalseNN for automatic de detection later.
        # 3) This implementation, by default, assumes that:
        #    - The caller wants to know the embedding at any instant;
        #    - The caller wants to learn the relationship between embedding and some
        #      specific output value (not an embedding, but only a slice).
        #      If the output should be an embedding, create a subclass and overwrite dis.
        # The symmetric flag indicates how the output frames match up with input frames.
        # As an example, consider input = [1,2,3,4,5] and output = [a,b,c,d,e],
        # with delay = 1 and dimension = 3.
        # If symmetric is False: 
        #     We'll have input as [[1,2,3], [2,3,4], [3,4,5]], and output as [c,d,e].
        # If symmetric is True:
        #     We'll have input as [[1,2,3], [2,3,4], [3,4,5]], and output as [b,c,d].
        if symmetric:
            offset = de-1 - (de//2)
        else:
            offset = de-1
            
        if Inputset is None:
            Inputset = self.Inputset
        if Outputset is None:
            Outputset = self.Outputset
        if len(inds) <= 0:
            inds = range(len(Inputset))
        dembed_in = []
        dembed_out = []
        for i in inds:
            dembed_in.append( self.framing_helper(Inputset[i], de, stride=dt_int) )
#             dembed_in.append( self.framing_helper(Inputset[i], de, interv=dt_int) ) # I think it should use stride instead.
            # Not sure what I was thinking when implementing this method. Same for the line below. - 07/07
            
            # If outputset is going to only include a scalar value for each frame...
            # Then keep the same stride, and our input would have to start from the first scalar instead.
            dembed_out.append( self.framing_helper(Outputset[i], 1, stride=dt_int, offset=offset, Nframes=dembed_in[-1].shape[0]) )
#             dembed_out.append( self.framing_helper(Outputset[i], 1, interv=dt_int, offset=de-1) )
        return (dembed_in, dembed_out)
    
    def framing_helper(self, data, framelen, interv=1, stride=1, axis=1, offset=0, Nframes=-1):
        # Helper method for framing. Handles more general cases.
        # Puts a data into multiple frames.
        # framelen: Length of each frame
        # interv  : Number of samples between neighboring frames
        # stride  : How many sample to go between two neighboring samples within a frame
        # axis    : Axis that's being framed
        # Example: Input is data=[1,2,3,4,5,6,7], N=3, interv=2, stride=1, then
        # output would be [ [1,2,3], [3,4,5], [5,6,7] ].
        # This method doesn't do error checking.
        if Nframes < 0:
            Nframes = int(np.ceil( (data.shape[axis] - (framelen-1)*stride - offset) / interv ))
        set_inds = [slice(None)]*data.ndim # https://stackoverflow.com/questions/42656930/numpy-assignment-like-numpy-take
        take_inds = set_inds[:]
        frames = np.zeros( tuple( [Nframes] + list( data.shape[:axis]) + [framelen] + list(data.shape[axis+1:] ) ) )
        
        for i in range(Nframes):
            set_inds[axis] = i
            take_inds[axis] = slice( i*interv+offset, i*interv+framelen*stride+offset, stride )
            frames[i] = data[tuple(take_inds)]
#             frames[tuple(set_inds)] = data[tuple(take_inds)]
        
        return frames
    
    
class Fma(System):
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0):
        super().__init__(2, init, t0, tf, dt, noise)
        # d = 2 for F=ma dynamics
        self.output_d = 2
    
    # Implementing Duffing-oscillator-specific dynamics and outputs
    def dynamics(self, x, t, u):
        # Arguments: x = [velocity, position]^T, t = current time.
        # Output: dx/dt= [acceleration, velocity]^T
        # Assumes scalar position
        return np.array([u(t), x[0]])
    
    def output(self, x):
        return x[1:]
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Ncol = 3 # Plots u(t), v(t), x(t)
        fig,axs = plt.subplots(Nseg, Ncol, constrained_layout=True, figsize = (9, 3*Nseg), squeeze=False)

        for i in range(Nseg):
            axs[i][0].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][-1,::plot_skip_rate])
            axs[i][0].set_title('Acceleration input u')
            axs[i][0].set_xlabel('t')
            axs[i][1].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][1,::plot_skip_rate])
            axs[i][1].set_title('v')
            axs[i][1].set_xlabel('t')
            axs[i][2].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][2,::plot_skip_rate])
            axs[i][2].set_title('x')
            axs[i][2].set_xlabel('t')
        fig.suptitle(title)
        return (fig, axs)

    
class Duffing(System):
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0):
        super().__init__(2, init, t0, tf, dt, noise)
        # d = 2 for Duffing dynamics
        self.output_d = 1
    
    # Implementing Duffing-oscillator-specific dynamics and outputs
    def dynamics(self, x, t, u):
        # Arguments: x = [velocity, position]^T, t = current time.
        # Output: dx/dt= [acceleration, velocity]^T
        # Assumes scalar position
        # The Duffing oscillator dynamics in this code / the source paper: 
        # y'' = u - 1.25y' - 2pi*y - 10y^3
        return np.array([u(t) - 1.25*x[0] - 2*np.pi*x[1] - 10*x[1]*x[1]*x[1], x[0]])
    
    def output(self, x):
        # Note that this input x depends on the Inputset, and might contain time.
        # Here, x = [t, v, x]
        return 1.25*x[1] + 2*np.pi*x[2] + 10*x[2]*x[2]*x[2]
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Ncol = 4 # Including u(t), v(t), x(t), and g(v,x)
        fig,axs = plt.subplots(Nseg, Ncol, constrained_layout=True, figsize = (12, 3*Nseg), squeeze=False)

        for i in range(Nseg):
            axs[i][0].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][3,::plot_skip_rate])
            axs[i][0].set_title('u')
            axs[i][0].set_xlabel('t')
            axs[i][1].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][1,::plot_skip_rate])
            axs[i][1].set_title('v')
            axs[i][1].set_xlabel('t')
            axs[i][2].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][2,::plot_skip_rate])
            axs[i][2].set_title('x')
            axs[i][2].set_xlabel('t')
            axs[i][3].plot(self.Inputset[i][0,::plot_skip_rate], self.Outputset[i][0,::plot_skip_rate])
            axs[i][3].set_title('g(v,x)')
            axs[i][3].set_xlabel('t')
        fig.suptitle(title)
        return (fig, axs)

class DuffingFullState(Duffing):
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0):
        super().__init__(init, t0, tf, dt, noise)
        # d = 2 for Duffing dynamics
        self.output_d = 2
    
    def output(self, x):
        return x[1:]

    
class Lorenz(System):
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0, 
                 sigma=16, rho=45.92, beta=4):
        super().__init__(3, init, t0, tf, dt, noise)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        # Returns z(t) in the Abarbanel (1994) paper
        self.output_d = 1
    
    # Implementing Duffing-oscillator-specific dynamics and outputs
    def dynamics(self, x, t, u):
        # Arguments: x = [x,y,z]^T, t = current time.
        # Output: dx/dt
        # Assumes scalar position. Lorenz doesn't accept inputs u(t).
        return np.array([self.sigma * (x[1] - x[0]), 
                         x[0] * (self.rho - x[2]) - x[1], 
                         x[0] * x[1] - self.beta * x[2]  ])
    
    def output(self, x):
        # Note that this input x depends on the Inputset, and might contain time.
        # Here, x = [t, x,y,z]. Different context from the dynamics method. 
        # The original Abarbanel paper only returned z(t).
        return x[3]
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Ncol = 2
        fig,axs = plt.subplots(Nseg, Ncol, constrained_layout=True, figsize = (12, 3*Nseg), squeeze=False)

        for i in range(Nseg):
            axs[0][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][1,::plot_skip_rate])
            axs[0][i].set_title('x(t)')
            axs[0][i].set_xlabel('t')
            axs[1][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][3,::plot_skip_rate])
            axs[1][i].set_title('z(t)')
            axs[1][i].set_xlabel('t')
        fig.suptitle(title)
        return (fig, axs)

class LorenzFullState(Lorenz):
    def __init__(self, init=None, t0=0, tf=1, dt=0.01, noise=0, 
                 sigma=16, rho=45.92, beta=4):
        super().__init__(init, t0, tf, dt, noise, sigma, rho, beta)
        # d = 3 for full state
        self.output_d = 3
    
    def output(self, x):
        return x[1:] # Returns full state
    
    # Telling the model how to plot the dataset
    def plot_dataset(self, title='Training data', plot_skip_rate=1):
        # Arguments:
        # plot_skip_rate: Only plot a point for every this number of samples
        plt.clf()
        Nseg = len(self.u_func_list)
        Nrow = 3
        fig,axs = plt.subplots(Nrow, Nseg, constrained_layout=True, figsize = (12, 3*Nseg), squeeze=False)

        for i in range(Nseg):
            axs[0][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][1,::plot_skip_rate])
            axs[0][i].set_title('x(t)')
            axs[0][i].set_xlabel('t')
            axs[1][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][2,::plot_skip_rate])
            axs[1][i].set_title('y(t)')
            axs[1][i].set_xlabel('t')
            axs[2][i].plot(self.Inputset[i][0,::plot_skip_rate], self.Inputset[i][3,::plot_skip_rate])
            axs[2][i].set_title('z(t)')
            axs[2][i].set_xlabel('t')
        fig.suptitle(title)
        return (fig, axs)