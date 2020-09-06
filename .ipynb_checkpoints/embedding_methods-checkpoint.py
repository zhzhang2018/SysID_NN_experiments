# Helper file for embedding-related methods.
import numpy as np
from dynamics_methods import *
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

# Generate delay-embedded data; typically for training and testing data generation.
def delay_embed(dt_int, de, Inputset, Outputset, pred, symmetric=False):
    # As of now, we assume that:
    # 1) dt_int is an integer specifying the delay as the number of dts(samples).
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
    # The "offset" value also decides other stuff, such as how much time into the future
    # would the dataset want to predict. 

    if symmetric:
        offset = (de-1 - (de//2))*dt_int +pred
    else:
        offset = (de-1)*dt_int +pred

    dembed_in = []
    dembed_out = []
    for i in range(len(Inputset)):
        dembed_in.append( framing_helper(Inputset[i], de, stride=dt_int) )
        dembed_out.append( framing_helper(Outputset[i], 1, stride=dt_int, offset=offset) )
        # Although stride doesn't matter here when framelen is 1 for dembed_out.
        
        # Check if they have the same dimensions. If not, shrink one of them.
        Nframes = min( dembed_in[-1].shape[0], dembed_out[-1].shape[0] )
        dembed_in[-1] = dembed_in[-1][:Nframes]
        dembed_out[-1] = dembed_out[-1][:Nframes]
#         print(offset, pred, Nframes, dembed_in[-1].shape[0], dembed_out[-1].shape[0])
    return (dembed_in, dembed_out)

def delay_embed_converter_Garcia_Cao(G2C, args):
    # Helper method that converts the two ways I used to record delay values for FNN methods (Garcia's vs Cao's)
    if G2C:
        js, ts = args
        tts = np.zeros((max(js), len(js)), dtype=int)
        jinds = [0] * len(js)
        for i,j in enumerate(js):
            tts[j,jinds[j]] = ts[i]
            jinds[j] += 1
        return tts
    else:
        js = []
        ts = []
        tts = args
        for j in range(tts.shape[0]):
            for k in range(tts.shape[1]):
                if tts[j,k] <= 0:
                    break
                # Store the index of the largest delay in the embedding
                ts.append(tts[j,k])
                js.append(j)
        return js, ts

# Generate delay-embedded data when delay is nonuniform.
def delay_embed_Garcia(js, ts, Inputset, Outputset, pred, Timeset=None):
    # js - list of row indices
    # ts - list of _accumulative_ delay values (matches each element in js; must have same length)
    # pred - the amount of timesteps the Outputset is supposed to be ahead of from the Inputset
    # New argument addition: "Timeset", where it stores time, and would be embedded/framed to match output.
    # The symmetric flag in the uniform case won't apply here.
    # Outputs:
    # dembed_in - supposed to be of the shape (Nframes, embed_dimension, 1)
    offset = ts[-1] +pred

    dembed_in = []
    dembed_out = []
    dembed_time = []
    for i in range(len(Inputset)):
        frames_list = [ np.transpose(framing_helper(Inputset[i], 1), (0,2,1)) ] # Obtain [X[:,i].T]
        # Shape of the element above becomes (Nframes, 1, Nfeatures) after transpose.
        for j in range(len(js)):
            # Add a layer of delayed variable... each with the shape (Nframes, 1, 1)
            frames_list.append( framing_helper(Inputset[i][ [js[j]] ], 1, offset=ts[j]) )
            # At the end of the loop, we have built a list that looks like:
            #[[X1[0], X1[1], ..., X1[N]],
            # [X2[0], ..., . . ., X2[N]],
            # ... ...
            # [Xn[0], ..., . . ., Xn[N]],  <-- size = (Nframes, 1, Nfeatures) for all rows above
            # [Xj1[tau1], ......, Xj1[N]], <-- size = (Nframes, 1, 1) for this and below
            # ... ...
            # [Xjd[taud], ......, Xjd[N]] ]
            # except that each "row" is actually 3D. Later they will be trimmed to have the same length,
            # and then concatenated into an array of delay embeddings. 
        Nframes_list = [ frame.shape[0] for frame in frames_list ]
        # Produce output frames as well
        dembed_out.append( framing_helper(Outputset[i], 1, offset=offset) )
        
        # Truncate any frames that need to be truncated, so we have the same lengths
        Nframes = min(min(Nframes_list), dembed_out[-1].shape[0])
        dembed_out[-1] = dembed_out[-1][:Nframes]
        frames_list = [frame[:Nframes] for frame in frames_list]
#         print(Nframes,Nframes_list,[fm.shape for fm in frames_list])
        dembed_in.append( np.transpose( np.concatenate( frames_list, axis=2 ), (0,2,1) ) )
        # Result shape returns to being (Nframes, embed_dimension, 1)
#         dembed_in.append( framing_helper(Inputset[i], de, stride=dt_int) )
        if Timeset is None:
            dembed_time.append(dembed_in[-1][:,0,-1])
#             dembed_time = [dembed_in_frame[:,0,-1] for dembed_in_frame in dembed_in]
        else:
            dembed_time.append( framing_helper(Timeset[i], 1, offset=offset)[:Nframes] )
    return (dembed_in, dembed_out, dembed_time)

# Generate delay-embedded data when delay is nonuniform, and every variable contributes at least its current value.
def delay_embed_Cao(ts, Inputset, Outputset, pred, Timeset=None):
    # ts - array of _accumulative_ delay values, i.e. [ [<delays for var1>], ..., [<delays for varM>] ]. Must all be ints.
    # pred - the amount of timesteps the Outputset is supposed to be ahead of from the Inputset
    # Timeset - time step information for each sample; would be embedded/framed to match output.
    maxdelay = np.amax(ts)
    offset = maxdelay + pred

    dembed_in = []
    dembed_out = []
    dembed_time = []
    for i in range(len(Inputset)):
        indcount = 0
        maxind = 0
        # We stack embeddings of each variable onto this list here:
        frames_list = []
#         Nframes_list = []
        for j in range(ts.shape[0]):
            # Stack the current value, and then go through the ts array to pick other delays.
            indcount += 1
            frames_list.append( framing_helper(Inputset[i][ [j] ], 1, offset=0) )
            for k in range(ts.shape[1]):
                if ts[j,k] <= 0:
                    break
                # Store the index of the largest delay in the embedding
                if ts[j,k] == maxdelay:
                    maxind = indcount
                indcount += 1
                frames_list.append( framing_helper(Inputset[i][ [j] ], 1, offset=ts[j,k]) )
#                 Nframes_list.append(frames_list[-1].shape[0])
        
        dembed_out.append( framing_helper(Outputset[i], 1, offset=offset) )
        
        # Truncate any frames that need to be truncated, so we have the same lengths
#         print(Nframes_list, maxind)
#         Nframes = min(min(Nframes_list), dembed_out[-1].shape[0])
        Nframes = min(frames_list[maxind].shape[0], dembed_out[-1].shape[0])
        dembed_out[-1] = dembed_out[-1][:Nframes]
        frames_list = [frame[:Nframes] for frame in frames_list]
        dembed_in.append( np.transpose( np.concatenate( frames_list, axis=2 ), (0,2,1) ) )
        # Result shape returns to being (Nframes, embed_dimension, 1)
        
        if Timeset is None:
#             dembed_time.append(dembed_in[-1][:,0,-1])
            dembed_time.append(dembed_in[-1][:,0,maxind])
        else:
            dembed_time.append( framing_helper(Timeset[i], 1, offset=offset)[:Nframes] )
    return (dembed_in, dembed_out, dembed_time)

def make_grid(X, Nbin=50, axis=1):
    # Helper method for generating a grid out of 1D or 2D data.
    # Useful for estimating probability distribution.
    # Example: Given 1D input X, this method outputs [min(X), ..., max(X)].
    # Example: Given 2D input X, this method outputs [[min(X[0]), ..., max(x[0])], ..., [min(X[-1]), ..., max(x[-1])]].
    # If axis is changed, then the grid would be going along that axis instead.
    # The output also includes the step of each data row. 
    if len(X.shape) < 2:
        axis = 0
    # Construct grids for estimation.
    # The shape of the final grid is (Nfeatures,Nbin).
    grid_step1 = (X.max(axis=axis) - X.min(axis=axis)) / Nbin
    grid1 = np.linspace(X.max(axis=axis), X.min(axis=axis), num=Nbin).T
#     print(grid1.shape, grid_step1)
    return (grid1, grid_step1)

def get_kernel_density(X, method='sk', Nbin=50, ep=1e-10, total=False, kernel='gaussian', grid_vars=None):
    # Helper method for estimating probability density from observations.
    # X could be 1D or 2D. We'll treat each row separately if 2D.
    # If the input data is of shape (Nsamples, Nfeatures) instead, then the caller should've changed that with a .T
    # Output if total==False:
    #     Returns a grid of shape (Nfeatures, Nbin), where the (i,j)-th element is p(X[i] = j) approximately.
    # Output if total==True:
    #     Returns a Nfeatures-D array of probabilities and a (Nfeatures, Nbin) grid.
    #     The (i1,...,iN)-th element in the prob grid is p(X[0] = grid[i1], ..., X[N-1] = grid[iN]).
    # The entire output is (probability array, grid matrix, grid step array).
    
    ## WARNING: If the data dimension is too large (e.g. >=3) and Nbin is not very small,
    ## then the total=True option could be exponential in computation.

    # If the user provided their own grid variables, then they're expected to be in the same form as this method's default.
    if grid_vars is None:
        grid1, grid_step1 = make_grid(X, Nbin)
    else:
        (grid1, grid_step1) = (grid_vars[0], grid_vars[1])
    # 1D data case - the simplest
    if len(X.shape)==1:
        if method == 'sc':
            # Scipy method
            kde = gaussian_kde(X)
            probs = kde.pdf(grid1) * grid_step1 + ep
        else:
            # Default sklearn method
            kde = KernelDensity(kernel=kernel, bandwidth=grid_step1).fit(X.reshape(-1,1))
            probs = np.exp(kde.score_samples(grid1.reshape(-1,1)).ravel()) * grid_step1 + ep
        return (probs, grid1, grid_step1)
    
    # For higher dimensions, check if we want the joint probability
    elif total:
        # Recall: grid1.shape = (Nfeatures, Nsamples); grid_step1.shape = (Nfeatures,).
        if grid1.shape[0] >= 3:
            print('Warning: The {0}-dimension data might be too much for this method.'.format(grid1.shape[0]))
        # To get the ND array of probabilities, we need to generate mesh coordinates.
        meshes = np.meshgrid(*grid1)
        # meshes is a list of arrays. The list [meshes[0][i], ..., meshes[-1][i]] is a ND coordinate,
        # and in this way, every different i gives a unique coordinate.
        # We place "*" before "grid1", because we don't know N previously. 
        # Reference: https://stackoverflow.com/a/8148493
        # To make the mesh usable for kernel methods, we need to convert them into a collection of (Nfeatures,) vectors:
        coords = np.vstack( [mesh.ravel() for mesh in meshes] )
        
        if method == 'sc':
            # Scipy method
            kde = gaussian_kde(X)
            probs = kde.pdf(coords) * np.prod(grid_step1) + ep
        else:
            # Default sklearn method
            kde = KernelDensity(kernel=kernel, bandwidth=np.amin(grid_step1)).fit(X.T)
            probs = np.exp(kde.score_samples(coords.T).reshape([len(mesh) for mesh in meshes])) * np.prod(grid_step1) + ep
        return (probs, grid1, grid_step1)
        
    # If we don't want the joint probability, but the individual probability of each row
    else:
        # If we're doing this separately, then we recursively call this method row by row.
        return np.array( 
            [get_kernel_density(X[i], method, Nbin, ep, kernel=kernel)[0].ravel() for i in range(X.shape[0])]
        ), grid1, grid_step1

def AMI(X, Y, method=None, Nbin=50, ep=1e-10):
    # Helper function for finding AMI (Average Mutual Information) for two given data, 
    # assuming both inputs are only for one variable each.
    # Expects 2D-shaped inputs with shape = (Nfeatures, Nsamples) for all situations. 
    if method == 'kernel':
        prob12, grid12, grid12step = get_kernel_density(np.vstack((X,Y)), Nbin=Nbin, ep=ep, total=True)
#         # Construct grids for estimation
#         grid_step1 = (np.amax(X) - np.amin(X)) / Nbin
#         grid1 = np.arange(np.amin(X), np.amax(X), grid_step1)
#         grid_step2 = (np.amax(Y) - np.amin(Y)) / Nbin
#         grid2 = np.arange(np.amin(Y), np.amax(Y), grid_step2)
#         Xo, Xd = np.meshgrid(grid1, grid2)

#         # Find the pdf from the object. The matrix operations are learned from the examples of:
#         # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
#         skkde_12 = KernelDensity(kernel='gaussian', bandwidth=min(grid_step1, grid_step2)).fit(np.hstack((X, Y)))

#         # Get probability. Notice the "T" after the end of vstack.
#         prob12 = skkde_12.score_samples( np.vstack( (Xo.ravel(), Xd.ravel()) ).T ).reshape(Xo.shape)
#         # sklearn's KernelDensity also automatically uses log. We also add ep to avoid zero probability.
#         prob12 = np.exp(prob12) * grid_step1 * grid_step2 + ep # Is this the right way?

        # Compute the entropies and marginal probs
        H12 = np.sum( prob12 * np.log(prob12) )
        prob1 = np.sum( prob12, axis=0 )
        H1 = np.sum( prob1 * np.log(prob1) )
        prob2 = np.sum( prob12, axis=1 )
        H2 = np.sum( prob2 * np.log(prob2) )

        # Get MI value
        MI = H12 - H1 - H2

    elif method == 'hist':
        # Histogram fixed bin version. Note that histogram2d wants 1D inputs.
        # Source: https://stackoverflow.com/a/20505476
        H, edge_Xo, edge_Xd = np.histogram2d( X.ravel(), Y.ravel(), bins=Nbin )
        MI = mutual_info_score(None, None, contingency=(H+ep))

    else:
        # Default method: Scikit-learn, continuous method
        # Notice that X has to be 2D and Y has to be 1D in this case...
        MI = mutual_info_regression( X.T, Y.ravel() )[0]
    return MI

# Below are a bunch of methods to find quantities for higher-dimension data
def KL_divergence(X, Nbin=50, ep=1e-10, max_total=False): 
    # K-L divergence / total correlation, according to Wikipedia, is just taking MI to multiple independent variables
    # Ref: https://en.wikipedia.org/wiki/Total_correlation
    total_prob = get_kernel_density(X, method='sk', Nbin=Nbin, ep=ep, total=True)[0]
    indiv_prob = get_kernel_density(X, method='sk', Nbin=Nbin, ep=ep, total=False)[0]
    H_all = np.sum( total_prob * np.log(total_prob) )
    H_indiv = np.sum( indiv_prob * np.log(indiv_prob), axis=1 )
    if max_total:
        # Maximum total entropy - likelihood that one variable measures all others
        return max(H_indiv) - np.sum(H_indiv)
    else:
        return H_all - np.sum(H_indiv)

def KL_total_divergence(X, Y, Nbin=50, ep=1e-10, Ywider=False):
    # Directly uses K-L divergence, and treats each data (X and Y) as a whole probability distribution
    # Ref: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    # Xwider determines which data decides the grid for the other data. By default, Xwider=True,
    #     meaning that we assume X has wider data range (thus larger grid) than Y.
    # Note that this interpretation of K-L divergence is ASYMMETRIC. We'll treat X as P, and Y as Q.
    # This method measures the amount of information lost when Y is used to approximate X.
    if Ywider:
        (X,Y) = (Y,X)
    total_probX, gridX, gridXstep = get_kernel_density(X, method='sk', Nbin=Nbin, ep=ep, total=True)
    total_probY = get_kernel_density(Y, method='sk', Nbin=Nbin, ep=ep, total=True, grid_vars=(gridX, gridXstep))[0]
    if Ywider:
        # We swapped them before, so we'll swap them up again.
        return np.sum( total_probY * np.log(total_probY) - np.log(total_probX) )
    else:
        return np.sum( total_probX * np.log(total_probX) - np.log(total_probY) )
    
def interaction_information(X, Nbin=50, ep=1e-10):
    # Interaction information among all rows of input X. 
    # Ref: https://en.wikipedia.org/wiki/Interaction_information
    ## WARNING: Inefficient implementation! Exponential computation with the number of rows in X.
    axis_inds = np.arange(X.shape[0])
    axis_TF = [[True, False]] * X.shape[0]
    # We get all possible combinations of axis via meshgrid
    meshes = np.meshgrid( *axis_TF )
    # Now, for example, if X has 3 rows, then meshes is [ [TTTTFFFF], [TTFFTTFF], [TFTFTFTF] ].
    # Notice that each row of meshes is itself a ND array.
    coords = np.vstack( [mesh.ravel() for mesh in meshes] )
    # Now, each row of coords is unraveled into an 1D array. 
    # We can thus begin to find the joint probability distribution of every combination of variables / rows.
    total_prob = get_kernel_density(X, method='sk', Nbin=Nbin, ep=ep, total=True)[0]
    total_info = 0
    for i in range(coords.shape[1]):
        axis_i = axis_inds[coords[:,i]]
        axis_oddity = len(axis_i)
#         axis_oddity = np.count_nonzero(axis_i)
#         print(i, coords[:,i], axis_i, axis_oddity)
        if axis_oddity > 0:
            # If everything is false, then we pass. Otherwise, proceed.
            # Pick the correct axis index by axis_inds[coords[:,i]].
            joint_prob = np.sum(total_prob, axis=tuple(axis_i))
            if axis_oddity%2==1:
                # If there are odd number of axis taken, then we're supposed to add H(...).
                # But, because H() is defined as -()*log(), we subtract them instead.
                total_info -= np.sum(joint_prob * np.log(joint_prob))
            else:
                total_info += np.sum(joint_prob * np.log(joint_prob))
    return total_info
    
def distance_to_diagonal(X, tau, dim):
    # DD criterion by Simon & Verleysen (2006).
    # Expects 2D input in the shape of (Nfeatures, Nsamples). Anything else could cause errors.
    # Expects full state trajectory history as input. Input should also give embedding dimension.
    # For a given delay value tau, obtains (Nsamples-dimension+1) samples and finds distance to main diagonal.
    # In other words, form trajectory history data:
    #     { d0=[X[0],X[tau],...,X[tau*(dim-1)]], d1=[X[1],X[tau+1],...,X[tau*(dim-1)+1]], ... },
    # and then find the sum of distances between the one-vector [1,1,...,1] and all the points d_i.
    # This distance can be seen as the difference between d_i and d_i's projection onto the one-vector.
    # Bt Py---'s Thm., each point's squared distance to the main diagonal is the same as the difference between
    # the squared distance from origin to the point, and the squared distance from origin to point of projection. 
    # Notice that the second distance can be found as the dot product between [1,1,...,1] and the datapoint.
    # The returned value is the !!negative distance!!, because we want the caller to find a !local minimum!,
    # but the DD criterion would choose the first local maximum. I also normalize it by # of vectors added.
    
    # Not sure if this method could be applied to higher-dimension... 
    # I'm not sure how to do this for higher dimension. 
    # Should I make it into a (dim*Nfeatures,) vector? Or how should I design a distance measure
    # to evaluate how homogeneous the values within d_i are? 
    # Right now I'm just evaluating it for each variable inside X, and then summing them up. This disregards any
    # interactions between variables, but would be the same way employed by several other methods (e.g. Cao et al 1998's FNN)
    # E.g. if X = [X1, X2], then I evaluate the DD for X1 and X2 separately. 
    Xd = framing_helper(X, dim, interv=1, stride=tau, axis=1) # Shape = (Nframes, Nfeatures, frame_length=dim)
    # Use equation (6) to calculate the value.

    # Explanation of the one-liner below:
    # np.sum(np.square(Xd)) calculates the sum of element-wise square of Xd out of convenience. To make more sense to it,
    #   break it down as the sum of each point's squared distance from the origin. Each point's variable is represented as 
    #   the vector along axis 2 - a dim-dimensional embedding using uniform delay.
    # np.sum(np.square(np.sum(Xd, axis=2))) calculates the squared distance from origin to each projection point on
    #   the main diagonal. Inside, np.sum(Xd, axis=2) computes the dot product with [1,1,...,1] for each point's each variable,
    #   and the np.square() makes that distance squared. The np.sum() on the outside just sums everything up for
    #   convenience; if you want clarity, you can use the commented-out line below instead:
#   #  return np.sum( np.sum(np.square(Xd), axis=2) - np.square(np.sum(Xd, axis=2)) )
#     print(tau, dim)
#     print(np.sum(np.square(Xd),axis=2), np.square(np.sum(Xd, axis=2)))
    return -( np.sum(np.square(Xd)) - np.sum(np.square(np.sum(Xd, axis=2)))/dim ) / Xd.shape[0]

def average_displacement(X, tau, dim):
    # Similar to distance_to_diagonal criterion. 
    # In Rosenstein et al. 1993, it keeps calculating distance from the original no-delay point,
    # using things like \sum{|| [X[0], X[tau], ...] - [X[0], X[0], X[0], ...] ||}. 
    # Here we try to address high-dim case (dim of X > 1), so we just calculate the distance between X(n) and X(n+tau) instead?
    # Proposed method: \sum_i{ \sum_j{ || X[i+tau*j] - X[i] || } }
    Xd = framing_helper(X, dim, interv=1, stride=tau, axis=1)
    return np.sum( np.linalg.norm( Xd - Xd[:,:,[0]], axis=1 ) ) / Xd.shape[0]
    
def MMI_helper(X, Y, method=None, Nbin=50, ep=1e-10, pairwise=False, tau=1, dim=1):
    # Helper function for finding mutual information (or similar values) for higher dimension data.
    # Assumes that both X and Y are 2D and have the same shape. 
    # X, Y, method, Nbin, ep are standard input arguments.
    # tau and dim are for DD only.
    
    # If pairwise = True, then we're supposed to run each row of X against the entire Y,
    # and return the sum of all the correlation quantities.
    # This approach makes more sense than finding correlation of vstack((X,Y)), but lacks theoretical proof.
    if pairwise:
        total_sum = 0
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                total_sum += MMI_helper(X[[i],:], Y[[j],:], method=method, Nbin=Nbin, ep=ep)
        return total_sum

    # Else, risk stacking two arrays together and run one of the following methods: 
    if method == 'KL' or method == 'total correlation': # Recorded by Kraskov et al. 2004 etc. - easy extension from 2D AMI
        return KL_divergence(np.vstack((X,Y)), Nbin=Nbin, ep=ep, max_total=False)
    elif method == 'max total entropy':
        return KL_divergence(np.vstack((X,Y)), Nbin=Nbin, ep=ep, max_total=True)
    elif method == 'total divergence' or method == 'KL total':
        # I should consider allowing user to specify which one has larger grid later.
        return KL_total_divergence(X, Y, Nbin=Nbin, ep=ep, Ywider=(np.var(Y)>np.var(X)))
    elif method == 'interaction':
        return interaction_information(np.vstack((X,Y)), Nbin=Nbin, ep=ep)
    elif method == 'DD':
        return distance_to_diagonal(X, tau, dim)
    elif method == 'AD':
        return average_displacement(X, tau, dim)

def iterate_for_local_min(data, max_iter=20, method=None, Nbin=50, ep=1e-10, start_iter=1, dim=5, 
                          pairwise=False, end_early=True, verbose=False):
    # For multi-dimensional data.
    # Applies for methods that iterate through different delay choices to get the first local minimum. 
    # 'method' decides which one of the above MI quantities are to be used. 
#     if verbose:
#         print('shape is ', data.shape)
    if start_iter <= 1:
        prev_cor = np.Inf
    else:
        # This step is mainly for verifying if the first one is truly a local minimum.
        t = start_iter-1
        x1 = data[:,:-t]
        x2 = data[:,t:]
        if x1.shape[0] == 1 and x2.shape[0] == 1:
            prev_cor = AMI(x1, x2, method=method, Nbin=Nbin, ep=ep)
        else:
            prev_cor = MMI_helper(x1, x2, method=method, Nbin=Nbin, ep=ep, pairwise=pairwise, tau=t, dim=dim)
    local_min_found = False
    had_decrease = False
    local_min_delay = 0

    for t in range(start_iter, max_iter+1):
        # I need to allow the user to specify how to stack delayed data with normal data...
        x1 = data[:,:-t]
        x2 = data[:,t:]
        if x1.shape[0] == 1 and x2.shape[0] == 1:
            curr_cor = AMI(x1, x2, method=method, Nbin=Nbin, ep=ep)
        else:
            curr_cor = MMI_helper(x1, x2, method=method, Nbin=Nbin, ep=ep, pairwise=pairwise, tau=t, dim=dim)

        # Check if the candidate local minimum delay actually qualifies
#         print(curr_cor, prev_cor)
        if curr_cor >= prev_cor and had_decrease:
            local_min_found = True
        # If local min is not found, then we update the current candidate for a local minimum delay
        if curr_cor < prev_cor and not local_min_found:
            local_min_delay = t
        if curr_cor < prev_cor:
            had_decrease = True

        prev_cor = curr_cor
        # Optional termination if you don't want to see how the MI value turns out later
        if verbose:
            print('At delay = {0}, the AMI is {1}.'.format(t, curr_cor))
            if local_min_found:
                print('{0} is the optimal delay it found.'.format(local_min_delay))
        if local_min_found and end_early:
            print('---Ending AMI/MMI local min now---')
            break
    if not local_min_found:
        print('---AMI/MMI didn\'t find a local minimum within {0} delays'.format(max_iter))
    return local_min_delay, local_min_found

def count_FNN_helper(X, ratio, js, ttau, j, t, pred, irange, istart=0, verbose=False, method='garcia'):
    # Helper method for finding FNN value using Garcia method.
    # Arguments: 
    # X - input dataset. Cropped to only reveal the available part (from "init_i" in caller method to end)
    # js - list of fixed variable indices for each delay
    # ttau - list of fixed total delay timesteps for each delay; corresponds to each entry in js, and doesn't need to be ordered.
    # j - tentative delay variable index selection
    # t - tentative delay timestep choice (absolute value)
    # pred - number of prediction steps
    # irange - end index + 1 of available samples
    #          (Used to represent total number of samples to be used in finding the FNN. Not any more.)
    # Outputs:
    # falseNN_count
    # total_p_count - number of data points
    # (NN_d1, NN_d2) (distances between nearest neighbors in current and prediction)
    if method == 'garcia':
        X2d = np.vstack([ np.concatenate(( X[:,i], # [X[0,i], X[1,i], ..., X[k-1,i]]
                                      [X[js[jind], i+ttau[jind+1]] for jind in range(len(js))], # X[j1,i+tau1],...
                                      [X[j,i+t]] # The next delay to be determined
                                     )) for i in range(istart, irange) ]) #range(init_i, irange+init_i) ])
    else:
        # In the standard Kennel method, don't include the future unadded dimension yet.
        X2d = np.vstack([ np.concatenate(( X[:,i], # [X[0,i], X[1,i], ..., X[k-1,i]]
                                      [X[js[jind], i+ttau[jind+1]] for jind in range(len(js))] # X[j1,i+tau1],...
                                     )) for i in range(istart, irange) ]) 
    # Run Nearest-Neighbor algorithms on rows. Store NN index information as well as d1.
    X_NN = NearestNeighbors(n_neighbors=2).fit(X2d)
    NN_d1, NN_inds = X_NN.kneighbors(X2d)
    NN_d1 = NN_d1[:,1] # Exclude the point themselves (which have distance 0 and locate in 1st col)

    if method == 'garcia':
        # Calculate distances between neighbors (d2) after adding the prediction value to 2D.
        # I don't think I'm going to use Kennel's method of only comparing the last value.
        # Anyways, that one is just a special case of G&A's multivariate nonuniform embedding. 
        # If we're using the Garcia method, then we move all samples one pred forward, 
        # and then check if the neighbors are still neighbors.
        if pred <= 0:
            pred = 1
        X2d_pred = np.vstack([ np.concatenate((  X[:,i], # [X[0,i], X[1,i], ..., X[k-1,i]]
                                                [X[js[jind], i+ttau[jind+1]+pred] for jind in range(len(js))], # X[j1,i+tau1],...
                                                [X[j,i+t+pred]] # The next delay to be determined
                                            )) for i in range(istart, irange) ]) #range(init_i, irange) ])
        # Use spicy indexing methods to get the corresponding neighbor's location after pred.
        NN_d2 = np.linalg.norm( X2d_pred - X2d_pred[ NN_inds[:,1] ], axis=1 )
    else:
        # The Kennel method - Find neighbors' distances in the augmented space. 
        # We don't need to recreate the entire augmented thing, because only the additional dimension's position mattered.
        X2d_aug = np.vstack([X[j,i+t] for i in range(istart, irange) ]) 
        NN_d2 = np.abs(X2d_aug - X2d_aug[ NN_inds[:,1] ])
    # Count the number of false neighbors. Store the FNN count ratio (known as N in paper).
    falseNN_count = ((NN_d2 / NN_d1) > ratio).sum()
    total_p_count = X2d.shape[0]
    
    if verbose:
        print('At delay={0} using j={1} for de={2}: average d1 = {3}; average d2 = {4}'.format(
                        t, j, len(js), np.mean(NN_d1), np.mean(NN_d2)))
        print('{0} out of {1} samples counted as false neighbors'.format(falseNN_count, total_p_count))
        if np.abs(np.mean(NN_d1) - np.mean(NN_d2)) < 1e-9:
            print('Maybe these distances are not right here. Their means are too close.')
#             print(np.array([NN_d1,NN_d2]).T)
#             print(NN_inds)
#             print(np.hstack((X2d, X2d_pred)))
    return falseNN_count, total_p_count, (NN_d1, NN_d2)

def find_2D_local_min_helper(grid, twoD=True, verbose=False):
    # Helper method for identifying local minimum. For this reason, this method is extremely specific for FNN.
    # So specific that it is supposed to be called iteratively - it only checks the newest columns of the input grid,
    # and that it assumes the first column is full of fillers and shouldn't be ever selected (thus the first if-clause).
    # Arguments:
    # grid - stores the data grid where we want to find the local minimum. Shape is (Nvariables, Ndelays)
    # twoD - whether this method should care about neighboring vectors or not.
    #        If twoD is False, then this method only examines along one direction, and returns the earliest local min.
    # Outputs:
    #  - list of local minimum location indices
    #  - whether a local minimum is found or not
    
    # This method is part of a iterative procedure, so it automatically pursues the second-to-last column.
    if grid.shape[1] <= 2:
        return ([], False)
    if grid.shape[0] < 2:
        # Means it's a 1D array. 
        if grid[0, -2] < grid[0, -1] and (grid.shape[1]==2 or grid[0, -2] <= grid[0, -3]):
            return ([0, grid.shape[1]-2], True)
        return ([], False)
    # Step 1: Find local minimum for the col-th column. But don't forget the boundaries.
    if twoD:
        col_min_inds = argrelextrema(grid[:,-2], np.less, axis=0)[0].tolist()
        if grid[0,-2] < grid[1,-2]:
            col_min_inds += [0]
        if grid[-1,-2] < grid[-2,-2]:
            col_min_inds += [grid.shape[0]-1]
    else:
        col_min_inds = [i for i in range(grid.shape[0])]
    # Step 2: Check those local minima one-by-one and see if they're still local minima along the delay axis
    for ind in col_min_inds:
        if grid[ind, -2] < grid[ind, -1] and (grid.shape[1]==2 or grid[ind, -2] <= grid[ind, -3]):
            if verbose:
                print('Detected local minimum at location ', (ind, grid.shape[1]-2), '. Check for yourself:')
                print(grid)
            return ([ind, grid.shape[1]-2], True)
    # If nothing magical happened, then tell the caller to start the next iteration.
    return ([], False)

def find_delay_by_FNN_Garcia( X, ratio=10, pred=1, stop_threshold=0, min_tau=1, max_tau=100, max_dim=10, 
                              init_i=0, end_early=True, verbose=False, inverse=True, local_max=False, twoD=True):
    # Implements Garcia & Almeida's method (2004 & 2005).
    # Arguments:
    # X - input dataset; assuming to be 2D with shape (Nfeatures, Nsamples);
    # ratio - empirical distance ratio threshold for determining whether a pair of neighbor is false
    # pred - the amount of future timesteps to skip into when evaluating false neighbors;
    # stop_threshold - terminate when percentage of points with false neighbors fall below this value;
    # max_tau, max_dim - maximum values of delay and dimension. Terminate if exceeding it.
    # uniform_delay - flag for deciding if each delay is the same. (Doesn't do anything in this method)
    # init_i - the index for the very first column (!!unspecified in the paper!!). 
    # Outputs:
    # taus - delay value for each additional dimension. Its size is the final embedding dimension.
    # F_history - values of F in every delay embedding dimension
    de = 1
    js = [] # List for all variable numbers for the delays
    taus = [] # List for all the delays
    ttau = [0]  # Value of total delay
    FNNratio = 1
    F_history = []
    
    while (FNNratio > stop_threshold and de <= max_dim):
        # Outer loop. The number of iterations here determines the resulting dimension.
        local_min_found = False
        local_min_delay = []
        local_min_NNval = 0
        prev_N = 2
        if verbose:
            print('Starting the {0}-th embedding cycle'.format(de))
        
#         if uniform_delay and len(taus) > 0:
#             # If the caller wants uniform delay values, then just copy the previous one.
#             # However, this option doesn't work well with G&A's method, because you'll have to choose a j as well.
#             # Thus, this option is disabled
#             taus.append(taus[-1])
#             if verbose:
#                 print('Keeping the same delay value {0}'.format(taus[-1]))
#         else:
        # Otherwise, run the process of selecting the next delay.
        # The original multivariate paper wanted to find both tau and j for embedding, where they
        # append the tau-delayed j-th variable to the vector for all variables.
#             (ignore the lines below)
#             # For now, I suppose we could simply choose the delay value without worrying about choosing a specific
#             # variable j each time... by attaching the entire vector at that delay, instead of attaching the single
#             # value of the j-th observation for all the vectors. Though this would lead to a high cost in training NN.
#             # But anyways, I still used the term "N_grid" to describe the array I used to document scores for finding
#             # the first local minimum. (Ended up not using it for computation time considerations)

        # Check if max_tau would exceed the input data size first
        if max_tau > X.shape[1] - pred - ttau[-1] -1:
            max_tau = X.shape[1] - pred - ttau[-1] -1
            stop_threshold = 0 # Hack the threshold value for fast loop termination... this value doesn't have other use
            print('Warning: Running out of data')
        # Create a place to store all the historical data, so that we can find the first minimum
        N_grid = np.zeros((X.shape[0], max_tau+1))
#         if local_max:
#             N_grid[:,:min_tau] = -np.ones((X.shape[0], min_tau)) 
#         else:
#             N_grid[:,:min_tau] = np.ones((X.shape[0], min_tau))
        N_grid[:,:min_tau] = np.ones((X.shape[0], min_tau)) # Fill the col with no delay by ones, to make it the biggest

        for t in range(min_tau, max_tau+1):
            # irange: The max index of any valid sample. Because, if using forward embedding, we can't use samples
            # that would exceed the available data range after adding the embedding dimensions and the predictions.
            irange = X.shape[1] - ttau[-1] - t - pred 
#             irange = X.shape[1] - ttau[-1] - max_tau - pred - 1

            # Check over all the observation variables
            for j in range(X.shape[0]):
                # FNN algorithm.
                # We stack every possible putative embedded vectors together.
                if inverse:
                    # Lines of logic: If we're doing backwards embedding, then the delay can be seen as negative.
                    # The pred argument doesn't change, because it's meant to serve as prediction. 
                    # The index range needs changing as well - the sample index can't start before the earliest delay.
                    falseNN_count, total_p_count, _ = count_FNN_helper(
                        X[:,init_i:], ratio, js, [-tt for tt in ttau], j, -t-ttau[-1], pred, 
                        X.shape[1]-pred-init_i, istart=ttau[-1]+t, verbose=verbose)
                else:
                    # In the forward embedding version, we use the samples from init_i to irange.
                    falseNN_count, total_p_count, _ = count_FNN_helper(
                        X[:,init_i:], ratio, js, ttau, j, t+ttau[-1], pred, irange-init_i, verbose=verbose)
                    
                curr_N = falseNN_count / total_p_count
                N_grid[j, t] = curr_N

            # Every time we complete another delay value t, try to see if a local minimum location (j, tau)
            # has shown up for N. I assume this is what they mean the "first local minimum".
            if local_max:
                (local_min_delay_ind, local_min_found_here) = find_2D_local_min_helper(-N_grid[:,:t+1], twoD, verbose=verbose)
            else:
                (local_min_delay_ind, local_min_found_here) = find_2D_local_min_helper(N_grid[:,:t+1], twoD, verbose=verbose)
            if local_min_found_here and not local_min_found:
                local_min_delay = local_min_delay_ind
#                 js.append(local_min_delay_ind[0])
#                 taus.append(local_min_delay_ind[1])
                local_min_NNval = N_grid[local_min_delay_ind[0], local_min_delay_ind[1]]
                local_min_found = True
#                 print('Found local min at {0}. local_min_found_here={1}. local_min_found={2}.'.format(
#                 local_min_delay, local_min_found_here, local_min_found))
                if end_early:
                    break
        if not local_min_found:
            print('Warning: Did not find a proper local minimum. Terminating the cycle.')
            break
        js.append(local_min_delay[0])
        taus.append(local_min_delay[1])
        if verbose:
            print(N_grid[:,:t+1].T)
            print('Found first local min at j={0}, tau={1} as the {2}-th delay'.format(js[-1], taus[-1], de))
        
        # Other housekeeping things to do after determining the most recent delay:
        # 1. Calculate the FNN ratio (known as F in the paper)
        # 2. Update or create any helper variables
        FNNratio = local_min_NNval
        F_history.append(FNNratio)
        if verbose:
            print('Updated F value from the {0}-th embedding cycle is {1}'.format(de, FNNratio))
        ttau.append(taus[-1] + ttau[-1])
        de += 1
    if verbose:
        print('The final list of delay variables and delay values:')
        print(np.array([js, taus]))
    return js, taus, F_history

def find_delay_by_FNN_Cao( X, pred=1, min_tau=1, max_tau=100, max_dim=10, uniform_delay=True, 
                              init_i=0, end_early=True, verbose=False):
    # Implements the method inspired by Cao et al. (1998).
    # In the original paper, the delay value is determined independently for each variable, and each variable uses
    # uniform delay embedding. To acheive this, set uniform_delay as True.
    # We modified this approach by allowing non-uniform delay to happen, even though doing this might not make much
    # sense. If uniform_delay is set as False, then after one variable is chosen as the next embedding variable,
    # we find the next local minimum (instead of using the first local minimum * 2) as the next delay. Can this
    # take be justified? I'm not sure. But more options is always better. 
    # One further modification: The original paper searches over a large range of possible dimensions (Nfeatures^max_dim).
    # We can't usually afford that, so I turned to a greedy approach instead. I iteratively choose the best addition, and
    # terminate the algorithm when more dimensions can't bring better results. Well, this is the case when end_early=True.
    # Else, when end_early=False, this implementation greedily seeks more additional dimensions until the dimension
    # exceeds max_dim, and finds the minimum among all the errors. 
    # Arguments:
    # X - input dataset; assuming to be 2D with shape (Nfeatures, Nsamples);
    # pred - the amount of future timesteps to skip into when evaluating false neighbors;
    # max_tau, max_dim - maximum values of delay and dimension. Terminate if exceeding it.
    # uniform_delay - See descriptions above.
    # end_early     - See descriptions above.
    # Outputs:
    # taus - delay value for each additional dimension. Different from G&A, this one is a 2D int array,
    #        because it has to document information for each variable.
    
    # Notice that one difference between our implementation of G&A and Cao's FNN is that Cao's doesn't use cumulative delays.
    # In G&A, the delay values are monotonically increasing, even though they correspond to different variables.
    # In Cao, each variable has its set of monotonically increasing delay values.
    
    # To initialize, define some variables, and find the first best delays of each variable.
    de = 1
    js = [] # List for all variable numbers for the delays
    ttau = [0] # Value of total delay (That 0 here is a placeholder; it won't be used, but is here for compatibility)
    taus = np.zeros((X.shape[0], max_dim), dtype=int) # List for all the delays
    prev_err = 0
    curr_err = -1
    min_err = np.inf
    
    err_list = [] # Stores the total error for each variable if the variable were to be used as the next dimension
    delayinds = [] # Index of the latest delay added to each variable (i.e. its # of the current candidate)
    max_delays = [] # Maximum acceptable delay value for each variable
    best_delays = [] # Stores the best currently available delay value for each variable
    if verbose:
        print('--- KNN (Cao 1998) initializing itself ---')
    for j in range(X.shape[0]):
        # For each variable, first check what's the maximum possible delay it could afford, and then find the
        # optimal delay for that variable alone.
        # If no delay could be found, then mark that delay value as 0. Let the loop below check out for that.
        delayinds.append(0)
        max_delays.append( max_tau )
        (delay, found) = iterate_for_local_min(X[[j], :], max_iter=max_delays[-1], start_iter=min_tau, dim=1, verbose=verbose)
        if found:
            best_delays.append(delay)
            # Also find the error caused by the embedding here
            _, total_p_count, dists = count_FNN_helper(
                        X, 10, js, ttau, j, delay, pred, X.shape[1]-max_delays[-1]-pred-1, verbose=verbose)
            err_list.append( np.mean(dists[1]) )
        else:
            best_delays.append(0)
            err_list.append(np.inf)
    if verbose:
        print('Obtained a bunch of initial best delays here:')
        print(best_delays) 
        print(err_list)
        print('--- Entering main loop to find more delays ---')
    '''iterate_for_local_min(data, max_iter=20, method=None, Nbin=50, ep=1e-10, start_iter=1, dim=5, 
                          pairwise=False, end_early=True, verbose=False)'''
    min_err_val_snapshot = (js[:], ttau[:], np.copy(taus))
    # This next loop chooses one delay for each iteration. 
    while (de <= max_dim):
        # Select the best delay out of all variables.
        prev_err = curr_err
        ind = np.argmin(err_list)
        curr_err = err_list[ind]
        if curr_err >= np.inf:
            print('Warning: Running out of data. No delay available.')
            break
        # Update all the marker variables we've been using
        js.append(ind)
        ttau.append(best_delays[ind])
        taus[ ind, delayinds[ind] ] = best_delays[ind]
        delayinds[ind] += 1
        if verbose:
            print('Delay = {0} at the {1}-th variable was chosen. It brought the total FNN error to {2}'.format(
                            best_delays[ind], ind, curr_err))
        
        if curr_err < min_err:
            min_err = curr_err
            min_err_val_snapshot = (js[:], ttau[:], np.copy(taus))
            
        # Replenish the selected variable's newest best delay value. If it can't afford a delay anymore (due to either
        # accumulative delay exceeding max_tau or not enough data to verify the next delay), reflect that in max_delays.
        if uniform_delay:
            best_delays[ind] += taus[ind, 0] # Because the uniform delay value is stored in the first column of taus
            if best_delays[ind] > max_delays[ind]:
                best_delays[ind] = 0
                err_list[ind] = np.inf
            else:
                delay = best_delays[ind]
                _, total_p_count, dists = count_FNN_helper(
                            X, 10, js, ttau, ind, delay, pred, X.shape[1]-max(np.amax(ttau),delay)-pred-1, verbose=verbose)
                err_list[ind] = np.mean(dists[1])
            if verbose:
                print('The next available delay for the {0}-th variable is {1} = {2}*{3}'.format(
                                        ind, best_delays[ind], delayinds[ind]+1, taus[ind, 0]))
        else:
            max_delays[ind] = min(max_tau, X.shape[1] - best_delays[ind] - pred - 1)
            (delay, found) = iterate_for_local_min(X[[ind], :], max_iter=max_delays[ind], 
                                                   start_iter=best_delays[ind]+1, dim=1, verbose=verbose)
            if found:
                best_delays[ind] = delay
                _, total_p_count, dists = count_FNN_helper(
                            X, 10, js, ttau, ind, delay, pred, X.shape[1]-max(np.amax(ttau),delay)-pred-1, verbose=verbose)
                err_list[ind] = np.mean(dists[1])
            else:
                best_delays[ind] = 0
                err_list[ind] = np.inf
            if verbose:
                if best_delays[ind] != 0:
                    print('We replenished the {0}-th variable with delay at {1} - this is its next best delay after {2}'.format(
                                        ind, best_delays[ind], ttau[-1]))
                else:
                    print('The {0}-th variable is out of available delays...'.format(ind))
        
        if de <= 1: # Quick hack to maintain loop logic; don't read too much into it.
            prev_err = curr_err+1
        de += 1
        
        if end_early and prev_err <= curr_err:
            break
            
    if verbose:
        print('The final list of delay variables and delay values:')
        print(js)
        print(ttau)
        print('To put it in table form:')
        print(taus)
    return min_err_val_snapshot
#     return js, ttau, taus

def find_delay_by_FNN_Kennel(X, ratio=10, pred=1, stop_threshold=0, min_tau=1, max_tau=100, max_dim=10, uniform_delay=True, 
                              init_i=0, end_early=True, inverse=True, verbose=False):
    # Kennel version, but not the original version demonstrated in the original 1992 paper.
    # Our modifications:
    # 1. To select a delay value, the original paper used AMI and presumably only used the same value throughout. 
    #    If uniform_delay==True, then we do that (assuming we treat each variable independently). 
    #    Else, we use the next delay that gives another AMI local minimum. (Alternatively, we can test around
    #    different values at each dimension, and select the one that gives the first local minimum total error
    #    when doing "pred"-step prediction, just like in Cao et al. (Not sure if this will work, tho). But note
    #    that if we do this, we might have to re-calculate the FNN for every delay every time a new one is chosen.)
    #    (Of course, this can be done by the G&A method as well, but I just want to reduce # of hyperparameters.)
    # 2. To deal with multivariate observations, we use G&A's idea of picking over which variable to choose as the
    #    next embedded variable. For each candidate addition, we count the number of FNNs with this additional dimension
    #    (just like what Kennel does). We greedily choose the candidate that minimizes the number/ratio of FNN points,
    #    because our goal is to reduce the number of FNNs to 0. If any candidate makes it to zero, then we're done.
    # The method terminates when max_dim is reached, or when FNN ratio is smaller than stop_threshold.
    # Notice that the method from the original paper is implemented when uniform_delay = True, and Nfeatures = 1.
    # Ideally, this method could be combined with the Cao method... they have the same logic flow.
    # 
    # Arguments:
    # X - input dataset; assuming to be 2D with shape (Nfeatures, Nsamples);
    # ratio - empirical distance ratio threshold for determining whether a pair of neighbor is false
    # stop_threshold - terminate when percentage of points with false neighbors fall below this value;
    # max_tau, max_dim - maximum values of delay and dimension. Terminate if exceeding it.
    # uniform_delay - flag for deciding if each delay is the same. 
    # init_i - the index for the very first sample that we'll use (!!unspecified in the paper!!). 
    # end_early - Whether to teminate the function when the FNN ratio reaches stop threshold. 
    # Outputs:
    # js - index of each choice of embedded observation variables
    # taus - delay value for each additional dimension (matches with js). Its size is the final embedding dimension.
    # F_history - values of F in every delay embedding dimension
    de = 0
    js = [] # List for all variable numbers for the delays
    taus = np.zeros((X.shape[0], max_dim), dtype=int) # List for all the delays; makes it compatible with delay_embed_Cao.
    ttau = [0]  # Value of total delay
    prev_N = 1.1
    curr_N = 1
    min_N = 1.1
    
    delayinds = [] # Index of the latest delay added to each variable (i.e. its # of the current candidate)
    max_delays = [] # Maximum acceptable delay value for each variable
    best_delays = [] # Stores the best currently available delay value for each variable
    
    if verbose:
        print('--- KNN (Kennel 1992) initializing itself ---')
    # Find the first best delays
    for j in range(X.shape[0]):
        # For each variable, first check what's the maximum possible delay it could afford, and then find the
        # optimal delay for that variable alone.
        # If no delay could be found, then mark that delay value as 0. Let the loop below check out for that.
        delayinds.append(0)
        max_delays.append( max_tau )
        (delay, found) = iterate_for_local_min(X[[j], :], max_iter=max_delays[-1], start_iter=min_tau, dim=1, verbose=verbose)
        if found:
            best_delays.append(delay)
        else:
            best_delays.append(0)
    if verbose:
        print('Obtained a bunch of initial best delays here:')
        print(best_delays) 
        print('--- Entering main loop to find more delays ---')
        
    min_val_snapshot = (js[:], ttau[:], np.copy(taus))
    while (de <= max_dim):
        
        FNN_list = []
        # Check over all the observation variables
        for j in range(X.shape[0]):
            # FNN algorithm.
            # We stack every possible putative embedded vectors together.
            t = best_delays[j]
            if t <= 0:
                FNN_list.append(1)
            else:
                if inverse:
                    istart = ttau[-1]+t
                    irange = X.shape[1]-pred
                    falseNN_count, total_p_count, _ = count_FNN_helper(
                        X[:,init_i:], ratio, js, [-tt for tt in ttau], j, -t, pred, 
                        irange-init_i, istart=istart, verbose=verbose)
                else:
                    istart = 0
                    irange = X.shape[1] - ttau[-1] - t - pred 
                    falseNN_count, total_p_count, _ = count_FNN_helper(
                        X[:,init_i:], ratio, js, ttau, j, t, pred, irange-init_i, verbose=verbose)

                FNN_list.append(falseNN_count / total_p_count)
                
        ind = np.argmin(FNN_list)
        curr_N = FNN_list[ind]
        # The only difference between Cao's logic flow and this one's is that this one stops when N reaches 0,
        # without caring what the next embedding dimension would look like. 
        # Thus, "de" would naturally have one more value allowed, hence de starting at 0 instead of 1. 
        
        # Even if the algorithm doesn't stop until we're out of dimensions, moving this block before documenting
        # all the js and stuff assures that the max recoded dimension is still max_dim.
            
        if curr_N < stop_threshold and end_early:
            if verbose:
                print('FNN ratio is low enough. Ending it now with dimension {0}.'.format(de))
            break
        
        if curr_N >= 1:
            print('We\'re out of available delay variables. Exiting now.')
            break
            
        if de == max_dim:
            print('We\'ve exhausted the allowed dimension limit. Exiting now.')
            break
        
        # Update all the marker variables we've been using if we haven't reached break condition yet.
        if verbose:
            print(ind, best_delays[ind], delayinds[ind])
        js.append(ind)
        ttau.append(best_delays[ind])
        taus[ ind, delayinds[ind] ] = best_delays[ind]
        delayinds[ind] += 1
        if verbose:
            print('Delay = {0} at the {1}-th variable was chosen. It brought the total FNN ratio to {2}'.format(
                            best_delays[ind], ind, curr_N))
        
        if curr_N < min_N: # Preparation for if the FNN ratio N never reaches below stop_threshold.
            min_N = curr_N
            min_val_snapshot = (js[:], ttau[:], np.copy(taus))
            
        # Replenish the selected variable's newest best delay value. 
        if uniform_delay:
            best_delays[ind] += taus[ind, 0] # Because the uniform delay value is stored in the first column of taus
            if best_delays[ind] > max_delays[ind]:
                best_delays[ind] = 0
            else:
                delay = best_delays[ind]
            if verbose:
                print('The next available delay for the {0}-th variable is {1} = {2}*{3}'.format(
                                        ind, best_delays[ind], delayinds[ind]+1, taus[ind, 0]))
        else:
            max_delays[ind] = min(max_tau, X.shape[1] - best_delays[ind] - pred - 1)
            (delay, found) = iterate_for_local_min(X[[ind], :], max_iter=max_delays[ind], 
                                                   start_iter=best_delays[ind]+1, dim=1, verbose=verbose)
            if found:
                best_delays[ind] = delay
            else:
                best_delays[ind] = 0
            if verbose:
                if best_delays[ind] != 0:
                    print('We replenished the {0}-th variable with delay at {1} - this is its next best delay after {2}'.format(
                                        ind, best_delays[ind], ttau[-1]))
                else:
                    print('The {0}-th variable is out of available delays...'.format(ind))
        de += 1
        
    if verbose:
        print('The final list of delay variables and delay values:')
        print(min_val_snapshot[0])
        print(min_val_snapshot[1])
        print('To put it in table form:')
        print(min_val_snapshot[2])
    return min_val_snapshot


def find_delay_by_FNN(X, method='kennel', ratio=10, pred=1, stop_threshold=0, 
                      min_tau=1, max_tau=100, max_dim=10, uniform_delay=True, 
                      init_i=0, try_all_i=False, end_early=True, verbose=False,
                      inverse=True, local_max=False, twoD=False):
    # Wrapper method for all FNN stuff
    if method.lower() == 'garcia':
        return find_delay_by_FNN_Garcia(X, ratio=ratio, pred=pred, stop_threshold=stop_threshold, 
                                        min_tau=min_tau, max_tau=max_tau, max_dim=max_dim, init_i=init_i, 
                                        end_early=end_early, verbose=verbose, 
                                        inverse=inverse, local_max=local_max, twoD=twoD)
    elif method.lower() == 'cao':
        return find_delay_by_FNN_Cao( X, pred=pred, min_tau=min_tau, max_tau=max_tau, max_dim=max_dim, 
                                      uniform_delay=uniform_delay, init_i=init_i, 
                                      end_early=end_early, verbose=verbose)
    else:
        return find_delay_by_FNN_Kennel(X, ratio=ratio, pred=pred, stop_threshold=stop_threshold, 
                                        min_tau=min_tau, max_tau=max_tau, max_dim=max_dim, uniform_delay=uniform_delay, 
                                        init_i=init_i, end_early=end_early, inverse=inverse, verbose=verbose)
        
# The ultimate method for finding delay. Once and for all.
def find_delay(data, method='mi', MImethod='mir', FNNmethod='kennel', 
               min_delay=1, max_delay=20, max_dim=10, end_early=True, verbose=False, pairwise=False, 
               ratio=10, pred=1, stop_threshold=0, uniform_delay=True, init_i=0, inverse=True, local_max=True, twoD=False,
               Nbin=50, ep=1e-10   ):
    # Check the dimension of data first. If it's 1D data (or 2D data with only one variable), call AMI.
    if len(data.shape) == 1:
        data = data.reshape(1,-1)
    # Different evaluation method might need different procedures.
    if method == 'mi':
        # iterate_for_local_min() contains method that only uses mutual information or entropy or similar things.
        # If data is 1D, it calls AMI(), because other methods should be nearly equivalent.
        # If data is not 1D, it calls MMI_helper(), which would then use MImethod to decide which method to use.
        return iterate_for_local_min(data, max_iter=max_delay, method=MImethod, Nbin=Nbin, ep=ep, 
                                     start_iter=min_delay, dim=max_dim, 
                                     pairwise=pairwise, end_early=end_early, verbose=verbose)
        # Note: find_delay_from_MI() is phased out. You won't see it at your door anymore.
#         return find_delay_from_MI(data, max_delay=max_delay, method=MImethod, 
#                                   Nbin=Nbin, ep=ep, start_delay=start_delay, end_early=end_early, verbose=verbose)
    else:
        return find_delay_by_FNN(data, method=FNNmethod, ratio=ratio, pred=pred, stop_threshold=stop_threshold, 
                                 min_tau=min_delay, max_tau=max_delay, max_dim=max_dim, uniform_delay=uniform_delay, 
                                 init_i=0, try_all_i=False, end_early=end_early, verbose=verbose,
                                 inverse=inverse, local_max=local_max, twoD=twoD)






# Dump pile for unused / outdated / previous-version code

# def find_delay_from_MI(data, max_delay=50, method=None, Nbin=50, ep=1e-10, start_delay=1, dim=5, 
#                        pairwise=False, end_early=True, verbose=False):
#     # Future modification: Run this for each state in data, and return each state's mutual info.
#     if len(data.shape) == 1:
#         data = data.reshape(1,-1)
#     # Different evaluation method might need different procedures.
#     if method == 'FNN':
#         return find_delay_by_FNN(X, ratio=10, pred=1, stop_threshold=0, max_tau=max_delay, max_dim=dim, uniform_delay=True, 
#                       init_i=0, try_all_i=False, end_early=end_early, verbose=verbose)
#     else:
#         return iterate_for_local_min(data, max_iter=max_delay, method=method, Nbin=Nbin, ep=ep, 
#                                      start_iter=start_delay, dim=dim, 
#                                      pairwise=pairwise, end_early=end_early, verbose=verbose)

    # For higher dimension data, choose the corresponding method as provided. 
#     prev_cor = AMI(data, data, method=method, Nbin=Nbin)*100
#     local_min_found = False
#     local_min_delay = 0

#     for t in range(start_delay, max_delay+1):
#         # Currently only works with 1D data...
#         x1 = data[:,:-t]
#         x2 = data[:,t:]#,0]
#         curr_cor = AMI(x1, x2, method=method, Nbin=Nbin)

#         if curr_cor >= prev_cor:
#             local_min_found = True
#         if curr_cor < prev_cor and not local_min_found:
#             local_min_delay = t

#         prev_cor = curr_cor
#         # Optional termination if you don't want to see how the MI value turns out later
#         if verbose:
#             print('At delay = {0}, the AMI is {1}.'.format(t, curr_cor))
#             if local_min_found:
#                 print('{0} is the optimal delay it found.'.format(local_min_delay))
#         if local_min_found and end_early:
#             break
#     return local_min_delay, local_min_found

# def FNN_by_Garcia(...):
    # X2d = np.hstack([X[:,i+tau] for tau in ttau] + [X[:,ttau[-1]+t]])
#     de = 1
#     js = [] # List for all variable numbers for the delays
#     taus = [] # List for all the delays
#     ttau = [0]  # Value of total delay
#     FNNratio = 1
#     F_history = []
    
#     while (FNNratio > stop_threshold and de < max_dim):
#         # Outer loop. The number of iterations here determines the resulting dimension.
#         local_min_found = False
#         local_min_delay = 0
#         local_min_NNval = 0
#         prev_N = 2
#         if verbose:
#             print('Starting the {0}-th embedding cycle'.format(de))
        
#         if uniform_delay and len(taus) > 0:
#             # If the caller wants uniform delay values, then just copy the previous one.
#             taus.append(taus[-1])
#             if verbose:
#                 print('Keeping the same delay value {0}'.format(taus[-1]))
#         else:
#             # Otherwise, run the process of selecting the next delay.
#             # The original multivariate paper wanted to find both tau and j for embedding, where they
#             # append the tau-delayed j-th variable to the vector for all variables.
# #             (ignore the lines below)
# #             # For now, I suppose we could simply choose the delay value without worrying about choosing a specific
# #             # variable j each time... by attaching the entire vector at that delay, instead of attaching the single
# #             # value of the j-th observation for all the vectors. Though this would lead to a high cost in training NN.
# #             # But anyways, I still used the term "N_grid" to describe the array I used to document scores for finding
# #             # the first local minimum. (Ended up not using it for computation time considerations)
            
#             # Check if max_tau would exceed the input data size first
#             if max_tau > X.shape[1] - pred - ttau[-1] -1:
#                 max_tau = X.shape[1] - pred - ttau[-1] -1
#                 stop_threshold = 0 # Hack the threshold value for fast loop termination... this value doesn't have other use
#                 print('Warning: Running out of data')
#             # Create a place to store all the historical data, so that we can find the first minimum
#             N_grid = np.zeros((X.shape[0], max_tau+1))
#             N_grid[:,0] = np.ones(X.shape[0]) # Fill the col with no delay by ones, to make it the biggest
            
#             for t in range(1, max_tau+1):
#                 # One question is: Should I repeat the below procedures for all possible values of i? Or just one?
#                 # Temporary implementation: If try_all_i is false, then only use one possible initial value for i.
#                 #     Else, try all possible values of i, and count the total number of false neighbors.
#                 irange = (X.shape[1] - ttau[-1] - max_tau - pred - 1) if try_all_i else init_i+1
                
#                 # Check over all the observation variables
#                 for j in range(X.shape[0]):
#                     falseNN_count = 0
#                     total_p_count = 0 # Total pairs of neighbors we checked
# #                     for i in range(init_i, irange):
# #                         # Create 2D vectors.
# #                         # The vector looks like [ X[0,i], X[1,i], ..., X[k-1,i], X[j1,i+tau1], ... ]
# #                         X2d = np.hstack([X[:,i+tau] for tau in ttau] + [X[:,ttau[-1]+t]])
# #                         # Run Nearest-Neighbor algorithms on rows. Store NN index information as well as d1.
# #                         X_NN = NearestNeighbors(n_neighbors=2).fit(X2d)
# #                         NN_inds, NN_d1 = X_NN.kneighbors(X2d)
# #                         NN_d1 = NN_d1[:,1] # Exclude the point themselves (which have distance 0 and locate in 1st col)
# #                         # Calculate distances between neighbors (d2) after adding the prediction value to 2D.
# #                         X2d_pred = np.hstack([X[:,i+tau+pred] for tau in ttau] + [X[:,ttau[-1]+t+pred]])
# #                         #          np.hstack(( X[:,i+pred], X[:,i+t+pred] ))
# #                         # Use spicy indexing methods to get the corresponding neighbor's location after pred.
# #                         NN_d2 = np.linalg.norm( X2d_pred - X2d_pred[ NN_inds[:,1] ], axis=0 )
# #                         # Count the number of false neighbors. Store the FNN count ratio (known as N in paper).
# #                         falseNN_count += (NN_d2 / NN_d1 > ratio).sum()
# #                         total_p_count += X2d.shape[0]
#                     # Create the vector for FNN algorithm.
#                     # If combine_all_i is true, then we stack every possible putative embedded vectors together.
#                     # Otherwise, we go through a loop, each loop using only one possible i (unrecommended I guess).
#                     if combine_all_i:
# #                         X2d_pred = np.vstack([ np.hstack( 
# #                                             [X[:,i+tau+pred] for tau in ttau] + [[X[j,ttau[-1]+t+pred]] * X.shape[0]] 
# #                                                     ) for i in range(init_i, irange) ])
#                         irange = init_i+1
                    
#                     ### This thing is a mess right now. 
#                     ### We still need to put the entire vector [X[0,i], X[1,i], ..., X[k-1,i]] in the front, but right now it's nt there. 
#                     X2d = np.vstack([ np.hstack(  X[:,[i]].T
#                                                 + [X[j,i+tau] for tau in ttau]
#                                                 + [[X[j,ttau[-1]+t]] * X.shape[0]]) for i in range(init_i, ) ])
#                     for i in range(init_i, irange):
#                         # Create 2D vectors.
#                         if use_Garcia:
#                             # For each possible i, the vector looks like [ X[0,i], X[1,i], ..., X[k-1,i], X[j1,i+tau1], ... ].
#                             # If we combine all i, then the number of rows should be (Nsamples - largest_total_delay).
#                             # If we don't combine all i, then I don't know how should you even proceed to find neighbors.
#                             if not combine_all_i:
#                                 # If not combine_all_i, 
#                                 X2d = np.hstack(  X[:,[i]].T
#                                                 + [X[:,i+tau] for tau in ttau]
#                                                 + [[X[j,ttau[-1]+t]] * X.shape[0]])
#                             else:
#                                 X2d = np.vstack([ np.hstack( 
#                                                     [X[:,i+tau] for tau in ttau] + [[X[j,ttau[-1]+t]] * X.shape[0]] 
#                                                             ) for i in range(init_i, irange) ])
#                         else:
#                             # If we don't use the method in the Garcia paper, then we could probably just overlay delays.
#                             # I.e. generate stuff like [ X[:,i].T, X[:,i+tau1].T, ... ]
#                             if not combine_all_i:
#                                 X2d = np.hstack([X[:,i+tau] for tau in ttau] + [X[:,ttau[-1]+t]])
#                             else:
                                
#                         # Run Nearest-Neighbor algorithms on rows. Store NN index information as well as d1.
#                         X_NN = NearestNeighbors(n_neighbors=2).fit(X2d)
#                         NN_inds, NN_d1 = X_NN.kneighbors(X2d)
#                         NN_d1 = NN_d1[:,1] # Exclude the point themselves (which have distance 0 and locate in 1st col)
                        
#                         # Calculate distances between neighbors (d2) after adding the prediction value to 2D.
#                         if use_Garcia:
#                             # If we're using the Garcia method, then we move all samples one pred forward, 
#                             # and then check if the neighbors are still neighbors.
#                             if not combine_all_i:
#                                 X2d_pred = np.hstack([X[:,i+tau+pred] for tau in ttau] + [X[:,ttau[-1]+t+pred]])
#                             # Use spicy indexing methods to get the corresponding neighbor's location after pred.
#                             NN_d2 = np.linalg.norm( X2d_pred - X2d_pred[ NN_inds[:,1] ], axis=0 )
#                             # Count the number of false neighbors. Store the FNN count ratio (known as N in paper).
#                             falseNN_count += ((NN_d2 / NN_d1) > ratio).sum()
#                             total_p_count += X2d.shape[0]
#                         else:
#                             # If we're using the Kennel method, then we find the distance of samples one pred forward.
#                             # Because we're not using the Garcia & Almeida method, we would have different augmentations
                            
#                     # Find the first local minimum location (j, tau) for N.
#                     curr_N = falseNN_count / total_p_count
#                     N_grid[j, t] = curr_N
#                 if curr_N >= prev_N:
#                     local_min_found = True
#                     local_min_NNval = prev_N
#                 if curr_N < prev_N and not local_min_found:
#                     local_min_delay = t
#                 prev_N = curr_N
#                 if verbose:
#                     print('At delay = {0}, the N is {1}/{2}.'.format(t, falseNN_count, total_p_count))
#                 if local_min_found and end_early:
#                     break
#             if local_min_found:
#                 taus.append(local_min_delay)
#                 if verbose:
#                     print('Found first local min = {0} as the {1}-th delay'.format(local_min_delay, de))
#             else:
#                 print('Warning: Did not find a proper local minimum. Terminating the cycle.')
#                 break
        
#         # Other housekeeping things to do after determining the most recent delay:
#         # 1. Calculate the FNN ratio (known as F in the paper)
#         # 2. Update or create any helper variables
#         FNNratio = local_min_NNval
#         F_history.append(FNNratio)
#         if verbse:
#             print('Updated F value from the {0}-th embedding cycle is {1}'.format(de, FNNratio))
#         ttau.append(taus[-1] + ttau[-1])
#         de += 1
#     return taus, F_history

