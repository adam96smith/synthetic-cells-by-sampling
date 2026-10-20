import numpy as np
from scipy.ndimage import gaussian_filter
from utils.perlin_noise import *

''' FUNCTIONS '''

def next_power_of_two(x):
    """
    Compute the next power of 2 greater than or equal to x. Use this to 
    make sure the perlin noise generator works for any input shape.
    
    Args:
        x (int): Input value (must be >= 1).
        
    Returns:
        int: The next power of 2 greater than or equal to x.
    """
    if x < 1:
        raise ValueError("Input must be greater than or equal to 1.")
    
    # If x is already a power of 2, return x
    if (x & (x - 1)) == 0:
        return x

    # Compute the next power of 2
    return 1 << (x - 1).bit_length()

def npt_seq(X):
    output = []
    for x in X:
        output.append(next_power_of_two(x))
    return output
    

def speckle_transform(image, alpha, p, mask=None, sigma=1, sf = 2, per_instance=False, wavelengths=[[2,3,4],[2,3,4]]):
    '''
    Applies speckle scaling to image.

    image: 2-D or 3-D image array
    
    alpha: amplitude of positive scaling
    p: fraction of pixels to apply positive scaling
    sigma: smoothing parameter for scaled regions
    sf: speed up 3D perlin noise gen (default 2)
    per instance: Apply scaling per instance

    wavelengths: List of wavelengths to sample in z, and xy direction (eg. [[1,2], [3,4,5,6]])

    < within function >
    dI_0: amplitude of negative scaling *
    p0: fraction of pixels to apply negative scaling *

    * calculated such that net scaling = 1

    if mask is not None -> apply positive scaling to mask=1 regions only    
    '''
    
    assert (
        isinstance(wavelengths, list)
        and len(wavelengths) == 2
        and all(isinstance(i, list) and len(i) > 0 for i in wavelengths)
    )

    if mask is None:
        if per_instance:
            raise Exception('If no mask provided, per_instance cannot be True.')
        
    image = image.astype(np.float32)

    dI = alpha-1 # old formulation but consistent with paper
    
    # other parameters
    dI_0 = (p/(1-p))*dI
    
    if dI_0 > 1:
        # raise Exception(f'For dI = {dI}, p must be less than {1/(dI+1):.3f}. For p = {p}, dI must be less than {(1/p)-1:.3f}')
        raise Exception(f'For alpha = {alpha}, p must be less than {1/alpha:.3f}. For p = {p}, alpha must be less than {(1/p):.3f}')
    p0 = 1-p

    if per_instance: # apply foreground intensity scaling to each instance independently
        if image.ndim == 2:
            # negative regions 
            gp_0 = np.random.choice([2**i for i in wavelengths[-1]])
        
            x_n0 = generate_perlin_noise_2d(npt_seq(image.shape), [gp_0,gp_0])
            x_n0 =  x_n0[:image.shape[0],:image.shape[1]]
            x_n0 = (x_n0-x_n0.min())/(x_n0.max()-x_n0.min())
        
            # positive regions
            X_n1 = []
            for _ in np.unique(mask)[1:]:
                
                gp_1 = np.random.choice([2**i for i in wavelengths[-1]])
            
                x_n1 = generate_perlin_noise_2d(npt_seq(image.shape), [gp_1,gp_1])
                x_n1 =  x_n1[:image.shape[0],:image.shape[1]]
                x_n1 = (x_n1-x_n1.min())/(x_n1.max()-x_n1.min())

                X_n1.append(x_n1)
            
        elif image.ndim == 3:
    
            # negative regions 
            gp_0 = [np.random.choice([2**i for i in wavelengths[j] if 2**i < image.shape[j]/2]) for j in range(2)]
            gp_0.append(gp_0[-1])
    
            x_n0 = perlin_optim_3d(npt_seq(image.shape), gp_0, octaves=1, sf=sf, order=1)
            x_n0 =  x_n0[:image.shape[0],:image.shape[1],:image.shape[2]]
            x_n0 = (x_n0-x_n0.min())/(x_n0.max()-x_n0.min())
    
            # positive regions 
            X_n1 = []
            for _ in np.unique(mask)[1:]:
                
                gp_1 = [np.random.choice([2**i for i in wavelengths[j] if 2**i < image.shape[j]/2]) for j in range(2)]
                gp_1.append(gp_1[-1])
        
                x_n1 = perlin_optim_3d(npt_seq(image.shape), gp_1, octaves=1, sf=sf, order=1)
                x_n1 =  x_n1[:image.shape[0],:image.shape[1],:image.shape[2]]
                x_n1 = (x_n1-x_n1.min())/(x_n1.max()-x_n1.min())

                X_n1.append(x_n1)

        else:
            raise Exception('image dimentsions must be 2 or 3.')

        # sigmoid Perlin noise
        t0 = np.quantile(x_n0, p0) # get threshold so correct fraction decrease
        t1 = np.quantile(x_n1, 1-p) # get threshold so correct fraction increase
    
        S_0 = gaussian_filter((dI_0)*(x_n0>t0) + (1-dI_0), sigma=sigma)
        S_1 = [gaussian_filter((dI)*(x_n1>t1) + 1, sigma=sigma) for x_n1 in X_n1]
        
        # combine scaling (positive scaling at mask>0 only)
        S = S_0 + 1*(mask==0) - 1
        for v, s_1 in zip(np.unique(mask)[1:], S_1):
            S += s_1*(mask==v)
    
        im_bg = np.mean(image[mask==0])

        new_image = (image-im_bg)*S + im_bg
            

    else: # apply the same intensity scaling to all foreground regions
    
        if image.ndim == 2:
            # negative regions 
            gp_0 = np.random.choice([2**i for i in wavelengths[-1]])
        
            x_n0 = generate_perlin_noise_2d(npt_seq(image.shape), [gp_0,gp_0])
            x_n0 =  x_n0[:image.shape[0],:image.shape[1]]
            x_n0 = (x_n0-x_n0.min())/(x_n0.max()-x_n0.min())
        
            # positive regions
            gp_1 = np.random.choice([2**i for i in wavelengths[-1]])
            
            x_n1 = generate_perlin_noise_2d(npt_seq(image.shape), [gp_1,gp_1])
            x_n1 =  x_n1[:image.shape[0],:image.shape[1]]
            x_n1 = (x_n1-x_n1.min())/(x_n1.max()-x_n1.min())
            
        elif image.ndim == 3:
    
            # negative regions 
            gp_0 = [np.random.choice([2**i for i in wavelengths[j] if 2**i < image.shape[j]/2]) for j in range(2)]
            gp_0.append(gp_0[-1])
    
            x_n0 = perlin_optim_3d(npt_seq(image.shape), gp_0, octaves=1, sf=sf, order=1)
            x_n0 =  x_n0[:image.shape[0],:image.shape[1],:image.shape[2]]
            x_n0 = (x_n0-x_n0.min())/(x_n0.max()-x_n0.min())
    
            # positive regions 
            gp_1 = [np.random.choice([2**i for i in wavelengths[j] if 2**i < image.shape[j]/2]) for j in range(2)]
            gp_1.append(gp_1[-1])
    
            x_n1 = perlin_optim_3d(npt_seq(image.shape), gp_1, octaves=1, sf=sf, order=1)
            x_n1 =  x_n1[:image.shape[0],:image.shape[1],:image.shape[2]]
            x_n1 = (x_n1-x_n1.min())/(x_n1.max()-x_n1.min())
    
        else:
            raise Exception('image dimentsions must be 2 or 3.')
    
        # sigmoid Perlin noise
        t0 = np.quantile(x_n0, p0) # get threshold so correct fraction decrease
        t1 = np.quantile(x_n1, 1-p) # get threshold so correct fraction increase
    
        S_0 = gaussian_filter((dI_0)*(x_n0>t0) + (1-dI_0), sigma=sigma)
        S_1 = gaussian_filter((dI)*(x_n1>t1) + 1, sigma=sigma)
        
        if mask is not None:

            mask = mask.astype(bool)
            
            # combine scaling (positive scaling at mask>0 only)
            S = S_0 + S_1*mask + 1*(~mask) - 1
        
            im_bg = np.mean(image[~mask])
    
            new_image = (image-im_bg)*S + im_bg
    
        else:
    
            # combine scaling
            S = S_0 + S_1 - 1
    
            new_image = image*S

    return new_image