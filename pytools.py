def rebin(x, bin_avg=2, error_prop=False):
    '''
    Rebin a numpyarray (if error_prop==True compute the error propagation of the rebbinning procedure)

    Input parameters:
        - x: numpy vector
        - bin_avg: number of point used for the average (default == 2)
        - error_prop: return the square root of the sum of the squares - for uncorrelated errors is equivalent to compute the error propagation (default == False)
    '''
    if bin_avg == 1:
        return x
    else:
        npoint_old = x.shape[0]
        off_set = npoint_old % bin_avg
        npoint_new = npoint_old//bin_avg
        if not error_prop:
            new_x = x[off_set:].reshape(npoint_new, bin_avg).mean(1)
        else:
            new_x = np.sqrt((x[off_set:].reshape(npoint_new, bin_avg)**2).sum(1))/bin_avg
        return new_x
  
def moving_avg(x, n=3, error_prop=False, symmetric=True):
    '''
    Mooving average of a numerical numpy array
    
    Input parameters:
        - x: numpy vector
        - n: number of point used for the average (default == 3)
        - error_prop: return the square root of the sum of the squares - for uncorrelated errors is equivalent to compute the error propagation (default == False)
        - symetric: execute average symmetrically - i.e. x_i = \frac{1}{2n + 1} sum_{k=i-n}^{i+n} x_k (default == True)
    '''
    if n != 0:
        if not symmetric:
            new_x = np.zeros(x.shape[0]-n+1)
            if not error_prop:
                for i in range(n-1):
                    new_x += x[i:-n+1+i]
                new_x += x[n-1:]
                new_x = new_x / n
            else:
                for i in range(n-1):
                    new_x += x[i:-n+1+i]**2
                new_x += x[n-1:]**2
                new_x = new_x**0.5 / n
            return new_x
        else:
            size = x.shape[0]
            new_x = np.zeros(size)
            if not error_prop:
                dn = 2*n+1
                for i in range(n, size-n):
                    for j in range(dn):
                        new_x[i] += x[i+n-j]
                    new_x[i] = new_x[i] / dn
                for i in range(n):
                    k = i+1
                    dn = 2*k+1
                    for j in range(dn):
                        new_x[i]  += x[ dn-j]
                        new_x[-k] += x[-dn+j]
                    new_x[i]  = new_x[i]  / dn
                    new_x[-k] = new_x[-k] / dn
            else:
                dn = 2*n+1
                for i in range(n, size-n):
                    for j in range(dn):
                        new_x[i] += x[i+n-j]**2
                    new_x[i] = new_x[i]**0.5 / (dn-1)
                for i in range(n):
                    k = i+1
                    dn = 2*k+1
                    for j in range(dn):
                        new_x[i]  += x[ dn-j]**2
                        new_x[-k] += x[-dn+j]**2
                    new_x[i]  = new_x[i]**0.5  / (dn-1)
                    new_x[-k] = new_x[-k]**0.5 / (dn-1)
            return new_x
    else:
        return copy.deepcopy(x)

  
def trapz_int(x, y, y_std, check_dx=False):
    '''
    Function used to calculate the integral of a function using the trapezoid rule and to compute the propagation of the errors 
    
    Input parameters:
        - x: numpy vector
        - y: numpy vector, values of the integrand
        - y_std: numpy vector, errors of the integrand
        - check_dx: check if x_{i+1}-x_{i} = dx for all the i < len(x)-1 (default==False)
    '''
    dx = x[1] - x[0]
    
    if check_dx:
        for k in range(2, x.shape[0]):
            if dx != x[k] - x[k-1]:
                raise Exception('ERROR! The lenght of the sub-intervals is not the same for every sub-interval.')
                
    i     = dx * (np.sum(y[1:-1]) + 0.5 * (y[0] + y[-1]))
    i_std = dx * np.sqrt(np.sum(y_std[1:-1]**2) + 0.25 * (y_std[0]**2 + y_std[-1]**2))
    
    return np.array([i, i_std])

def integrate(x, y, yerr, dataset_len=10000, num_int_algorithm='simps'):
    '''
    Function used to calculate the integral of a function and statistically estimate the propagated errors

    Input parameters:
        - x: numpy vector
        - y: numpy vector, values of the integrand
        - yerr: numpy vector, errors of the integrand
        - dataset_len: number of estimations (default == 10000)
        - num_int_algorithm: numberical algorithm used for the integration, known algorithms are: simpson (simps), and trapezoid (trap) (default == simps)
    '''
    dataset = np.random.normal(y, np.abs(yerr), (dataset_len, x.shape[0]))
    if num_int_algorithm == 'trapz':
        integrated = np.trapz(dataset, x, axis=1)
    elif num_int_algorithm == 'simps':
        integrated = simps(dataset, x, axis=1)
    else:
        raise Exception('ERROR! Unknown algorithm for numerical integration: "{}"'.format(num_int_algorithm))
    return integrated.mean(), integrated.std()

def weighted_mean(values, errors, excluded_points=[], return_error='none'):
    '''
    return_error: 
      - 'none' : return only the weighted average (no errors)
      - 'prop' : return the weighted average and the error due to the propagation of the errors of the values
      - 'stat' : return the weighted average and the error due to the statistical disperzsion of values
      - 'tot'  : return the weighted average and the total error - i.e. np.sqrt(propagation**2 + statistical**2)
      - 'all'  : return the weighted average and all the errors: 'prop', 'stat', and 'tot'
    '''

    ids = np.full(values.shape[0], True)
    if len(excluded_points) > 0:
        for i in excluded_points:
            ids[i] = False
    errs = errors[ids]
    vals = values[ids]

    weights = 1 / errs
    mean    = np.sum(weights * vals) / weights.sum()
    
    if return_error == 'none':
        return mean
    else:
        if return_error == 'tot' or 'all':
            perr = np.sqrt(np.sum((weights * errs)**2)) / weights.sum()
            serr = np.sqrt(np.sum((weights * (vals - mean))**2)) / weights.sum()
            terr = np.sqrt(perr**2 + serr**2)
            if return_error == 'tot':
                return mean, terr
            else:
                return mean, perr, serr, terr
        elif return_error == 'prop':
            perr = np.sqrt(np.sum((weights * vals)**2)) / weights.sum()
            return mean, perr
        elif return_error == 'stat':
            serr = np.sqrt(np.sum((weights * (vals - mean))**2)) / weights.sum()
            return mean, serr
        else:
            raise Exception('ERROR! Unkown command: \'{}\'.'.formaqt(return_error))
