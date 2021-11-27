import numpy as np
import time
import os
import matplotlib
import matplotlib.pyplot as plt

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

            
def progress_bar(percent, output_line_format = '{progress_bar}', progress_char='->', bar_len = 20, new_line_after_intger_values=True):
    '''

    '''
    
    int_percent = int(percent)
    real_percent = percent - int_percent
    if percent != 0 and real_percent == 0:
        real_percent = 1
        
    if progress_char == '->':
        progress_chars = '-' * (int(bar_len * real_percent)-1)
        if real_percent < 1:
            progress_chars += '>'
        else:
            progress_chars += '-'
        bar = '[{:<{}}] {:.2f}%'.format(progress_chars, bar_len, percent * 100)
    else:
        bar = '[{:<{}}] {:.2f}%'.format(progress_char * int(bar_len * real_percent), bar_len, percent * 100)
    return output_line_format.format(progress_bar=bar)


def print_progress_bar(percent, output_line_format = '{progress_bar}', progress_char='->', bar_len = 20, new_line_after_intger_values=True):
    out = progress_bar(percent, output_line_format = output_line_format, progress_char = progress_char, bar_len = bar_len, new_line_after_intger_values=new_line_after_intger_values)
    
    sys.stdout.write('\r')
    sys.stdout.write(out)
    if new_line_after_intger_values and real_percent == int(real_percent) and percent != 0:
        sys.stdout.write('\n')
    sys.stdout.flush()


def pbar_with_timer(steps_done, n_steps, t_0, cmd_per_file=1, t_remaining = None):
    '''
    Example:
    
    ###########################################   code   ###########################################
    
    n_tests = 10
    test_iterations = 5
    test_duration   = 3.0

    n_total_iterations = n_tests * test_iterations
    dt_test = test_duration / test_iterations

    def execute_prosess(prosess):

        verbose = True

        counter_iter = 0
        t_0    = time.perf_counter()
        t_step = 0.0
        t_rem  = np.NaN

        for i_t in range(n_tests):
            for i_i in range(test_iterations):
                pname = 'Test_{:02d}-Iteration_{:02d}'.format(i_t, i_i)
                pbar, t_step, s_rem = ptl.pbar_with_timer(counter_iter, n_total_iterations, t_0)
                processing = 'Processing: \"{}\"'.format(pname)
                print('\r{:45} {}'.format(processing, pbar)+' '*40, end='')

                prosess

                counter_iter += 1
                if verbose:
                    print('\r{:3d}#  {}'.format(counter_iter, pname)+' '*160)
            if verbose:
                print('\r  ->  Executed: Test_{:02d}'.format(i_t)+' '*160+'\n')
            else:
                print('\r - Executed: Test_{:02d}'.format(i_t)+' '*160)

        pbar, t_step, s_rem = ptl.pbar_with_timer(counter_iter, n_total_iterations, t_0)
        print('PROCESS ENDED', pbar)
        
    ###########################################  output  ###########################################
      1#  Test_00-Iteration_00                                                                                                                                                           
      2#  Test_00-Iteration_01
      3#  Test_00-Iteration_02
      4#  Test_00-Iteration_03
      5#  Test_00-Iteration_04
      ->  Executed: Test_00                                                                                                                                                                

      6#  Test_01-Iteration_00
      7#  Test_01-Iteration_01
      8#  Test_01-Iteration_02
      9#  Test_01-Iteration_03
     10#  Test_01-Iteration_04
      ->  Executed: Test_01                                                                                                                                                                

     11#  Test_02-Iteration_00
     12#  Test_02-Iteration_01
     13#  Test_02-Iteration_02
    Processing: "Test_02-Iteration_03"            [------------>       ] 65.00%  -  Analyzed files:  13/ 20  -  Timer:     8 s (remaining     4 s)                                       
    '''
    t_step = time.perf_counter() - t_0
    if t_remaining == None:
        if steps_done:
            t_remaining = (float(n_steps) / steps_done - 1) * t_step
        else:
            t_remaining = np.NaN
    else:
        t_remaining = t_remaining - t_step
    timer = 'Timer: {:5.0f} s (remaining {:5.0f} s)'.format(t_step, t_remaining)
    file_num = 'Analyzed files: {:3d}/{:3d}'.format(int(steps_done/cmd_per_file), int(n_steps/cmd_per_file))
    pbar = progress_bar(steps_done/n_steps, output_line_format='{progress_bar}  -  '+file_num+'  -  '+timer)
    return pbar, t_step, t_remaining


def read_xvg(fname):
    '''
    Read xvg files
    '''
    data = []
    with open(fname, 'r') as fin:
        txt_lines = fin.readlines()
    for line in txt_lines:
        if line.find('#') == -1 and line.find('@') == -1:
        #if line[0] != '#' and line[0] != '@':
            data.append(line)
    
    x = np.empty(len(data), dtype=np.float32)
    y = np.empty(len(data), dtype=np.float32)
    for i in range(0, len(data)):
        x[i], y[i] = data[i].split()[0:2]
    return x, y


def scientific_notation_FLOAT(value, error, verbose=False):
    n = '{:e}'.format(error)
    if verbose:
        print('err:', n)
    idx = n.find('e')
    sgn_err = 1 if n[idx+1] == '+' else -1
    exp_err  = int(n[idx+2:])
    num_err  = int(np.round(float(n[:idx])))

    n = '{:e}'.format(value)
    if verbose:
        print('val:', n)
    idx = n.find('e')
    sgn_val = 1 if n[idx+1] == '+' else -1
    exp_val  = int(n[idx+2:])
    diff_exp = sgn_val * exp_val - sgn_err * exp_err
    num_val  = np.round(float(n[:idx]) * (10**diff_exp))

    if sgn_err < 0:
        num_err = np.round(num_err * (10**-exp_err), exp_err)
        num_val = np.round(num_val * (10**-exp_err), exp_err)

    if verbose:
        if exp_err != 0:
            print('ALL: ({} +- {}) · 10^({})'.format(num_val, num_err, sgn_err*exp_err))
        else:
            print('ALL: ({} +- {})'.format(num_val, num_err))
    return num_val, num_err, sgn_err*exp_err


def add_subplot_axes(ax, rect, facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)
    #x_labelsize = subax.get_xticklabels()[0].get_size()
    #y_labelsize = subax.get_yticklabels()[0].get_size()
    #x_labelsize *= rect[2]**0.5
    #y_labelsize *= rect[3]**0.5
    #subax.xaxis.set_tick_params(labelsize=x_labelsize)
    #subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def get_filenames(path, ext='', start='', keywords=[], exclude=[], sort=True, return_fnames_without_ext=False):
    fnames = []

    for f in os.listdir(path):

        if os.path.isfile(os.path.join(path, f)) and f[0] != ".":

            correct_ext     = False
            correct_start   = False
            correct_kwords  = False
            correct_exclude = True

            fdot = len(f) - f[::-1].find(".") - 1
            fext = f[fdot+1:]

            if fext == ext or ext == '':
                correct_ext = True

            if start != '':
                if start == f[:len(start)]:
                    correct_start = True
            else:
                correct_start = True

            kw_counter = 0
            for kword in keywords:
                if kword in f:
                    kw_counter += 1
            if kw_counter == len(keywords):
                correct_kwords = True

            for ex in exclude:
                if ex in f:
                    correct_exclude = False
                    break

            if correct_ext and correct_start and correct_kwords and correct_exclude:
                if return_fnames_without_ext:
                    if fext != '':
                        fnames.append(f[:fdot])
                    else:
                        if f[:fdot] not in fnames:
                            fnames.append(f[:fdot])
                else:
                    fnames.append(f)
    if sort:
        list.sort(fnames)

    return fnames

def get_filetypes(fnames):
    ftypes = dict()
    for f in fnames:
        fdot = len(f)-f[::-1].find('.')
        fname = f[:fdot-1]
        fext = f[fdot:]
        if fname not in ftypes:
            ftypes[fname] = [fext]
        else:
            ftypes[fname].append(fext)
    return ftypes