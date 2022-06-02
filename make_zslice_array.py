import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import straxen
from multihist import Histdd, Hist1d
from matplotlib.colors import LogNorm
import cutax
# Print out exactly what versions we are using, for reference / troubleshooting:
import sys
import os.path as osp
import scipy.signal
print(f"Python {sys.version} at {sys.executable}\n"
      f"Straxen {straxen.__version__} at {osp.dirname(straxen.__file__)}")
st = cutax.xenonnt_v8(cuts_for=['basic.s1_pattern', 'basic.s2_pattern', 'basic.s1_single_scatter', 'basic.s2_single_scatter', 'basic.s2_width', 'basic.s2_aft'])
import pandas as pd
runs = st.select_runs(#exclude_tags = ['missing_one_pmt,_sr0_preliminary,ambe_source_top_cw3d0', '_sr0_preliminary,ambe_source_top_far'],
                      available="event_info", 
                      run_mode="*ambe_link")
events = st.get_array(runs['name'], ['event_info', 'cut_s1_single_scatter', 'cut_s2_single_scatter','cut_s2_width'])
events_double_scatter = events[(events['cut_s2_width']) & (events['cut_s2_single_scatter'] == False)]
events_single_scatter = events[(events['cut_s2_width']) & (events['cut_s2_single_scatter'])]
events_double_scatter = events_double_scatter[(events_double_scatter['s2_area_fraction_top']>0.70) & (events_double_scatter['alt_s2_area_fraction_top']>0.7) & (events_double_scatter['s2_area_fraction_top']<0.78) & (events_double_scatter['alt_s2_area_fraction_top']<0.78) & (events_double_scatter['alt_cs2'] > 200)]
events_single_scatter = events_single_scatter[(events_single_scatter['s2_area_fraction_top']>0.70) & (events_single_scatter['alt_s2_area_fraction_top']>0.7) & (events_single_scatter['s2_area_fraction_top']<0.78) & (events_single_scatter['alt_s2_area_fraction_top']<0.78) & (events_single_scatter['alt_cs2'] > 200)]

new_dt = np.dtype(events.dtype.descr + [('time_difference', 'float')])
events_temp_double = np.zeros(events_double_scatter.shape, new_dt)
for i in events.dtype.names:
    events_temp_double[i] = events_double_scatter[i]
events_temp_double['time_difference'] = abs(events_temp_double['s2_time'] - events_temp_double['alt_s2_time'])
events_double_scatter = events_temp_double

events_temp_single = np.zeros(events_single_scatter.shape, new_dt)
for i in events.dtype.names:
    events_temp_single[i] = events_single_scatter[i]
events_temp_single['time_difference'] = abs(events_temp_single['s2_time'] - events_temp_single['alt_s2_time'])
events_single_scatter = events_temp_single

def dt_difference_hist_together(events, log = False, z_step = 400000, print_n=True):  
    x = events['time_difference']
    #xbins, ybins = (np.logspace(np.log10(0.1), np.log10(180), 100), 
    #                np.logspace(np.log10(1), np.log10(1700000), 100))

   # else:
    #mh = Histdd(x, y, bins=(xbins, ybins))
    #mh.plot(log_scale=True, cblabel="Events per bin")
        #return mh
    #plt.xscale("log")
    z_lower = 15000
    z_upper = z_lower+z_step
    fig, ax = plt.subplots(figsize = (10, 10))
    cm = plt.get_cmap('gist_rainbow')
    dat = []
    ax.set_prop_cycle('color', [cm(1.*i/7) for i in range(7)])
    while(z_upper < 2500000):
        z_range = [z_lower, z_upper]
        events_zslice = events[(events['drift_time']>z_range[0]) & (events['drift_time']<z_range[1])]
        hist, bins = np.histogram(events_zslice['time_difference'], 200, [0, 100000])
        errs = hist
        for i in errs:
            if i < 1:
                i = 1
        errs = np.sqrt(errs)
        bins_center = (bins[:-1] + bins[1:])/2.
        #plt.hist(events_zslice['time_difference'], 50, [0, 100000], lw = 3, color=color, edgecolor = color, label = "z_range: [{}, {}]".format(z_upper, z_lower), fc = 'none')
        plt.errorbar(bins_center, hist/np.sum(hist), yerr = errs/np.sum(hist), label = "[{:.1f} cm, {:.1f} cm]".format(0.0675*z_lower/1000, 0.0675*z_upper/1000), fmt = 'o')
        z_lower += z_step
        z_upper += z_step
        dat.append([bins_center, hist, errs])
    if log: plt.yscale("log")
    plt.xlabel("Time Difference (ns)", fontsize = 18)
    plt.ylabel("# Events", fontsize = 18)
    #plt.title("Histogram of $\Delta t$ by z Slice for Double Scatter Events", fontsize = 24)
    plt.legend(title = 'Depth', title_fontsize = 14, fontsize = 13)
    plt.show()
    if print_n:
        print(f"{len(events):,} events remaining")
    return np.asarray(dat)
    
def residual(p,func, xvar, yvar, err):
    return (func(p, xvar) - yvar)/err

from scipy import optimize

def data_fit(p0,func,xvar, yvar, err,tmi=0):
    try:
        fit = optimize.least_squares(residual, p0, args=(func,xvar, yvar, err),verbose=tmi)
    except Exception as error:
        print("Something has gone wrong:",error)
        return p0,np.zeros_like(p0),-1,-1
    pf = fit['x']

    print()

    try:
        cov = np.linalg.inv(fit['jac'].T.dot(fit['jac']))          
        # This computes a covariance matrix by finding the inverse of the Jacobian times its transpose
        # We need this to find the uncertainty in our fit parameters
    except:
        # If the fit failed, print the reason
        print('Fit did not converge')
        print('Result is likely a local minimum')
        print('Try changing initial values')
        print('Status code:', fit['status'])
        print(fit['message'])
        return pf,np.zeros_like(pf),-1,-1
            #You'll be able to plot with this, but it will not be a good fit.

    chisq = sum(residual(pf,func,xvar, yvar, err) **2)
    dof = len(xvar) - len(pf)
    red_chisq = chisq/dof
    pferr = np.sqrt(np.diagonal(cov)) # finds the uncertainty in fit parameters by squaring diagonal elements of the covariance matrix
    print('Converged with chi-squared {:.2f}'.format(chisq))
    print('Number of degrees of freedom, dof = {:.2f}'.format(dof))
    print('Reduced chi-squared {:.2f}'.format(red_chisq))
    print()
    Columns = ["Parameter #","Initial guess values:", "Best fit values:", "Uncertainties in the best fit values:"]
    print('{:<11}'.format(Columns[0]),'|','{:<24}'.format(Columns[1]),"|",'{:<24}'.format(Columns[2]),"|",'{:<24}'.format(Columns[3]))
    for num in range(len(pf)):
        print('{:<11}'.format(num),'|','{:<24.3e}'.format(p0[num]),'|','{:<24.3e}'.format(pf[num]),'|','{:<24.3e}'.format(pferr[num]))
    return pf, pferr, chisq,dof
def expfunc_bg(p, x):
    return (p[0]*np.exp(-x*p[1])) + p[2]



plt.rc("xtick", labelsize = 13)
plt.rc("ytick", labelsize = 13)
histograms = dt_difference_hist_together(events_double_scatter, z_step = 400000)

#*************************************************
#This is how I currently fit everything.   I don't think it's the best way to do it, so I'm only including it for the top z slice.  If you use this, just change the drift time bins.
#You can also play around with the value of approx_index to see how it changes the fit
#*************************************************

dat = []
approx_index = np.argmax(histograms[0][1])
eds = events_double_scatter[(events_double_scatter['drift_time']>15000) & (events_double_scatter['drift_time']<415000) & (events_double_scatter['time_difference']>histograms[0][0][approx_index])]
av = 0
for i in range(approx_index, len(histograms[0][1])-5):
    #print(histograms[0][1][i]/histograms[0][1][i+5])
    av = av + np.log(histograms[0][1][i]/histograms[0][1][i+5])/5000.
print(av)
decay_rate = av/((len(histograms[0][1])-5)-approx_index)
print(histograms[0][1][approx_index])
exp_guess = [histograms[0][1][approx_index]*np.exp(decay_rate*approx_index*1000), decay_rate, 0]
print(exp_guess)
pftest, pferrtest, chisqtest, doftest = data_fit(exp_guess, expfunc_bg, histograms[0][0][approx_index:], histograms[0][1][approx_index:], histograms[0][2][approx_index:])
dat.append([pftest, pferrtest, chisqtest, doftest])




