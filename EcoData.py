# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:14:18 2017
 
@author: herve.biancotto
 
@aim: Implements functions to import, standardise and visualise economic data.
"""
 
#import bloompy as bp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns

RECESSION_DATA = r'P:\Python Scripts\EcoProject/usrecspan.csv'

def PoP(data, **kwargs):
    """
    Returns the period over period data series.
    :param data: the data to transform.
    :type data: pandas DataFrame of 1 column.
   
    :key word parameters:
    :param per: period corresponding to the period over period transformation.
    :param freq: frequency of the series to return.
    :type per: string in list ['D', 'M', 'Q', 'A']
    :type freq: string in list ['D', 'M', 'Q', 'A']
    :param method: defines the method to perform the period over period transformation.
    :type method: ['ratio', 'diff']
    """
   
    # Correspondance tables
    CORRES_B = {'D':1, 'M':21, 'Q':63, 'H':130, 'A':262} #business days
    CORRES_D = {'D':1, 'M':30, 'Q':91, 'H':182, 'A':365} #calendar days
    CORRES_M = {'M':1, 'Q':3, 'H':6, 'A':12}
    CORRES_Q = {'Q':1, 'H':2, 'A':4}
    CORRES_LABEL = {'D':'DoD', 'M':'MoM', 'Q':'QoQ', 'H':'HoH', 'A':'YoY'}

    if "per" in kwargs:
        period = kwargs["per"].upper()
    else:
        period = 'A'
       
    if "freq" in kwargs:
        frequency = kwargs["freq"].upper()
    else:
        frequency = 'D'

    #map correspondance tables
    if frequency == 'D':
        CORRES = CORRES_D
    elif frequency == 'B':
        CORRES = CORRES_B
    elif frequency == 'M':
        CORRES = CORRES_M
    elif frequency == 'Q':
        CORRES = CORRES_Q
   
    meth = 'ratio'
    if "method" in kwargs:
        if kwargs["method"] == 'diff':
            meth = kwargs["method"]
   
    #Converts series to DataFrame if necessary
    if isinstance(data, pd.Series):
        data = data.to_frame(data.name)
   
    # Aligns frequency and shift the second series by the period
    data = data.asfreq(frequency, method='pad')
    data_per = data.shift(CORRES[period])
    if meth == 'ratio':
        result = data/data_per-1
    elif meth == 'diff':
        result = data - data_per
    result.columns = [list(result)[0] + ' ' + CORRES_LABEL[period]]
    # Returns the period over period transformation
    return result

 
def scale_data(data, mult=1, lead_lag=0, cum_data=False, freq='D'):
    """
    Scales data by applying a multiplier, lagging or leading and cumulating the
    pandas time series.
   
    :param data: the data to transform.
    :type data: pandas DataFrame.
   
    Named arguments:
    :param mult: multiplier to apply to the series.
    :param mult: a scalar, type float or int.
   
    :param lead_lag: number of period to lead(positive) or lag(negative) data.
    :type lead_lag: signed int.
    """
 
        #Converts series to DataFrame if necessary
    if isinstance(data, pd.Series):
        data = data.to_frame(data.name)
   
    if cum_data:
        value = mult*data.tshift(lead_lag, freq=freq)
        value = value.cumsum()
    else:
        value = mult*data.tshift(lead_lag, freq=freq)
    return value
 
       
 
def graph(*args, **kwargs):
    """
    Graphs pandas DataFrame that are passed through.
    Allows overlays of recession periods.
   
    :param data: input data as a pandas DataFrame.
    :type data: pandas DataFrame.
   
    :param recession: Overlays the chart with recession periods.
    :type recession: Boolean.
   
    :param multiple_series: Adds an extra Y axe for each series.
    :type multiple_series: Boolean.
   
    :param chart_size: Size of the output chart.
    :type chart_size: tuple of (int, int).
   
    :param title: Gives a title to the chart.
    :type title: str.
   
    :param subgraph: Adds a subgraph to the chart.
    :type subgraph: pandas Series.
   
    :param candle: Change graph type to candlesticks.
    :type candle: Boolean.
   
    :param legend: Display legend.
    :type legend: Boolean.
   
    :param save_fig: Path and name to save the plot.
    :type save_fig: str.
   
    :param has_dates: Input data have dates. Used only for candle graph. True by default.
    :type has_dates: Boolean.
   
    :param candle_width: candle width.
    :type candlewidth: float.
   
    :param axe_label: shows axe's labels.
    :type axe_label: Boolean.
   
    :param view_grid: shows grid.
    :type view_grid: Boolean.
    
    :param close_plot: Deletes the figure. Use for memory control. Default is False.
    :type close_plot: Boolean.
    """
   
    ## Check data
    data = [pd.DataFrame(data_set) for data_set in args]
    try:
        assert data is not None
    except AssertionError:
        print("No data to graph")
        return
   
    ## Deal with kwargs
    if 'colour' in kwargs:
        if len(kwargs['colour']) != len(args):
            raise Exception('The colour provided should be of same lenght\
            as the data provided')
        try:
            colour = kwargs['colour']
        except Exception as error:
            print(error)
            return
    else:
        colour = {1:'#4a4a4a',
                  2:'#951826',
                  3:'#ea7600',
                  4:'#007398',
                  5:'#33cc78',
                  6:'#f4ee00',
                  7:'#669999',
                  8:'#ff99ff',
                  9:'#b35900',}
   
    recess = False
    if 'recession' in kwargs:
        if kwargs['recession']:
            rec_data = pd.read_csv(RECESSION_DATA, header=0, names=['date',\
            'recession'], converters={'date':lambda x: pd.to_datetime(x,\
            dayfirst=True)})
            #rec_data = rec_data.asfreq(data[0].index.freqstr)
            recess = True  
   
    extra_ax = False
    sns.set_style('ticks', {'axes.grid': True})
    if 'multiple_series' in kwargs:
        if kwargs['multiple_series']:
            extra_ax = True
            sns.set_style({'axes.grid': False})
            side = 'right' # alternate right and left when adding a new Y axis.
   
    size = None
    if 'chart_size' in kwargs:
        size = kwargs['chart_size']
   
    g_title = ''
    if 'title' in kwargs:
        g_title = kwargs['title']

    sub = None
    n_sub = 0
    if 'subgraph' in kwargs:
        sub = kwargs['subgraph']
        if sub is not None:
            if isinstance(sub, pd.DataFrame) or isinstance(sub, pd.Series):
                n_sub = 1
            elif isinstance(sub, (list, tuple)):
                n_sub += len(sub)
     
    candle = False
    if 'candle' in kwargs:
        candle = kwargs['candle']
    
    has_dates = True
    if 'has_dates' in kwargs:
        has_dates = kwargs['has_dates']
    
    legend = True
    if 'legend' in kwargs:
        legend = kwargs['legend']
    
    legend_loc = 0
    if 'legend_loc' in kwargs:
        legend_loc = kwargs['legend_loc']
   
    save = False
    if 'save_fig' in kwargs:
        save_path = kwargs['save_fig']
        save = True
       
    candle_w = 0.2
    if 'candle_width' in kwargs:
        candle_w = kwargs['candle_width']
   
    axe_label = True
    if 'axe_label' in kwargs:
        axe_label = kwargs['axe_label']
   
    view_grid = True
    if 'view_grid' in kwargs:
        view_grid = kwargs['view_grid']
   
    trading_signal = None
    if 'trading_signal' in kwargs:
        trading_signal = kwargs['trading_signal'].copy()
        if isinstance(trading_signal, pd.DataFrame):
            trading_signal = pd.Series(trading_signal)
    
    close_plot = False
    if 'close_plot' in kwargs:
        close_plot = kwargs['close_plot']
   
    ## Graph creation
    if size is not None:
        fig = plt.figure(figsize=size)
    else:
        fig = plt.figure()
   
    rows = 5
    grid = plt.GridSpec(rows, 1)  
    ax = fig.add_subplot(grid[:rows-n_sub, 0])
   
    #main line
    if candle: # handle candlesticks
        if data[0].shape[1]!=4 and data[0].shape[1]!=5:
            raise IndexError('Need 4/5 data series. Got {}'.format(data[0].shape[1]))
        line, _ = candle_graph(ax, data[0], width=candle_w, has_dates=has_dates)
    else:
        line = plt.plot(data[0], label=data[0].columns[0], color=colour[1]) #base line
       
    # Displays trading signals
    if trading_signal is not None:
        if candle:
            dat = pd.Series(data[0].iloc[:,0])
        if isinstance(data[0], pd.DataFrame):
            dat = pd.Series(data[0].iloc[:,0])
        else:
            dat = data[0]
        y_range = ax.get_ylim()
       
        sig = dat.copy()
        trading_signal[trading_signal==0] = np.nan
        b_sig = trading_signal[trading_signal==1]
        s_sig = trading_signal[trading_signal==-1]
        buy_sig = b_sig * -0.1 * (y_range[1] - y_range[0]) + sig
        sell_sig = s_sig * -0.1 * (y_range[1] - y_range[0]) + sig
        plt.plot(buy_sig, 'g^')
        plt.plot(sell_sig, 'rv')
   
    # Multiple series
    if extra_ax:
        ax.set_ylabel(data[0].columns[0], color=colour[1])
    i = 2 # counter to handle colours
    for series in data[1:]:
        if extra_ax: #case multiple Y axis
            axn = ax.twinx()  
            axn.set_frame_on(True)#adds to right
            axn.patch.set_visible(False)
            axn.spines[side].set_position(('axes', 1+0.07*(i-2))) #pushes out the new Y axis
            axn.set_ylabel(series.columns[0], color=colour[i])
        line += plt.plot(series, colour[i], label=series.columns[0])
        i+=1 #change colour for next series
    if legend:
        labels = [l.get_label() for l in line] #gets label all together
        plt.legend(line, labels, loc=legend_loc)

   
    # Displays recessions as shaded area
    if recess:
        data_rec = rec_data[rec_data['date'] >= data[0].index[0]]
        data_rec.index = range(len(data_rec))
        beg = 0
        if len(data_rec) != 0:
            if len(data_rec)%2 == 1:
                axn = ax.twinx()
                plt.axvspan(data[0].index[0],data_rec[0]\
                        , facecolor='0.5', alpha=0.4)
                axn.grid(False)
                ax.set_yticklabels([])
                sns.despine(ax=axn, left=False, right=False)
                beg = 1
            try:
                for i in range(beg,len(data_rec['date']),2):
                    axn = ax.twinx()
                    plt.axvspan(data_rec.date[i],\
                                data_rec.date[i+1],\
                                facecolor='0.5', alpha=0.4)
                    axn.grid(False)
                    ax.set_yticklabels([]) #no tick label
                    sns.despine(ax=axn, left=False, right=False) # axis not displayed
            except IndexError:
                print('no recession during that period')
   
    # Axe's labels
    if axe_label==False:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
   
    # Grid
    if view_grid==False:
        ax.grid(False)
       
    #Title
    if g_title is not None:
        ax.axes.set_title(g_title)

    # Adds subplot
    if sub is not None:
        for i in range(n_sub):
            if candle:
                subg = fig.add_subplot(grid[rows- n_sub + i,0])
            else:
                subg = fig.add_subplot(grid[rows- n_sub + i,0], sharex=ax)
            subg.plot(sub)
            # Oscillator formating
            up_lim = np.array([70]*len(sub.values))
            dn_lim = np.array([30]*len(sub.values))
            subg.set_ylim([0,100])
            subg.axhline(30, color='green')
            subg.axhline(70,  color='red')
            subg.grid(False)
            x = sub.index.values
            y = sub.values.flatten()
            subg.fill_between(x, up_lim, y, \
                            where=(y>=up_lim), facecolor='red', interpolate=True, alpha='0.25')
            subg.fill_between(x, y, dn_lim,\
                            where=(y<=dn_lim), facecolor='green', interpolate=True, alpha='0.25')
            if axe_label==False:
                subg.axes.get_xaxis().set_visible(False)
                subg.axes.get_yaxis().set_visible(False)
   
   
   
    # Saves generated plot to disk
    if save:
        fig.savefig(save_path,  bbox_inches='tight')
    if close_plot:
        plt.close('all')
        
    
def candle_graph(ax, data, colorup='k', colordown='r', alpha=1.0, \
                 width=.5, freq=10, has_dates=True):
    """
    Creates candle graph for financial series.
    Input:
    ======
    :param data <pd.DataFrame>/<np.array>: open high low close of series.
    
    """
    
    if isinstance(data, pd.DataFrame):
        if has_dates:
            dates = data.index.strftime('%d-%b-%Y')
        data = data.as_matrix()
    elif isinstance(data, np.ndarray):
        if has_dates:
            dates = data[:,0]
            data = data[:,1:]
    
    data = np.c_[np.arange(1,data[:,0].size+1), data[:,:]]
    
    center = width / 2.0
    
    lines = []
    patches = []
    for d in data:
        t, op, high, low, close = d[:5]
    
        if close >= op:
            color = colorup
            lower = op
            height = close - op
        else:
            color = colordown
            lower = close
            height = op - close
    
        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )
    
        rect = Rectangle(
            xy=(t - center, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)
    
        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()
    if has_dates:
        ax.set_xticklabels(dates[::freq])
    
    return lines, patches

def period_ohlc(df, freq='w', method='pad'):
    """
    Transform OHLC data into lower frequency OHLC data.
    """
    
    df.columns=['open','high','low','price']
    #close
    c = df['price'].asfreq(freq, method=method)
    idx = c.index.values
    idx = np.insert(idx, 0, df.index[0])
    c = c.values
    
    #high, low, open
    h = [df.loc[idx[i]:idx[i+1],'high'].max() for i in range(idx.size-1)]
    l = [df.loc[idx[i]:idx[i+1],'low'].min() for i in range(idx.size-1)]
    o = [df.loc[df[idx[i]:idx[i+1]].first_valid_index(),'open'] for i in range(idx.size-1)]
    
    #gather results in a DataFrame.
    ret=pd.DataFrame({'open':o , 'high':h, 'low':l, 'price':c}, index=idx[1:])
    
    return ret[df.columns]



if __name__ == "__main__":
    pass
 
#    spx = bp.getHistoricalData('SPX Index','px last','20010101', \
#                                          periodicity='DAILY')
#    conv_like(spx, spx)
#    modSpx = PoP(spx, per='a', freq='m')
#    gdp = bp.getHistoricalData('GDP CHWG Index','px last','20010101', \
#                                          periodicity='DAILY')
#    modGdp = PoP(gdp, per='a', freq='m')
#    pmi = bp.getHistoricalData('NAPMPMI Index','px last','20010101', \
#                                          periodicity='DAILY')
#    modPmi = PoP(pmi, per='a', freq='m')
#    
#    cembi = bp.getHistoricalData('JBCDCOMP Index', 'px_last', '20010101', \
#                                periodicity='DAILY')
#    modCembi = PoP(cembi, per='h', freq='m')
#    modCembi2 = PoP(cembi, per='a', freq='m')
#    
#    usd = bp.getHistoricalData('USTW$ Index','px last','20010101', \
#                                          periodicity='DAILY')
#    modUsd = PoP(usd, per='a', freq='m')
#    
#    graph(modPmi,modCembi, modCembi2,\
#          recession=True, multiple_series=True, title='bla')
    