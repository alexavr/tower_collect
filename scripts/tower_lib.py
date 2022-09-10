# *** README
# ** INSTALLATION
# conda create --name tower jupyter xarray dask netCDF4 bottleneck pandas matplotlib numpy -y
# conda activate tower

import pandas as pd # conda install -c conda-forge pandas
import xarray as xr # conda install -c conda-forge xarray dask netCDF4 bottleneck
import matplotlib.pyplot as plt # conda install -c conda-forge matplotlib
import numpy as np # conda install -c conda-forge numpy
from pandas.tseries.frequencies import to_offset

class test:

    @staticmethod
    def create(time_start,time_end,frequency, trend=False):
        '''Creates data similar to real one.
        time_start,time_end - YYYY-mm-DD HH:MM:SS
        frequency in Hz
        trend - add linear trend
        '''

        lo = -2
        hi = 2

        ts = pd.to_datetime(time_start,format='%Y-%m-%d %H:%M:%S')
        te = pd.to_datetime(time_end,format='%Y-%m-%d %H:%M:%S')
        dt = f'{int(1/frequency*10**9)}N'

        time = pd.date_range(start=ts, end=te, freq=dt)
        ntime = len(time)

        # u = np.random.uniform(low=lo, high=hi, size=(ntime,))
        u = np.random.normal(0, 1, size=(ntime,))
        u = u+1*np.cos(np.linspace(-(1*np.pi), 2*np.pi, ntime))
        if trend:
            u = u + np.linspace(0, hi, ntime)
        u = np.float32(u)

        # data = {'u':u,'v':v,'w':w,'temp':temp,'e1':e1,'e2':e2,'e3':e3,'e4':e4}

        df = pd.DataFrame(data={'u':u}, index=time)
        df.index.name='time'

        return df

    @staticmethod
    def gap(data, gap_start=None, gap_size=0):
        '''Creates gap in data.
        gap_size - in steps
        '''

        ntime = len(data.index)

        # creating the gap
        if gap_size != 0:
            if gap_start == None:
                center = np.int(ntime/2)
                ist = center - np.int(gap_size/2)
                ied = center + np.int(gap_size/2)
                # print(f'{ntime} : {ist}-{ied}, center = {center}')
                data[ist:ied] = np.NaN
            else:
                # ist = df.index[df.index == gap_start].tolist()
                ist = data.index.searchsorted(gap_start)
                ied = ist + gap_size
                # print(f'{ntime} : {ist}-{ied}, starts = {ist}')
                data[ist:ied] = np.NaN


        return data


class plot:

    @staticmethod
    def simple(data, label=None):
        ''' Plot test results '''

        if label == None:
            plt.plot(data)
        else:
            plt.plot(data, label=label)

        return plt

    @staticmethod
    def web_accustic(data, window_min, frequency, attrs, figname):
        ''' Plot results for web
        for acoustic anemometer 
        '''
        import matplotlib.dates as mdates
        from matplotlib.axis import Axis  
        from mpl_toolkits.axes_grid1 import host_subplot
        from matplotlib.ticker import MultipleLocator, MaxNLocator
        from datetime import datetime, timedelta

        window = np.int(window_min*60*frequency)
        fine_curves = False
        no_data = False
        varname = attrs['name']
        vardesrt = attrs['long_name']
        varunits = attrs['units']
        time_lim_h = attrs['time_lim_h']

# fix this prior committing:
        # today_str = "2022-05-30 03:59:10" # delete
        # today_datetime = pd.to_datetime(today_str,format='%Y-%m-%d %H:%M:%S') # delete 
        today_datetime = datetime.now() # uncomment
        # yesterday = today_datetime - timedelta(hours=24) # delete
        yesterday = today_datetime - timedelta(hours=time_lim_h) # uncomment

        today_ind = data.index.searchsorted(today_datetime) 
        dtime = (today_datetime - data.index[-1]).total_seconds()/(60*60)
            
        
        if dtime >= time_lim_h or today_ind == 0: 
            no_data = True
            # yesterday = data.index[0]
            # today_datetime = data.index[-1]
            # today_ind = -1
        else:
          # slice the data
            data = data[yesterday:today_datetime]
        



        if fine_curves:
            tsm = data.rolling(window=window, min_periods=np.int(window/4),center=True).mean()
            std = data.rolling(window=window, min_periods=np.int(window/4),center=True).std() 
        else:           
            tsm = data.resample(f'{window_min}T').mean()
            tsm.index = tsm.index + to_offset(f'{np.int(window_min/2)}min')
        
        std = data.resample(f'{window_min}T').std()
        std.index = std.index + to_offset(f'{np.int(window_min/2)}min')

        tstd2 = tsm-std
        tstd1 = tsm+std

        mask = np.full_like(data, True)
        mask = np.where(np.isnan(data), mask, False)
        series = pd.Series(mask, index=data.index)
        # quality = (series.rolling(window=window, center=True).sum()/window)*100.
        # quality = (series.resample(timedelta(seconds=window*60), label="left").sum()/window)*100.
        quality = (1.-series.resample('30min').count()/(30*60*frequency))*100 ############################ UPDATE
        
        # PLOT
        
        # colors = ['olive','black','slategrey'] # all, hist_edgecolor, quality
        colors = ['darkgoldenrod','black','slategrey'] # all, hist_edgecolor, quality
        # colors = ['orangered','black','slategrey'] # all, hist_edgecolor, quality
        fonts  = [8,9] # all, titles

        fig = plt.figure(constrained_layout=True, figsize=(10, 3))
        (subfig_l, subfig_r) = fig.subfigures(1, 2, wspace=0.1, width_ratios=[3, 1])
        
        ax0 = subfig_l.subplots()

        par = ax0.twinx()

        make_patch_spines_invisible(par)

        par.spines["left"].set_visible(True)
        par.yaxis.set_label_position('left')
        par.yaxis.set_ticks_position('left')

        par.spines["left"].set_position(("axes", -0.1)) # green one

        # Raw data curves (data, running mean and quality)

        p1 = ax0.plot(data, label=f'{varname} raw data', color=colors[0], alpha=0.4, linewidth=0.5)
        ax0.fill_between(tstd1.index, tstd1, tstd2, alpha=0.6, color=colors[0], linewidth=0.0)
        p2 = ax0.plot(tsm, label=f'{varname} avg for {window_min} min, std', color=colors[0], alpha=1, linewidth=1.5)

        ax0.grid(linestyle=':', alpha=1, linewidth=0.8)
        ax0.tick_params(axis='both', labelsize=fonts[0], direction='in')
        ax0.tick_params(axis='y')
        ax0.set_xlabel('Time, UTC', fontdict={'size': fonts[0]})
        ax0.set_ylabel(f"{vardesrt} ({varname}) [{varunits}]", fontdict={'size': fonts[0]}, color=colors[0])
        ax0.set_title(f"{vardesrt} ({varname}), last 24hr", fontdict={'size': fonts[1]})
        ax0.set_xlim(xmin=yesterday, xmax=today_datetime) # (xmin=data.index[0], xmax=data.index[-1])
        Axis.set_major_formatter(ax0.xaxis, mdates.DateFormatter('%m/%d\n%H:%M'))
        
        par.set_yscale('log')
        par.set_ylim(ymin=0.001, ymax=110)
        par.tick_params(axis='both', labelsize=fonts[0], direction='in')
        par.tick_params(axis='y', labelsize=fonts[0], direction='in', colors="black")
        par.tick_params(which='major', length=4, width=0.5, direction='in')
        par.tick_params(which='minor', length=2, width=0.5, direction='in')
        p3 = par.plot(quality, label='quality', color=colors[2], alpha=0.4, linewidth=0.8, zorder=0)
        par.fill_between(quality.index, par.get_ylim()[0], quality, color=colors[2], alpha=0.2, zorder=0)
        par.set_ylabel('Protion of missing values [%]', fontdict={'size': fonts[0]}, color=colors[2])

        lns = p1+p2+p3
        labs = [l.get_label() for l in lns]
        ax0.legend(lns, labs, loc='best', prop={'size': fonts[0]})

        # Histogram
        # print(data)
        counts = data.resample('1S').count()

        ax1 = subfig_r.subplots()
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax1.set_title('Data quality, last 24 h', fontdict={'size': fonts[1]})
        ax1.set_ylabel('Number of measurements', fontdict={'size': fonts[0]})
        ax1.set_xlabel('Packages/sec', fontdict={'size': fonts[0]})
        ax1.grid(linestyle=':', alpha=1, linewidth=0.8)
        ax1.tick_params(axis='both', labelsize=fonts[0], direction='in')
        ax1.tick_params(labelsize=fonts[0])
        ax1.tick_params(which='major', length=4, direction='in')
        ax1.tick_params(which='minor', length=2, direction='in')
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.xaxis.set_minor_locator(MaxNLocator(integer=True))

        if not no_data:
            ax1.hist(counts, 
                bins=np.arange(np.min(counts)-0.5,np.max(counts)+0.6,1),
                # density=True, 
                log=True, color=colors[0], 
                rwidth=None, 
                edgecolor=colors[1], 
                linewidth=0.5)

        if no_data:
            ax0.text(0.5, 0.5, 'NO DATA', fontweight='normal',transform=ax0.transAxes,
                horizontalalignment='center',verticalalignment='center',
                bbox=dict(fc='pink', ec='black', 
                    linewidth=0.7,
                    alpha=0.9, 
                    boxstyle='square,pad=0.8'))
            ax1.text(0.5, 0.5, 'NO DATA', fontweight='normal',transform=ax1.transAxes,
                horizontalalignment='center',verticalalignment='center',
                bbox=dict(fc='pink', ec='black', 
                    linewidth=0.7,
                    alpha=0.9, 
                    boxstyle='square,pad=0.8'))
        
        plt.savefig(figname, dpi=150)
        # plt.show()

    @staticmethod
    def web_meteo(data, window_min, frequency, attrs, figname):
        ''' Plot results for web
        for meteo (slow) data 
        '''
        import matplotlib.dates as mdates
        from matplotlib.axis import Axis  
        from mpl_toolkits.axes_grid1 import host_subplot
        from matplotlib.ticker import MultipleLocator, MaxNLocator
        from datetime import datetime, timedelta

        window = np.int(window_min*60*frequency)
        fine_curves = True
        no_data = False
        varname = attrs['name']
        vardesrt = attrs['long_name']
        varunits = attrs['units']
        time_lim_h = attrs['time_lim_h']

# fix this prior committing:
        # today_str = "2022-05-30 03:59:10" # delete
        # today_datetime = pd.to_datetime(today_str,format='%Y-%m-%d %H:%M:%S') # delete 
        today_datetime = datetime.now() # uncomment
        # yesterday = today_datetime - timedelta(hours=24) # delete
        yesterday = today_datetime - timedelta(hours=time_lim_h) # uncomment

        today_ind = data.index.searchsorted(today_datetime) 
        dtime = (today_datetime - data.index[-1]).total_seconds()/(60*60)
            
        
        if dtime >= time_lim_h or today_ind == 0: 
            no_data = True
            # yesterday = data.index[0]
            # today_datetime = data.index[-1]
            # today_ind = -1
        else:
          # slice the data
            data = data[yesterday:today_datetime]
        

        if fine_curves:
            tsm = data.rolling(window=window, min_periods=np.int(window/4),center=True).mean()
            std = data.rolling(window=window, min_periods=np.int(window/4),center=True).std() 
        else:           
            tsm = data.resample(f'{window_min}T').mean()
            tsm.index = tsm.index + to_offset(f'{np.int(window_min/2)}min')
        
        std = data.resample(f'{window_min}T').std()
        std.index = std.index + to_offset(f'{np.int(window_min/2)}min')

        tstd2 = tsm-std
        tstd1 = tsm+std

        # PLOT
        
        # colors = ['olive','black','slategrey'] # all, hist_edgecolor, quality
        colors = ['darkgoldenrod','black','slategrey'] # all, hist_edgecolor, quality
        # colors = ['orangered','black','slategrey'] # all, hist_edgecolor, quality
        fonts  = [8,9] # all, titles

        if varname == "temp":
            colors = ['orangered','black','slategrey'] # all, hist_edgecolor, quality
        
        if varname == "rh":
            colors = ['olive','black','slategrey'] # all, hist_edgecolor, quality


        fig = plt.figure(constrained_layout=True, figsize=(10, 3))
        subfig_l = fig.subfigures(1, 1, wspace=0.1)
        
        ax0 = subfig_l.subplots()

        # Raw data curves (data, running mean and quality)

        p1 = ax0.plot(data, label=f'{varname} raw data', color=colors[0], alpha=0.4, linewidth=1)
        ax0.fill_between(tstd1.index, tstd1, tstd2, alpha=0.6, color=colors[0], linewidth=0.0)
        p2 = ax0.plot(tsm, label=f'{varname} avg for {window_min} min, std', color=colors[0], alpha=1, linewidth=1.5)

        ax0.grid(linestyle=':', alpha=1, linewidth=0.8)
        ax0.tick_params(axis='both', labelsize=fonts[0], direction='in')
        ax0.tick_params(axis='y')
        ax0.set_xlabel('Time, UTC', fontdict={'size': fonts[0]})
        ax0.set_ylabel(f"{vardesrt} ({varname}) [{varunits}]", fontdict={'size': fonts[0]}, color=colors[0])
        ax0.set_title(f"{vardesrt} ({varname}), last 24hr", fontdict={'size': fonts[1]})
        ax0.set_xlim(xmin=yesterday, xmax=today_datetime) # (xmin=data.index[0], xmax=data.index[-1])
        Axis.set_major_formatter(ax0.xaxis, mdates.DateFormatter('%m/%d\n%H:%M'))
        
        lns = p1+p2
        labs = [l.get_label() for l in lns]
        ax0.legend(lns, labs, loc='best', prop={'size': fonts[0]})

        if no_data:
            ax0.text(0.5, 0.5, 'NO DATA', fontweight='normal',transform=ax0.transAxes,
                horizontalalignment='center',verticalalignment='center',
                bbox=dict(fc='pink', ec='black', 
                    linewidth=0.7,
                    alpha=0.9, 
                    boxstyle='square,pad=0.8'))
        
        plt.savefig(figname, dpi=150)
        # plt.show()

    @staticmethod
    def web_accustic_stat_prep():
        ''' Plot stat results for web
        '''

        fig = plt.figure(constrained_layout=True, figsize=(10, 5))
        (subfig_l, subfig_r) = fig.subfigures(1, 2, wspace=0.05, width_ratios=[1, 1])
        
        ax0 = subfig_l.subplots()
        ax1 = subfig_r.subplots()

        return plt, ax0, ax1
    
    @staticmethod
    def web_accustic_stat(data, frequency, window, plt, ax0, ax1, label):
        from scipy import signal

        colors = ['darkgoldenrod','black','slategrey'] # all, hist_edgecolor, quality
        # colors = ['orangered','black','slategrey'] # all, hist_edgecolor, quality
        fonts  = [8,9] # all, titles

        # Fast quality check:
        qtke = data.tke.isna().sum()/data.tke.count()*100
        # qtmp = data.tempp.isna().sum()/data.tempp.count()*100

        # Get rid of NaN. Rude but fast.
        tke = data.tke.dropna()
        temp = data.tempp.dropna()

        if qtke >= 10:
            label = label + " BAD DATA!!!"

        f, Pxx = signal.welch(tke, frequency, nperseg=window)
        ax0.plot(f, Pxx, label=label, alpha=0.7)
        ax0.grid(linestyle=':', alpha=1, linewidth=0.8)
        ax0.tick_params(axis='both', labelsize=fonts[0], direction='in')
        ax0.tick_params(axis='y')
        ax0.tick_params(which='major', length=4, direction='in')
        ax0.tick_params(which='minor', length=2, direction='in')
        # ax0.ylim([0.5e-3, 1])
        ax0.set_yscale('log')
        ax0.set_xscale('log')
        ax0.set_xlabel('frequency [Hz]')
        ax0.set_ylabel('PSD [$m^{2} s^{-2}/Hz$]')
        ax0.set_title('Kinetic Energy')
        
        f, Pxx = signal.welch(temp, frequency, nperseg=window)
        ax1.plot(f, Pxx, label=label, alpha=0.7)
        ax1.grid(linestyle=':', alpha=1, linewidth=0.8)
        ax1.tick_params(axis='both', labelsize=fonts[0], direction='in')
        ax1.tick_params(axis='y')
        ax1.tick_params(which='major', length=4, direction='in')
        ax1.tick_params(which='minor', length=2, direction='in')
        # ax1.ylim([0.5e-3, 1])
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlabel('frequency [Hz]')
        ax1.set_ylabel('PSD [$K/Hz$]')
        ax1.set_title('Acoustic temperature')
        
        # plt.suptitle(f"", fontsize=16)

        return plt, ax0, ax1


class reader:

    @staticmethod
    def netcdf(file):
        '''Reads NetCDF and parses into pandas DataFrame'''
        ds = xr.open_dataset(file)
        df = ds.to_dataframe()

        return df

    @staticmethod
    def buffer(file):
        '''Reads npz files and parses into pandas DataFrame'''
        npz = np.load(file)
        df = pd.DataFrame(columns = npz.files)
        for item in npz.files:
           df[item] = npz[item]

        df['datetime'] = pd.to_datetime(df['time'],unit='s')        
        df = df.set_index('datetime')
        df = df.drop(['time'], axis=1)
        return df

class data:

    @staticmethod
    def clean(df):
        '''Cleans the data:
        * removing dublicates
        * sorting by time
        '''
        df = df.drop_duplicates()
        df = df.sort_index()

        return df

        plt.show()

    @staticmethod
    def create_empty_pd():
        ''' VERY BAD IDEA!!!!!!!!!!!!!!!!!!!!!!!!!!
        Improve ASAP! Done for no buffer file case.
        '''

        time = pd.date_range(start='1/1/1900', end='1/2/1900', freq='S')
        ntime = len(time)
        df = pd.DataFrame({'datetime': time,
            'u':    np.full(ntime, np.nan),
            'v':    np.full(ntime, np.nan),
            'w':    np.full(ntime, np.nan),
            'temp': np.full(ntime, np.nan),
            'rh':   np.full(ntime, np.nan),
            'year': np.full(ntime, np.nan)})
        df = df.set_index('datetime')

        return df


class math:

    @staticmethod
    def primes(data, window, detrend=None):
        ''' 
        Get fuluctuations from raw data using mean method and regression (COMMING SOON)
        '''

        vars = ["u","v","w","temp"]

        for var in vars:
            if var in data.columns:
                new_name = f"{var}p"
                if detrend == 'mean':
                    data[new_name] = data[var] - data[var].rolling(window=window).mean()
                else:
                    data[new_name] = data[var]
            else:
                print(f"No {var} component found! Skipping this variable...")

        return data

    @staticmethod
    def tke(data):
        ''' 
        Compute TKE
        '''

        if 'up' not in data.columns:
            data['up'] = 0
        if 'vp' not in data.columns:
            data['vp'] = 0
        if 'wp' not in data.columns:
            data['wp'] = 0

        data['tke'] = np.sqrt( data.up**2 + data.vp**2 + data.wp**2 )

        return data

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

