# *** README
# ** INSTALLATION
# conda create --name tower jupyter xarray dask netCDF4 bottleneck pandas matplotlib numpy -y
# conda activate tower

import pandas as pd # conda install -c conda-forge pandas
import xarray as xr # conda install -c conda-forge xarray dask netCDF4 bottleneck
import matplotlib.pyplot as plt # conda install -c conda-forge matplotlib
import numpy as np # conda install -c conda-forge numpy
from pandas.tseries.frequencies import to_offset
<<<<<<< HEAD
from datetime import datetime, timedelta
from pathlib import Path

CONFIG_NAME = "/var/www/data/domains/tower.ocean.ru/html/flask/tower.conf"
=======
>>>>>>> d6fdfdc7f5d11eab379daaf4327675357a67ecea

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


<<<<<<< HEAD
################################################################################

class notification:

    @staticmethod
    def send2bot(TOKEN, chat_id, msg):
        import requests
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={msg}"
        status = requests.get(url).json()

        return status
        # print(requests.get(url).json()) # Эта строка отсылает сообщение

################################################################################
class plot:


    @staticmethod
    def web_accustic_3d(tower_name,level,eq_type,figname):
    # def web_accustic_3d(tower_name, equipment, frequency, level, figname):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        # import matplotlib.ticker as mticker
        # from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
        from matplotlib import colors
        import matplotlib as mpl
        # import matplotlib.ticker as tick # for label format
        # from datetime import datetime, timedelta
        import seaborn as sns # conda install -c anaconda seaborn

        config = reader.config(CONFIG_NAME)
        window_min = float(config['hf_window_min'])
        fine_curves = False
        textsize = 12
        var_name = "u,v,w,temp"
        time_lim_h = int(config['buffer_time_lim_h'])

        fig = plt.figure(constrained_layout=True, figsize=(13,10))

        # fig.suptitle("Controlling subplot sizes with width_ratios and height_ratios")

        gs = GridSpec(4, 4, figure=fig)
        ax1 = fig.add_subplot(gs[0, :-1]) # u, v
        ax2 = fig.add_subplot(gs[1, :-1]) # w
        ax3 = fig.add_subplot(gs[2, :-1]) # temp
        ax4 = fig.add_subplot(gs[3, :-1]) # temp
        ax5 = fig.add_subplot(gs[:,  -1]) # histogram

        axs = [ax1,ax2,ax3,ax4,ax5]

        dbfile = f"{config['dbpath']}/{config['dbfile']}"
        cur = reader.db_init(dbfile)

        equipments = reader.bd_get_table_df(cur,f"SELECT equipment, Hz \
            FROM equipment \
            WHERE tower_name='{tower_name}' AND height='{level}' AND type='{eq_type}' AND status='online' \
            GROUP BY equipment \
            ORDER BY equipment ASC")

        for index,eq in equipments.iterrows():

            equipment = eq['equipment']
            window = np.int(window_min*60*float(eq['Hz']))

            buffer_file = Path(config['buffer_path'],f'{tower_name}_{equipment}_BUFFER.npz' )
            data, status = reader.read_buffer(buffer_file)


            today_datetime = datetime.now() # uncomment
            yesterday = today_datetime - timedelta(hours=time_lim_h) # uncomment

            if not status:

                print(f"Working on var   named: {var_name:>10} {level:4.1f} {equipment:<7} NO DATA!")

                for ax in axs:

                    ax.text(0.5, 0.5, f"NO DATA", va="center", ha="center",transform=ax.transAxes)

                    # ax.text(0.5, 0.5, 'NO DATA', fontweight='normal',transform=ax.transAxes,
                    #     horizontalalignment='center',verticalalignment='center',
                    #     bbox=dict(fc='pink', ec='black',
                    #         linewidth=0.7,
                    #         alpha=0.9,
                    #         boxstyle='square,pad=0.8'))

            else:

                if status:

                    today_ind = data.index.searchsorted(today_datetime)
                    dtime = (today_datetime - data.index[-1]).total_seconds()/(60*60)

                    if dtime >= time_lim_h or today_ind == 0:
                        status = False
                    else:
                        data = data[yesterday:today_datetime]

                    print(f"Working on var   named: {var_name:>10} {level:4.1f} {equipment:<7} OK ")

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

                mask = np.full_like(data['u'], True)
                mask = np.where(np.isnan(data['u']), mask, False)
                series = pd.Series(mask, index=data.index)
                quality = (1.-series.resample(f"{config['hf_window_min']}min").count()/(window))*100 ############################ UPDATE
                # quality = quality.rolling(window=window, min_periods=np.int(window/4),center=True).mean()
                
                # plt.style.use('bmh')
                tsm['u'].plot(x=tsm['u'].index, ax=ax1, grid=True, linewidth=2, color='tab:blue', label='Earth-relative zonal wind (u)')
                ax1.fill_between(tstd1['u'].index, tstd1['u'], tstd2['u'], alpha=0.2, color='tab:blue', linewidth=0.0)
                tsm['v'].plot(x=tsm['v'].index, ax=ax1, grid=True, linewidth=2, color='tab:orange', label='Earth-relative meridional wind (v)')
                ax1.fill_between(tstd1['v'].index, tstd1['v'], tstd2['v'], alpha=0.2, color='tab:orange', linewidth=0.0)
                ax1.legend()

                tsm['w'].plot(x=tsm['w'].index, ax=ax2, grid=True, color='tab:blue', linewidth=2)
                ax2.fill_between(tstd1['w'].index, tstd1['w'], tstd2['w'], alpha=0.2, color='tab:blue', linewidth=0.0)

                tsm['temp'].plot(x=tsm['temp'].index, ax=ax3, grid=True, color='tab:blue', linewidth=2)
                ax3.fill_between(tstd1['temp'].index, tstd1['temp'], tstd2['temp'], alpha=0.2, color='tab:blue', linewidth=0.0)

                quality.plot(x=quality.index, ax=ax4, grid=True, linewidth=2, color='tab:grey', logy=True)
                ax4.fill_between(
                        x= quality.index, 
                        y1= quality, 
                        y2= 0, 
                        color= "#ABB2B9",
                        # ylim = [1, 100],
                        alpha= 0.3)

                
                counts = data['u'].resample('1S').count()
                bins = np.arange(np.min(counts), np.max(counts)+1,1)
                ax5.hist(x=counts,
                    # height=np.arange(len(bins)),
                    density=False,
                    log=True,
                    # label="var text",
                    color="tab:grey", #  '#a1c9f4', # tab:blue
                    # color=colors[0],
                    width=1.5,
                    # rwidth=None,
                    edgecolor='black', # colors[1],
                    linewidth=0.5)


        # DESIGN ########
        # yesterday = pd.to_datetime(yesterday).floor('3H')
        # today_datetime = pd.to_datetime(today_datetime).ceil('3H')
        ticks = pd.date_range(start=yesterday, end=today_datetime, freq='3H' ).round("3H") # inclusive='neither'
        labels = ticks.strftime('%H:%M\n%d %b') # inclusive='neither'
        
        for ii, ax in enumerate(axs):
            ax.tick_params(direction='in')
            ax.grid(axis='both', color='tab:grey',linestyle=':', linewidth=1)
            # ax.axes.get_xaxis().set_visible(False)
            ax.set_xticklabels([])
            ax.set_xticks(ticks)
            ax.set_xlabel("")
            # plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlim( (np.min(ticks), today_datetime ) )
            if not status:
                ax.set_yticklabels([])


            # ax.axes.get_yaxis().set_visible(False)
            if ii == len(axs)-2:
                # ax.set_xticks(ticks)
                # ax.axes.get_xaxis().set_visible(True)
                # ax.tick_params(direction='in')
                # plt.setp(ax.get_xticklabels(), visible=True)
                ax.set_xticklabels(labels)
                ax.set_xlabel("Time (UTC)", fontsize=textsize+2)
                # print(labels)

            if ii == len(axs)-1:
                ax.tick_params(direction='in', which='both',labelleft=False,labelright=True,labelbottom=True,right=True,left=False)
                ax.yaxis.set_label_position('left')
                # ax.axes.get_xaxis().set_visible(True)
                # ax.set_xticklabels(labels)

                if status:
                    ax.set_xlim( (bins[0], bins[-1]) )
                    ax.set_xticks(bins[::5])
                    ax.set_xticklabels(bins[::5])
                else:
                    xmin, xmax = 10, 30
                    ax.set_xlim( (xmin, xmax) )
                    ax.set_ylim( (0, 24*3600*20) )
                    ax.set_xticks( np.arange(xmin, xmax, 5) )
                    ax.set_xticklabels( np.arange(xmin, xmax, 5) )

        ax1.set_ylabel(f"u/v wind, ms-1", color="black", fontsize=textsize+2)
        ax2.set_ylabel(f"w wind, ms-1", color="black", fontsize=textsize+2)
        ax3.set_ylabel(f"temperature, degC", color="black", fontsize=textsize+2)
        ax4.set_ylabel(f"data quality, %", color="black", fontsize=textsize+2)
        ax5.set_xlabel(f"Data quality\n(measures/sec)", fontsize=textsize+2)

        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(figname, dpi=150)
        # plt.show()

        return True

    @staticmethod
    def web_meteo_new(tower_name,level,eq_type, figname):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        # import matplotlib.ticker as mticker
        # from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
        from matplotlib import colors
        # from datetime import datetime, timedelta
        import seaborn as sns # conda install -c anaconda seaborn

        CONFIG_NAME = "../tower.conf"
        textsize = 10

        config = reader.config(CONFIG_NAME)
        time_lim_h = int(config['buffer_time_lim_h'])
        # window_min = float(config['hf_window_min'])
        # window = np.int(float(config['hf_window_min'])*60*frequency)
        # fine_curves = False

        dbfile = f"{config['dbpath']}/{config['dbfile']}"
        cur = reader.db_init(dbfile)

        equipments = reader.bd_get_table_df(cur,f"SELECT equipment \
            FROM equipment \
            WHERE tower_name='{tower_name}' AND height='{level}' AND type='{eq_type}' AND status='online' \
            GROUP BY equipment \
            ORDER BY equipment ASC")

        var_names = reader.bd_get_table_df(cur,f"SELECT name \
            FROM variables \
            WHERE tower_name='{tower_name}' AND height='{level}' AND type='{eq_type}' AND status='online'")
        nvars = len(var_names)

        today_datetime = datetime.now() # uncomment
        yesterday = today_datetime - timedelta(hours=time_lim_h) # uncomment


        # PLOT #######################################################################################################
        fig, axs = plt.subplots(nrows=nvars,ncols=1,figsize=(10, nvars*2.5)) # sharex=True, sharey=True
        plt.tight_layout()

        for index,eq in equipments.iterrows():

            equipment = eq['equipment']

            buffer_file = Path(config['buffer_path'],f'{tower_name}_{equipment}_BUFFER.npz')
            data, status = reader.read_buffer(buffer_file)

            if status: 
                today_ind = data.index.searchsorted(today_datetime) 
                dtime = (today_datetime - data.index[-1]).total_seconds()/(60*60)
                      
                if dtime >= time_lim_h or today_ind == 0: 
                    status = False
                else:
                    data = data[yesterday:today_datetime]

            var_details = reader.bd_get_table_df(cur,f"SELECT name,short_name,long_name,description,units,missing_value \
                FROM variables \
                WHERE tower_name='{tower_name}' AND height='{level}' AND equipment='{equipment}' AND status='online'")

            for index, row in var_details.iterrows():

                var_name = row['name']

                # var_descr = reader.bd_get_table_df(cur,f"SELECT equipment,short_name,long_name,units,missing_value \
                #     FROM variables \
                #     WHERE tower_name='{tower_name}' AND height='{level}' AND equipment='{equipment}' AND status='online'")

                if not status:
                    print(f"Working on var   named: {var_name:>10} {level:4.1f} {equipment:<7} NO DATA! ")
                    axs[index].text(0.5, 0.5, f"NO DATA", fontsize=textsize+3, va="center", ha="center", transform=axs[index].transAxes)
                    axs[index].set_ylabel(f"{row['long_name']}, {row['units']}", color="black", fontsize=textsize)
                    # print(index,len(fig.axes))
                    # axs[index].axis('off')
                else:
                    for indexv, var in var_descr.iterrows():

                        print(f"Working on var   named: {var_name:>10} {level:4.1f} {equipment:<5} OK ")
                        axs[index].plot(data[var_name], linewidth=2.0, color="tab:blue")
                        axs[index].set_ylabel(f"{row['long_name']}, {row['units']}", color="black", fontsize=textsize)

        # DESIGN #######################################################################################################
        # axs = [ax1,ax2,ax3,ax4]
        ticks = pd.date_range(start=yesterday, end=today_datetime, freq='3H' ).round("3H") # inclusive='neither'
        labels = ticks.strftime('%H:%M\n%d %b') # inclusive='neither'


        for ii, ax in enumerate(axs):

            # ax.yaxis.set_label_position('left')
            ax.tick_params(direction='in')
            ax.grid(axis='both', color='tab:grey',linestyle=':', linewidth=0.5)
            # ax.axes.get_xaxis().set_visible(False)
            ax.set_xticklabels([])
            ax.set_xticks(ticks)
            ax.set_xlabel("")
            ax.set_xlim( (ticks[0], ticks[-1]) )

            yticks = ax.get_yticks()
            yticks_min = np.floor(np.min(yticks))
            yticks_max = np.ceil(np.max(yticks))
            yticks = np.linspace(yticks_min, yticks_max, 11)
            ax.set_ylim( ( yticks_min, yticks_max ))
            ax.set_yticks(yticks[::2])
            ax.set_yticklabels(yticks[::2], fontsize=textsize)


            if ii == 0:
                ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                # ax.set_xlabel('X LABEL') 
                # ax.xaxis.tick_top()   
                # ax.xaxis.set_label_position('top')
                # ax.yaxis.set_label_position('left')
                # ax.axes.get_xaxis().set_visible(True)
                # ax.set_xticks(ticks)
                # ax.set_xlabel(f"Time, UTC", color="black")
                ax.set_xticklabels(ticks.strftime('%H:%M'), fontsize=textsize )
                # ax.set_xticklabels(labels, fontsize=textsize )

            if ii == len(axs)-1:
                ax.tick_params(direction='in', which='both',labelleft=True,labelright=False,labelbottom=True,right=False,left=True)
                # ax.yaxis.set_label_position('left')
                # ax.axes.get_xaxis().set_visible(True)
                ax.set_xticks(ticks)
                ax.set_xlabel(f"Time, UTC", color="black")
                ax.set_xticklabels(labels, fontsize=textsize )

            if not status:
                ax.set_yticklabels([])

        plt.subplots_adjust(wspace=0, hspace=0.05)
        # plt.tight_layout()
        plt.savefig(figname, dpi=150)
        # plt.show()

        return True


    @staticmethod
    def simple(data, label=None):
        ''' Plot test results '''
        
        if label == None:
            plt.plot(data)
        else:
            plt.plot(data, label=label)

        return plt

    
    # @staticmethod
    # def web_accustic_stat(data, frequency, window, plt, ax0, ax1, label): # OLD
    #     from scipy import signal

    #     colors = ['darkgoldenrod','black','slategrey'] # all, hist_edgecolor, quality
    #     # colors = ['orangered','black','slategrey'] # all, hist_edgecolor, quality
    #     fonts  = [8,9] # all, titles

    #     # Fast quality check:
    #     qtke = data.tke.isna().sum()/data.tke.count()*100
    #     # qtmp = data.tempp.isna().sum()/data.tempp.count()*100

    #     # Get rid of NaN. Rude but fast.
    #     tke = data.tke.dropna()
    #     temp = data.tempp.dropna()

    #     if qtke >= 10:
    #         label = label + " BAD DATA!!!"

    #     f, Pxx = signal.welch(tke, frequency, nperseg=window)
    #     ax0.plot(f, Pxx, label=label, alpha=0.7)
    #     ax0.grid(linestyle=':', alpha=1, linewidth=0.8)
    #     ax0.tick_params(axis='both', labelsize=fonts[0], direction='in')
    #     ax0.tick_params(axis='y')
    #     ax0.tick_params(which='major', length=4, direction='in')
    #     ax0.tick_params(which='minor', length=2, direction='in')
    #     # ax0.ylim([0.5e-3, 1])
    #     ax0.set_yscale('log')
    #     ax0.set_xscale('log')
    #     ax0.set_xlabel('frequency [Hz]')
    #     ax0.set_ylabel('PSD [$m^{2} s^{-2}/Hz$]')
    #     ax0.set_title('Kinetic Energy')
        
    #     f, Pxx = signal.welch(temp, frequency, nperseg=window)
    #     ax1.plot(f, Pxx, label=label, alpha=0.7)
    #     ax1.grid(linestyle=':', alpha=1, linewidth=0.8)
    #     ax1.tick_params(axis='both', labelsize=fonts[0], direction='in')
    #     ax1.tick_params(axis='y')
    #     ax1.tick_params(which='major', length=4, direction='in')
    #     ax1.tick_params(which='minor', length=2, direction='in')
    #     # ax1.ylim([0.5e-3, 1])
    #     ax1.set_yscale('log')
    #     ax1.set_xscale('log')
    #     ax1.set_xlabel('frequency [Hz]')
    #     ax1.set_ylabel('PSD [$K/Hz$]')
    #     ax1.set_title('Acoustic temperature')
        
    #     # plt.suptitle(f"", fontsize=16)

    #     return plt, ax0, ax1


class reader:

    @staticmethod
    def db_init(dbfile):  
        import sqlite3
        from sqlite3 import Error

        conn = None

        try:
            conn = sqlite3.connect(dbfile)
        except Error as e:
            print(e)
        
        return conn.cursor()



    @staticmethod
    def create_connection(dbfile):
        import sqlite3
        from sqlite3 import Error

        conn = None

        try:
            conn = sqlite3.connect(dbfile)
        except Error as e:
            print(e)
        
        return conn, conn.cursor()


    @staticmethod
    def bd_update(conn, cur, request):
        """
        """
        cur.execute(request)
        conn.commit()

    @staticmethod
    def bd_get_table(cur,request):
        import sqlite3
        
        query = cur.execute(request)

        result = []
        for ii in query:
            result.append(ii)

        result = [item for t in result for item in t]

        return result

    @staticmethod
    def bd_get_table_df(cur,request):
        import sqlite3
        
        query = cur.execute(request)

        result = []
        for ii in query:
            result.append(ii)

        columns = request.split('SELECT')[1]
        columns = columns.split('FROM')[0]
        columns = columns.split(',')
        columns = [item.strip() for item in columns]

        d = {}
        for i, col in enumerate(columns):
            d[col] = [item[i] for item in result]

        result = pd.DataFrame(data=d, columns = columns)

        return result
=======
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
>>>>>>> d6fdfdc7f5d11eab379daaf4327675357a67ecea

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
<<<<<<< HEAD

        cnames = []
        for item in npz.files:
            df[item] = npz[item]
            cnames.append(item)
        df['datetime'] = pd.to_datetime(df['time'],unit='s')        
        df = df.set_index('datetime')
        df = df.drop(['time'], axis=1)
        df.columns = cnames[:-1]

        return df

    @staticmethod
    def read_buffer(buffer_file):
        my_file = Path(buffer_file)

        if my_file.is_file():

            status = True 
            result = reader.buffer(buffer_file)

        else:

            equipment = my_file.stem.split('_')[1]
            result = data.create_empty_pd(equipment)
            status = False

        return result, status

    @staticmethod
    def config(path):

        import configparser

        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation()
        )

        my_file = Path(path)
        if my_file.is_file():
            config.read(path)
        else:
            print(f"tl.reader.config: FILE {path} NOT FOUND!")
            exit()

        settings = dict(config["paths"])

        return settings

    # @staticmethod
    # def db_init(dbfile):
    #     import sqlite3

    #     conn = sqlite3.connect(dbfile)
    #     return conn.cursor()


class data:

    @staticmethod
    def check4dataflow(data,status):

        window = '15min' # if filled with NaN => bad data
        state = 'bad'
        check_interval_hr = 1 # ищем пустые значения в данных за 'check_interval_hr' час

        # если фала BUFFER нет вообще, но сразу помечаем
        if status:
            data.sort_index(inplace=False)
            
            dtime = datetime.utcnow() - data.index[-1]
            
            # Если за последний час есть данные, то проходим эту стадию
            if dtime < timedelta(hours=check_interval_hr):
                state = 'good'

                stime = datetime.utcnow() - timedelta(hours=check_interval_hr)
                etime = datetime.utcnow()
                
                data = data[stime:etime]
                
                counts = data.iloc[:, 1].resample(window).count()

                if counts.isna().sum() > 0:
                    state = 'bad'

        return state

    # @staticmethod
    # def get_lastupdate_data_l0(stname, equipment):

    #     config = reader.config(CONFIG_NAME)

    #     buffer_file = Path(config['buffer_path'],f"{stname}_{equipment}_BUFFER.npz" )
    #     files = sorted(Path(f"{config['l0_path']}/{stname}/{equipment}/").rglob("{stname}_{equipment}*.nc"))
    #     if buffer_file.is_file():

    #         data = np.load(buffer_file)
    #         time_from_data = data['time'][-1]

    #     else:

    #         if len(files) != 0:
    #             fin = files[-1] # newest(config['l0_path']+"/"+stname+"_"+equipment+"*.nc")
    #             f = nc.Dataset(fin)
    #             time_from_data = f.variables['time'][-1]
    #             f.close()
    #         else: # No files found at all (SHOULD NEVER HAPPEN!)
    #             time_from_data = 0
    #             return ['NO FILES FOUND!!!', '...', True]


    #     lastupdate_time = datetime.fromtimestamp(time_from_data)

    #     lastupdate_dtime = datetime.utcnow() - lastupdate_time
    #     lastupdate_dtime -= timedelta(microseconds=lastupdate_dtime.microseconds)
    #     lastupdate_time_str = lastupdate_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    #     hours = lastupdate_dtime.total_seconds() / (60 * 60)

    #     alert = False
    #     if hours > 1:
    #         alert = True


    #     return [lastupdate_time_str, lastupdate_dtime, alert]

    @staticmethod
    def get_lastupdate_rrd(stname):
        import rrdtool
        
        config = reader.config(CONFIG_NAME)
        
        lastupdate = rrdtool.lastupdate(config['rrdpath']+"/"+stname+".rrd")
        lastupdate_time = lastupdate["date"]
        lastupdate_dtime = datetime.utcnow() - lastupdate_time
        lastupdate_dtime -= timedelta(microseconds=lastupdate_dtime.microseconds)
        lastupdate_time_str = lastupdate_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        minutes = lastupdate_dtime.total_seconds() / (60)

        alert = False
        if minutes > 3:
            alert = True

        return [lastupdate_time_str, lastupdate_dtime, alert]

    @staticmethod
=======
        for item in npz.files:
           df[item] = npz[item]

        df['datetime'] = pd.to_datetime(df['time'],unit='s')        
        df = df.set_index('datetime')
        df = df.drop(['time'], axis=1)
        return df

class data:

    @staticmethod
>>>>>>> d6fdfdc7f5d11eab379daaf4327675357a67ecea
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
<<<<<<< HEAD
    def create_empty_pd(eq):
=======
    def create_empty_pd():
>>>>>>> d6fdfdc7f5d11eab379daaf4327675357a67ecea
        ''' VERY BAD IDEA!!!!!!!!!!!!!!!!!!!!!!!!!!
        Improve ASAP! Done for no buffer file case.
        '''

<<<<<<< HEAD
        # time = pd.date_range(start='1/1/1900', end='1/2/1900', freq='S')
        ntime = 0 # len(time)
        if eq[0] == "M":
            df = pd.DataFrame({'datetime': datetime.now(),
                'u':    np.full(ntime, np.nan),
                'v':    np.full(ntime, np.nan),
                'w':    np.full(ntime, np.nan),
                'temp': np.full(ntime, np.nan),
                'rh':   np.full(ntime, np.nan),
                'year': np.full(ntime, np.nan)})
        elif eq[0] == "A":
            df = pd.DataFrame({'datetime': datetime.now(),
                'u':    np.full(ntime, np.nan),
                'v':    np.full(ntime, np.nan),
                'w':    np.full(ntime, np.nan),
                'temp': np.full(ntime, np.nan),
                'e1':   np.full(ntime, np.nan),
                'e2':   np.full(ntime, np.nan),
                'e3':   np.full(ntime, np.nan),
                'e4': np.full(ntime, np.nan)})

=======
        time = pd.date_range(start='1/1/1900', end='1/2/1900', freq='S')
        ntime = len(time)
        df = pd.DataFrame({'datetime': time,
            'u':    np.full(ntime, np.nan),
            'v':    np.full(ntime, np.nan),
            'w':    np.full(ntime, np.nan),
            'temp': np.full(ntime, np.nan),
            'rh':   np.full(ntime, np.nan),
            'year': np.full(ntime, np.nan)})
>>>>>>> d6fdfdc7f5d11eab379daaf4327675357a67ecea
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

