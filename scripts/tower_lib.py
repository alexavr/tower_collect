# *** README
# ** INSTALLATION
# conda create --name tower jupyter xarray dask netCDF4 bottleneck pandas matplotlib numpy -y
# conda activate tower

import pandas as pd # conda install -c conda-forge pandas
import xarray as xr # conda install -c conda-forge xarray dask netCDF4 bottleneck
import matplotlib.pyplot as plt # conda install -c conda-forge matplotlib
import numpy as np # conda install -c conda-forge numpy
from pandas.tseries.frequencies import to_offset
from datetime import datetime, timedelta
from pathlib import Path

CONFIG_NAME = "/var/www/data/domains/tower.ocean.ru/html/flask/tower.conf"

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


################################################################################

class notification:

    @staticmethod
    def send2bot(TOKEN, chat_id, msg):
        from telegram import Bot

        # channel_id="@TowerMSU"
        bot = Bot(TOKEN)
        bot.send_message(chat_id, msg) 
        # import requests
        # url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={msg}"
        # status = requests.get(url).json()

        # return status

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
            ax.tick_params(direction='in', which='both')
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

        levels = [1, 10, 30, 50, 70, 100]
        ax4.set_ylim( (1, 100) )
        ax4.set_yticks( levels )
        ax4.set_yticklabels( levels )
        
        ax1.set_ylabel(f"u/v wind, ms-1", color="black", fontsize=textsize+2)
        ax2.set_ylabel(f"w wind, ms-1", color="black", fontsize=textsize+2)
        ax3.set_ylabel(f"temperature, degC", color="black", fontsize=textsize+2)
        ax4.set_ylabel(f"gaps amount, %", color="black", fontsize=textsize+2)
        ax5.set_xlabel(f"data quality\n(measures/sec)", fontsize=textsize+2)

        # plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(figname, dpi=150)
        # plt.show()

        return True

    @staticmethod
    def web_meteo(tower_name,level,eq_type, figname):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.ticker import FormatStrFormatter
        from matplotlib import colors
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import seaborn as sns # conda install -c anaconda seaborn
        import re
        import windrose

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
            WHERE tower_name='{tower_name}' AND height='{level}' AND type='{eq_type}' AND status='online' AND plot=1")
        nvars = len(var_names) 

        today_datetime = datetime.now() # uncomment
        yesterday = today_datetime - timedelta(hours=time_lim_h) # uncomment

        if nvars == 1:
            width = 4
        else:
            width = 2.0

        # PLOT #######################################################################################################
        fig, axs = plt.subplots(nrows=nvars,ncols=1,figsize=(10, nvars*width)) # sharex=True, sharey=True
        plt.tight_layout()

        try:
            tmp = len(axs)
        except:
            axs = np.atleast_1d(axs)

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
                WHERE tower_name='{tower_name}' AND height='{level}' AND equipment='{equipment}' AND status='online' AND plot=1")

            for index, row in var_details.iterrows():

                var_name = row['name']
    
                if not status:
                    print(f"Working on var   named: {var_name:>10} {level:4.1f} {equipment:<7} NO DATA! ")
                    axs[index].text(0.5, 0.5, f"NO DATA", fontsize=textsize+3, va="center", ha="center", transform=axs[index].transAxes)
                    axs[index].set_ylabel(f"{row['long_name']}, {row['units']}", color="black", fontsize=textsize)
                else:

                    color="steelblue" # Default color
                    if re.search('temp',            row['long_name'], re.IGNORECASE): color = "tomato"
                    if re.search('hum|dew',         row['long_name'], re.IGNORECASE): color = "olive"
                    if re.search('wind',            row['long_name'], re.IGNORECASE): color = "tab:grey"


                    print(f"Working on var   named: {var_name:>13} {level:4.1f} {equipment:<5} OK ")

                    var = data[var_name].rolling(window="30min").mean()

                    # ymin,ymax = axs[index].get_ylim()
                    ymin,ymax = var.min(), var.max()
                    if re.search('pressure tendency',row['long_name'], re.IGNORECASE): 


                        var_lim = np.max( [ np.abs(var.min()), np.abs(var.max()) ] )

                        var = var.resample('30min').mean()
                        var_pos = var[ var >= 0 ]
                        var_neg = var[ var <  0 ]
                        axs[index].bar(var_pos.index, var_pos, linewidth=0.9, width=0.015, edgecolor="gray", color="lightskyblue")
                        axs[index].bar(var_neg.index, var_neg, linewidth=0.9, width=0.015, edgecolor="gray", color="pink")

                        axs[index].set_ylim( ( -var_lim, var_lim ))

                        # LINE PLOT
                        # ymin = 0
                        # axs[index].plot(var, linewidth=1.5, color=color)
                        # plt.fill_between(var.index, var, ymin,
                        #                  where=(var < ymin),
                        #                  alpha=0.1, color='red', interpolate=True)
                        # plt.fill_between(var.index, var, 0,
                        #                  where=(var >= ymin),
                        #                  alpha=0.1, color='green', interpolate=True)

                        # Zero line
                        xmin, xmax = axs[index].get_xlim()
                        axs[index].plot([xmin,xmax], [0,0], linewidth=1.0, color="black")


                    else:
                        axs[index].plot(var, linewidth=1.5, color=color)
                        axs[index].fill_between(
                                x=var.index, 
                                y1=var, 
                                y2=np.floor(ymin), 
                                color=color,
                                label=var_name,
                                # ylim=[1, 100],
                                alpha=0.05)
                        # axs[index].set_ylabel(f"{row['long_name']}, {row['units']}", color="black", fontsize=textsize)
                    plt.text(0.01, 0.92, f"{row['long_name']}, {row['units']}",
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=textsize+3,
                        # weight='bold',
                        transform = axs[index].transAxes)
                    if re.search('Wind_direction', row['short_name'], re.IGNORECASE): 
                        axins = inset_axes(axs[index], 
                            width=1.6, 
                            height=1.6,
                            loc=3,
                            # bbox_transform=axs[index].transData,
                            axes_class=windrose.WindroseAxes,
                        )
                        wd = data[var_name].rolling(window="30min").mean()
                        ws = data["WD1AVG1"].rolling(window="30min").mean()
                        axins.grid(axis='both', color='tab:gray', linestyle='--', linewidth=0.3)
                        axins.set_axisbelow(True)
                        axins.bar(wd, ws, alpha=1, color="dimgray")
                        # axins.bar(wd, ws, alpha=1, edgecolor="dimgray", facecolor="dimgray")
                        axins.patch.set_alpha(0.7) # transparent backgound
                        axins.tick_params(labelleft=False, labelbottom=False)


        # DESIGN #######################################################################################################
        ticks = pd.date_range(start=yesterday, end=today_datetime, freq='3H' ).round("3H") # inclusive='neither'
        labels = ticks.strftime('%H:%M\n%d %b') # inclusive='neither'


        for ii, ax in enumerate(axs):

            # ax.yaxis.set_label_position('left')
            ax.tick_params(direction='in')
            ax.grid(axis='both', color='tab:grey',linestyle=':', linewidth=0.5)
            ax.set_xticklabels([])
            ax.set_xticks(ticks)
            ax.set_xlabel("")
            ax.set_xlim( (ticks[0], ticks[-1]) )

            yticks = ax.get_yticks()
            yticks_min, yticks_max = ax.get_ylim()
            yticks = np.linspace(yticks_min, yticks_max, 11)
            ax.set_ylim( ( yticks_min, yticks_max ))
            ax.set_yticks(yticks[1:-1:2])
            ax.set_yticklabels(yticks[1:-1:2], fontsize=textsize)


            if ii == 0:
                ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                ax.set_xticklabels(ticks.strftime('%H:%M'), fontsize=textsize )

            if ii == len(axs)-1:
                ax.tick_params(direction='in', which='both',labelleft=True,labelright=False,labelbottom=True,right=False,left=True)
                ax.set_xticks(ticks)
                ax.set_xlabel(f"Time, UTC", color="black")
                ax.set_xticklabels(labels, fontsize=textsize )

            if not status:
                ax.set_yticklabels([])

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.subplots_adjust(wspace=0, hspace=0.05)
        # plt.tight_layout()
        plt.savefig(figname, dpi=150)
        # plt.show()

        return True

    @staticmethod
    def gradient_fill(x, y, fill_color=None, ax=None, **kwargs):
        """
        Plot a line with a linear alpha gradient filled beneath it.

        Parameters
        ----------
        x, y : array-like
            The data values of the line.
        fill_color : a matplotlib color specifier (string, tuple) or None
            The color for the fill. If None, the color of the line will be used.
        ax : a matplotlib Axes instance
            The axes to plot on. If None, the current pyplot axes will be used.
        Additional arguments are passed on to matplotlib's ``plot`` function.

        Returns
        -------
        line : a Line2D instance
            The line plotted.
        im : an AxesImage instance
            The transparent gradient clipped to just the area beneath the curve.
        """
        import matplotlib.colors as mcolors
        from matplotlib.patches import Polygon

        if ax is None:
            ax = plt.gca()

        line, = ax.plot(x, y, **kwargs)
        if fill_color is None:
            fill_color = line.get_color()

        zorder = line.get_zorder()
        alpha = line.get_alpha()
        alpha = 1.0 if alpha is None else alpha

        z = np.empty((100, 1, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        z[:,:,:3] = rgb
        z[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        print(xmin, xmax, ymin, ymax)
        im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                       origin='lower', zorder=zorder)

        xy = np.column_stack([x, y])
        xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
        clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
        ax.add_patch(clip_path)
        im.set_clip_path(clip_path)

        ax.autoscale(True)
        return line, im

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
    def create_empty_pd(eq):
        ''' VERY BAD IDEA!!!!!!!!!!!!!!!!!!!!!!!!!!
        Improve ASAP! Done for no buffer file case.
        '''

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

