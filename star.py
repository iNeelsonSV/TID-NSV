class Star():
    
    '''Initiation function'''

    def __init__(self, name, tic_id):
        self.name = name
        self.tic_id = tic_id
        self.sectors = []
        self.times = []
        self.fluxes = []
        self.fluxM_state = False
        self.planet = False
        self.complete = False
   
    '''Get data function'''

    def get_data(self):

        from IPython.display import Image
        import warnings
        warnings.filterwarnings('ignore')
        import eleanor
        import numpy as np
        from astropy import units as u
        import matplotlib.pyplot as plt
        from astropy.coordinates import SkyCoord
        import pandas as pd
        import math

        self.star_info = eleanor.multi_sectors(tic = self.tic_id, sectors = 'all', tc = True)

        for i in range(len(self.star_info)):
            self.data = eleanor.TargetData(self.star_info[i], height=15, width=15, bkg_size=31, regressors='corner')
                
            self.sectors.append(self.star_info[i].sector)   

            self.times.append(self.data.time) 
            self.fluxes.append(self.data.corr_flux/np.nanmedian(self.data.corr_flux))

        self.merged_time = np.concatenate((self.times))
        self.merged_flux = np.concatenate((self.fluxes))

        # Lightcurves

        if len(self.sectors) > 1:

            iter = len(self.sectors)
            fig, ax = plt.subplots(1, iter, figsize=(15, 8))
            
            for k in range(len(self.sectors)):
                ax[k].plot(self.times[k], self.fluxes[k])
            fig.set_figheight(8)
            fig.set_figwidth(15)
            plt.suptitle(f'{self.name} - {self.tic_id} - raw data')
            fig.text(0.5, 0.04, 'Time [BJD - 2457000]', ha='center', va='center')
            fig.text(0.06, 0.5, 'Normalized Flux', ha='center', va='center', rotation='vertical')
            plt.savefig(fr'your path goes here\{self.name}-raw.jpg')
            plt.close()

        elif len(self.sectors) == 1:

            plt.figure(figsize=(15, 8))

            plt.plot(self.data.time, self.data.corr_flux/np.nanmedian(self.data.corr_flux))

            plt.ylabel('Normalized Flux')
            plt.xlabel('Time [BJD - 2457000]')
            plt.title(f'{self.name} - {self.tic_id} - raw data')
            plt.savefig(fr'your path goes here\{self.name}-raw.jpg')
            plt.close()

    
    '''Mask data function'''

    def mask_data(self, lower_bound, upper_bound):

        import numpy as np
        import matplotlib.pyplot as plt

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        self.flux_masked = []

        for i in self.merged_flux:
            if self.upper_bound > i > self.lower_bound:
                self.flux_masked.append(i)
            else:
                self.flux_masked.append(np.nan)

        self.fluxM = np.array(self.flux_masked)
        self.fluxM_state = True

        self.fluxS = []

        for j in self.fluxes:
            self.fluxS_masked = []
            for k in j:
                if self.upper_bound > k > self.lower_bound:
                    self.fluxS_masked.append(k)
                else:
                    self.fluxS_masked.append(np.nan)
            self.fluxS.append(np.array(self.fluxS_masked))

        # Masked lightcurves

        if len(self.sectors) > 1:

            iter = len(self.sectors)
            fig, ax = plt.subplots(1, iter, figsize=(15, 8))
            
            for k in range(len(self.sectors)):
                ax[k].plot(self.times[k], self.fluxes[k])
                ax[k].plot(self.times[k], self.fluxS[k])
                ax[k].set_ylim([0.975,1.025])
            plt.suptitle(f'{self.name} - {self.tic_id} - masked data')
            fig.text(0.5, 0.04, 'Time [BJD - 2457000]', ha='center', va='center')
            fig.text(0.06, 0.5, 'Normalized Flux', ha='center', va='center', rotation='vertical')
            plt.savefig(fr'your path goes here\{self.name}-masked.jpg')
            plt.close()
            
        elif len(self.sectors) == 1:

            plt.figure(figsize=(15, 8))

            plt.plot(self.data.time, self.fluxM, color = 'r')

            plt.ylabel('Normalized Flux')
            plt.xlabel('Time [BJD - 2457000]')
            plt.title(f'{self.name} - {self.tic_id} - masked data')
            plt.savefig(fr'your path goes here\{self.name}-masked.jpg')
            plt.close()

    '''TLS function'''

    def run_tls(self):
        
        from transitleastsquares import (
        transitleastsquares,
        cleaned_array,
        catalog_info,
        transit_mask
        )
        
        ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID = self.tic_id)
        print('Searching with limb-darkening estimates using quadratic LD (a,b)=', ab)
        model = transitleastsquares(self.merged_time, self.fluxM)
        self.results = model.power(u=ab, period_max = 27)

        return

    '''Vetting routine function'''

    def vettingRoutine(self, depthThreshold = 50):

        self.depthThreshold = 50

        import numpy as np
        from transitleastsquares import (
        transitleastsquares,
        cleaned_array,
        catalog_info,
        transit_mask
        )

        if self.results.snr >= 7.1:
            print('SNR test... Passed')
            print('SNR: ', self.results.snr)
            self.snr_results = 'A'
        else:
            print('SNR test... Failed')
            print('SNR: ', self.results.snr)
            self.snr_results = 'F'


        if self.results.SDE >= 10:
            print('SDE test... Passed')
            print('SDE: ', self.results.SDE)
            self.sde_results = 'A'    
        else:
            print('SDE test... Failed')
            print('SDE: ', self.results.SDE)
            self.sde_results = 'F'

        if self.results.transit_depths is np.nan:

            self.max_deviation = 0
            self.avg_deviation = 0
            self.depth_results = 'F'

        else:
            self.deltanan = self.results.transit_depths[~np.isnan(self.results.transit_depths)]
            self.delta = 1 - self.deltanan
            self.avgDelta = np.nanmean(self.delta)
            self.deviation = (self.delta - self.avgDelta)/self.avgDelta
            if len(self.deviation) >= 1:
                self.max_deviation = max(np.absolute(self.deviation))
                self.avg_deviation = np.nanmean(np.absolute(self.deviation))
            else:
                self.max_deviation = np.nan
                self.avg_deviation = np.nan

            if self.max_deviation*100 <= self.depthThreshold:
                print('Depth test... Passed')
                print('Max deviation: ', self.max_deviation)
                print('Average deviation: ', self.avg_deviation)
                self.depth_results = 'A'
            else:
                print('Depth test... Failed')
                print('Max deviation: ', self.max_deviation)
                print('Average deviation: ', self.max_deviation)
                self.depth_results = 'F'

        return

    '''Definitive test for stars'''

    def transits_test(self):

        if self.snr_results == 'F' or self.sde_results == 'F' or self.depth_results == 'F':
            print("There's probably no transiting planet in here")
            self.results.distinct_transit_count = '-'
            self.transits_results = '-'
            self.radius = '-'
            self.results.period = '-'
            self.results.duration = '-'
            self.complete = True
       
        else:
            if self.results.distinct_transit_count > 1:
                self.planet = True
                self.transits_results = 'A'

            else:
                self.radius = '-'
                self.results.period = '-'
                self.results.duration = '-'
                self.transits_results = 'F'
                self.complete = True


    '''Results function'''

    def get_results(self):

        import math
        from transitleastsquares import catalog_info, transit_mask
        import matplotlib.pyplot as plt
        import numpy as np

        if self.planet is True:
            ab, mass, mass_min, mass_max, radius, radius_min, radius_max = catalog_info(TIC_ID = self.tic_id)

            self.depth = 1 - self.results.depth
            self.radius = radius*math.sqrt(self.depth)*109

            print('Period', format(self.results.period, '.5f'), 'd at T0=', self.results.T0)
            print(len(self.results.transit_times), 'transit times in time series:', ['{0:0.5f}'.format(i) for i in self.results.transit_times])
            print('Number of data points during each unique transit', self.results.per_transit_count)
            print('The number of transits with intransit data points', self.results.distinct_transit_count)
            print('The number of transits with no intransit data points', self.results.empty_transit_count)
            print('Transit depth', format(self.results.depth, '.5f'), '(at the transit bottom)')
            print('Transit duration (days)', format(self.results.duration, '.5f'))
            print('Transit depths (mean)', self.results.transit_depths)
            print('Transit depth uncertainties', self.results.transit_depths_uncertainties)
            print('SDE', self.results.SDE)
            print('SNR', self.results.snr)
            print('Radius: ', self.radius, ' [earth radius]')

            # Periodogram

            plt.figure(figsize=(15, 8))
            plt.axvline(self.results.period, alpha=0.4, lw=3)
            plt.xlim(np.min(self.results.periods), np.max(self.results.periods))
            for n in range(2, 10):
                plt.axvline(n*self.results.period, alpha=0.4, lw=1, linestyle="dashed")
                plt.axvline(self.results.period / n, alpha=0.4, lw=1, linestyle="dashed")
            plt.ylabel(r'SDE')
            plt.xlabel('Period (days)')
            plt.plot(self.results.periods, self.results.power, color='black', lw=0.5)
            plt.xlim(0, max(self.results.periods))
            plt.title(f'Periodogram - {self.name} ({self.tic_id})')
            plt.savefig(fr'your path goes here\{self.name}-periodogram.jpg')
            plt.close()

            # Phase-folded lightcurve

            plt.figure(figsize=(15, 8))
            plt.plot(self.results.model_folded_phase, self.results.model_folded_model, color='red')
            plt.scatter(self.results.folded_phase, self.results.folded_y, color='blue', s=10, alpha=0.5, zorder=2)
            plt.xlabel('Phase')
            plt.ylabel('Relative flux')
            plt.title(f'Phase-folded lightcurve - {self.name} ({self.tic_id})')
            plt.savefig(fr'your path goes here\{self.name}-phaselc.jpg')
            plt.close()

            # Best fit plotted on the lightcurve

            if len(self.sectors) > 1:
                iter = len(self.sectors)
                fig, ax = plt.subplots(1, iter, figsize = (15, 8))

                for k in range(len(self.sectors)):

                    in_transit = transit_mask(self.merged_time, self.results.period, self.results.duration, self.results.T0)
                    ax[k].scatter(self.merged_time[in_transit], self.fluxM[in_transit], color='red', s=2, zorder=0)
                    ax[k].scatter(self.merged_time[~in_transit], self.fluxM[~in_transit], color='blue', alpha=0.5, s=2, zorder=0)
                    ax[k].plot(self.results.model_lightcurve_time, self.results.model_lightcurve_model, alpha=0.5, color='red', zorder=1)
                    ax[k].set_ylim([0.975,1.025])
                    ax[k].set_xlim([self.times[k].min(), self.times[k].max()])
                fig.text(0.5, 0.04, 'Time [days]', ha='center', va='center')
                fig.text(0.06, 0.5, 'Normalized Flux', ha='center', va='center', rotation='vertical')
                plt.suptitle(f'Best fit - {self.name} ({self.tic_id}) ')
                plt.savefig(fr'your path goes here\{self.name}-bestf.jpg')
                plt.close()

            elif len(self.sectors) == 1:

                plt.figure(figsize=(15, 8))

                in_transit = transit_mask(self.merged_time, self.results.period, self.results.duration, self.results.T0)
                plt.scatter(self.merged_time[in_transit], self.fluxM[in_transit], color='red', s=2, zorder=0)
                plt.scatter(self.merged_time[~in_transit], self.fluxM[~in_transit], color='blue', alpha=0.5, s=2, zorder=0)
                plt.plot(self.results.model_lightcurve_time, self.results.model_lightcurve_model, alpha=0.5, color='red', zorder=1)
                plt.ylim(0.975,1.025)
                plt.xlabel('Time [days]')
                plt.ylabel('Relative flux')
                plt.title(f'Best fit - {self.name} ({self.tic_id}) ')
                plt.savefig(fr'your path goes here\{self.name}-bestf.jpg')
                plt.close()
            

            self.complete = True

        return