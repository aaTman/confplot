import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
from datetime import datetime, timedelta
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
import calendar, math, sys, os
import datetime as dt
import requests
from bs4 import BeautifulSoup
import pdb
import scipy.ndimage as ndimage
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import random

##Initiate variables used in multiple functions
scriptstart = dt.datetime.utcnow()
now = datetime.utcnow()
nowetc = datetime.now()
lats = np.linspace(23,52,30)
lons = np.linspace(233,295,63)
anomcolors=[]


##Color map provided by Trevor Alcott
anomcolors.append([240,140,250])
anomcolors.append([200,40,160])
anomcolors.append([120,0,200])
anomcolors.append([90,0,80])
anomcolors.append([0,50,255])
anomcolors.append([100,100,255])
anomcolors.append([180,180,255])
anomcolors.append([235,235,255])
anomcolors.append([255,255,255])
anomcolors.append([255,255,255])
anomcolors.append([245,255,160])
anomcolors.append([255,255,20])
anomcolors.append([240,200,40])
anomcolors.append([180,140,30])
anomcolors.append([100,0,0])
anomcolors.append([170,0,0])
anomcolors.append([255,0,0])
anomcolors.append([255,170,170])
anomcolors2=anomcolors[9:18]

anomcolors=(np.array(anomcolors)/255.0).tolist()
anomcolors2=(np.array(anomcolors2)/255.0).tolist()
anomcmap=ListedColormap(anomcolors,name='anommap',N=None)
anomcmap2=ListedColormap(anomcolors2,name='anommap2',N=None)


##Gradient fuction finds magnitude of climatology gridpoint at point (y,z), i.e. subtracts and squares (x+1, x-1), (y+1, y-1), adds those and sqrt the resulting value.
##Corners of the plot are created with (x+1,x), (y+1,y), (x,x-1), (y,y-1)
def gradFunc(x,y,z,mean630):
    a=mean630[x,y,z]
    if y == 0 and z == 0:
        grad=math.sqrt((mean630[x,y+1,z]-a)**2+(mean630[x,y,z+1]-a)**2)
    elif y == 29 and z == 62:
        grad=math.sqrt((mean630[x,y-1,z]-a)**2+(mean630[x,y,z-1]-a)**2)
    elif y == 29:
        grad=math.sqrt((mean630[x,y,z+1]-mean630[x,y,z-1])**2+(mean630[x,y-1,z]-a)**2)
    elif z == 62: 
        grad=math.sqrt((mean630[x,y+1,z]-mean630[x,y-1,z])**2+(mean630[x,y,z-1]-a)**2)
    elif y == 0:
        grad=math.sqrt((mean630[x,y,z+1]-mean630[x,y,z-1])**2+(mean630[x,y+1,z]-a)**2)
    elif z == 0:
        grad=math.sqrt((mean630[x,y+1,z]-mean630[x,y-1,z])**2+(mean630[x,y,z+1]-a)**2)
    else:
        grad=math.sqrt((mean630[x,y+1,z]-mean630[x,y-1,z])**2+(mean630[x,y,z+1]-mean630[x,y,z-1])**2)
    return grad
    
##Realtime GEFS gradient function
def gfrt(y,z,fm):
    a=fm[y,z]
    if y == 0 and z == 0:
        rtgm=math.sqrt((fm[y+1,z]-a)**2+(fm[y,z+1]-a)**2)
    elif y == 29 and z == 62:
        rtgm=math.sqrt((fm[y-1,z]-a)**2+(fm[y,z-1]-a)**2)
    elif y == 29:
        rtgm=math.sqrt((fm[y,z+1]-fm[y,z-1])**2+(fm[y-1,z]-a)**2)
    elif z == 62: 
        rtgm=math.sqrt((fm[y+1,z]-fm[y-1,z])**2+(fm[y,z-1]-a)**2)
    elif y == 0:
        rtgm=math.sqrt((fm[y,z+1]-fm[y,z-1])**2+(fm[y+1,z]-a)**2)
    elif z == 0:
        rtgm=math.sqrt((fm[y+1,z]-fm[y-1,z])**2+(fm[y,z+1]-a)**2)
    else:
        rtgm=math.sqrt((fm[y+1,z]-fm[y-1,z])**2+(fm[y,z+1]-fm[y,z-1])**2)
    return rtgm
    
def boxfuncg(x,y,z,q,mean630,meanG,stdG,gradzscore,gboxbin,gm,w,boxbin,spread630):
    if y+q < 0:
        return np.nan
    if z+w < 0:
        return np.nan
    if y+q >=30 or z+w >=63:
        return np.nan
    gradMCli = ((gm[x,y+q,z+w] - meanG)/stdG)
    if ((gradzscore[y,z]-0.5) <= gradMCli <= (gradzscore[y,z]+0.5)):
        
        gboxbin[(631+x*4),y,z]=gm[x,y+q,z+w]
       
        
    
    
    return gboxbin[(631+x*4),y,z]
    
def boxfuncgs(x,y,z,q,mean630,meanG,stdG,gradzscore,gboxbin,gm,w,boxbin,spread630, c):
    if y+q < 0:
        return np.nan
    if z+w < 0:
        return np.nan
    if y+q >=30 or z+w >=63:
        return np.nan
    u = 631+x*4   
    gradMCli = ((gm[x,y+q,z+w] - meanG)/stdG)
    if ((gradzscore[y,z]-1) <= gradMCli <= (gradzscore[y,z]+1)):
        
        gboxbin[u,y,z]=spread630[x,y+q,z+w]
        
    


    return gboxbin[u,y,z]
def boxfunc(x,y,z,q,mean630,meanM,stdM,zscore,boxbin,spread630,w):
    
    if y+q < 0:
        return np.nan
    if z+w < 0:
        return np.nan
    if y+q >=30 or z+w >=63:
        return np.nan
    mCliZScore = ((mean630[x,y+q,z+w] - meanM)/stdM)
    
    
    if ((zscore[y,z]-1) <= mCliZScore <= (zscore[y,z]+1)):
        
        boxbin[(631+x*4),y,z]=spread630[x,y+q,z+w]
        
    
    
    return boxbin[631+x*4,y,z]
    
def boxfunct(x,y,z,q,mean630,meanM,stdM,zscore,boxbin,spread630,w):
    if y+q < 0:
        return np.nan
    if z+w < 0:
        return np.nan
    if y+q >=30 or z+w >=63:
        return np.nan
    mCliZScore = ((mean630[x,y+q,z+w] - meanM)/stdM)
    if ((zscore[y,z]-5) <= mCliZScore <= (zscore[y,z]+5)):
        
        boxbin[(631+x*4),y,z]=spread630[x,y+q,z+w]
        
    
    
    return boxbin[(631+x*4),y,z]

def gefsLoad():
    url1 = 'http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GEFS/Global_1p0deg_Ensemble/members/latest.html'
    
    response = requests.get(url1)
    page = str(BeautifulSoup(response.content))
    start_link = page.find("a href")
    
    start_quote = page.find('"', start_link)
    end_quote = page.find('"', start_quote + 1)
    url = page[start_quote + 1: end_quote]
    
    url = url.split('html')
    endurl = url[1]
    gefs = TDSCatalog(url1+endurl)
    ds = list(gefs.datasets.values())[0]
    ncss = NCSS(ds.access_urls['NetcdfSubset'])
    
    query = ncss.query()
    print 'obtained ' + endurl[83:94] + 'z run'
    
    print 'taking time range from ' + str(now)+ ' to ' +str(now+timedelta(days=6.75))
    run = endurl[92:94]
    day = endurl[89:91]
    query.lonlat_box(233,295,23,52).time_range(now,now+timedelta(days=7))
    query.accept('netcdf4')
    query.variables('Pressure_reduced_to_MSL_msl_ens', 'Temperature_height_above_ground_ens')
    data = ncss.get_data(query)
    t = data.variables.keys()[1]
    t = str(t)
    mslp = data.variables['Pressure_reduced_to_MSL_msl_ens']
    tmps = data.variables['Temperature_height_above_ground_ens'] 
    time = data.variables[t]
    
    return mslp, tmps, time, run, day
    
def confPt(tmpMean, tmpStd, time, run, day):
    forecastSprdGrid = tmpStd.squeeze()  
    forecastMeanGrid = tmpMean.squeeze()
    
    
    month = nowetc.strftime('%m')
    for t in range (0,28):
        hour = time[t]
        hour = int(hour)
        print hour
        if hour > 168:
            break
        mean630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_tmp2m_mean.npy')
        
        spread630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_tmp2m_sprd.npy')  
        boxbin=np.empty((630*9,30,63))
        boxbin[:] = np.nan
        binnedsprd = np.empty((630))
        binnedmean = np.empty((630))
        zscore = np.empty((30,63))
        zscoresprd = np.empty((30,63))
        c10=0
        
        for y in range (0,30):
            for z in range (0,63):
        
                binnedsprd[:] = spread630[:,y,z]
        
                binnedmean[:] = mean630[:,y,z] 
                
                stdM = np.std(binnedmean)
                meanM = np.mean(binnedmean)
                stdS = np.std(binnedsprd)
                meanS = np.mean(binnedsprd)
                
                zscore[y,z] = (forecastMeanGrid[t,y,z] - meanM)/stdM
                zscoresprd[y,z] = (forecastSprdGrid[t,y,z] - meanS)/stdS
            
                
                for x in range (0,630):
                    
                    mCliZScore = ((mean630[x,y,z] - meanM)/stdM)
                    
                    
                    if ((zscore[y,z]-5) <= mCliZScore <= (zscore[y,z]+5)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                        boxbin[x,y,z] = spread630[x,y,z]
                        c10+=1
                    else:              
                        for q in range (-2,3):
                            for w in range (-2,3):
                                boxbin[(631+x*4),y,z] = boxfunct(x,y,z,q,mean630,meanM,stdM,zscore,boxbin,spread630,w)

                         
                            
                    
        mCliMean = np.nanmean(boxbin, axis=0)
        mCliStd =  np.nanstd(boxbin,axis=0)       
        
        for lat in range (0,30):
            for lon in range (0,63):
                if mCliStd[lat,lon] > 1000:
                    print lat, lon
                if mCliStd[lat,lon] <0.0005:
                   print lat, lon  
        
        confPlot = (forecastSprdGrid[t,:,:] - mCliMean)/mCliStd
        
        confPlot = np.array(confPlot)
        
        ticks = np.linspace(-4,4,num=8)
        tmpspace = range(-50,50,2)
        plt.figure(figsize=(15,10)) 
        m = Basemap(llcrnrlon=lons.min(),llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i')
        x,y = m(*np.meshgrid(lons,lats))
        
        z0 = m.contourf(x,y,confPlot,ticks,cmap=anomcmap,extend='both')
        cbar = m.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        
        cbar.set_label(r'$\frac{F_g-\mu_a}{\sigma}$')
        isoth=forecastMeanGrid[t,:,:]
        isoth=isoth - 273
        z1 = m.contour(x,y,isoth,levels=tmpspace,linestyle='--',colors='k',linewidths=0.2)
        z1.collections[25].set_linewidth(1)
        plt.clabel(z1, fontsize=6, inline_spacing=-1,fmt='%3.0f')
        m.drawcountries()
        m.drawcoastlines()
        m.drawstates()
        plt.title('GEFS SFC TEMP (C) and M-Climate Spread Anomaly \n HOUR %03d' % hour)# ++':'+strftime('%M')+ ' UTC '+ strftime('%a ') +strftime(' %b ') + strftime(' %d ') + strftime( '%Y'))
        plt.savefig('/media/taylor/Storage/mcli/realtime/tmp2m_'+str(month)+'_'+str(day)+'_'+str(run)+'z_run_'+str(hour)+'hr.png')
        print 'finished plot number ' +str(t+1)+'!'                                    
        plt.close()
                    
def confPm(mslpMean, mslpStd, time, run, day):
    forecastSprdGrid = mslpStd
    forecastMeanGrid = mslpMean
    
    month = now.strftime('%m')
    
    for t in range (0,28):
        hour = time[t]
        hour = int(hour)
        print hour
        if hour > 168:
            break
        mean630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_mslp_mean.npy')

        spread630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_mslp_sprd.npy')
        boxbin = np.empty((630*9,30,63))
        boxbin[:] = np.nan
        mbin = np.empty((630,30,63))
        binnedsprd = np.empty((630))
        binnedmean = np.empty((630))
        zscore = np.empty((30,63))
        zscoresprd = np.empty((30,63))
        c10=0
        
        for y in range (0,30):
            for z in range (0,63):
        
                binnedsprd[:] = spread630[:,y,z]
        
                binnedmean[:] = mean630[:,y,z] 
                
                stdM = np.std(binnedmean)
                meanM = np.mean(binnedmean)
                stdS = np.std(binnedsprd)
                meanS = np.mean(binnedsprd)
                
                zscore[y,z] = (forecastMeanGrid[t,y,z] - meanM)/stdM
                zscoresprd[y,z] = (forecastSprdGrid[t,y,z] - meanS)/stdS
            
                
                for x in range (0,630):
                    
                    mCliZScore = ((mean630[x,y,z] - meanM)/stdM)
                    
                    
                    if ((zscore[y,z]-1) <= mCliZScore <= (zscore[y,z]+1)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                        boxbin[x,y,z] = spread630[x,y,z]
                        c10+=1
                    else:
                        for q in range (-2,3):
                            for w in range (-2,3):
                                boxbin[(631+x*4),y,z] = boxfunc(x,y,z,q,mean630,meanM,stdM,zscore,boxbin,spread630,w)
                            
                   
        
        mCliMean = np.nanmean(boxbin, axis=0)
        mCliStd =  np.nanstd(boxbin,axis=0)
        
        for lat in range (0,30):
            for lon in range (0,63):
                if mCliStd[lat,lon] > 1000:
                    print lat, lon
                if mCliStd[lat,lon] <0.0005:
                    print lat, lon
                
                    
                                 
                    
        confPlot = (forecastSprdGrid[t,:,:] - mCliMean)/mCliStd
        
        confPlot = np.array(confPlot)
        '''for l in range (0,30):
            for m in range (0,63):
                if np.isnan(confPlot[l,m]) == True:
                    confPlot[l,m] = 10'''
                    

        
        
        ticks = np.linspace(-4,4,num=8)
        mslspace = range(900,1100,2)
        
        plt.figure(figsize=(15,10)) 
        m = Basemap(llcrnrlon=lons.min(),llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i')
    
        x,y = m(*np.meshgrid(lons,lats))

        
        z0 = m.contourf(x,y,confPlot,ticks,cmap=anomcmap,extend='both')
        cbar = m.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        
        cbar.set_label(r'$\frac{F_g-\mu_a}{\sigma}$')
        mslpLines= ndimage.filters.gaussian_filter(forecastMeanGrid[t,:,:]/100,1)
        
        z1 = m.contour(x,y,mslpLines,levels=mslspace,colors='k',linewidths=0.7)
        plt.clabel(z1, fontsize=10, inline_spacing=-1,fmt='%3.0f')
        m.drawcountries()
        m.drawcoastlines()
        m.drawstates()
        plt.title('GEFS MSLP (hPa) and M-Climate Spread Anomaly \n HOUR %03d' % hour)# ++':'+strftime('%M')+ ' UTC '+ strftime('%a ') +strftime(' %b ') + strftime(' %d ') + strftime( '%Y'))
        plt.savefig('/media/taylor/Storage/mcli/realtime/mslp_'+str(month)+'_'+str(day)+'_'+str(run)+'z_run_'+str(hour)+'hr.png')
        print 'finished plot number ' +str(t+1)+'!'                                    
        plt.close()
         
def confPt4(tmpMean, tmpStd, time, run, day):
    forecastSprdGrid = tmpStd.squeeze()  
    forecastMeanGrid = tmpMean.squeeze()
    
    
    month = nowetc.strftime('%m')
    for t in range (0,28):
        hour = time[t]
        hour = int(hour)
        print hour
        if hour > 168:
            break
        mean630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_tmp2m_mean.npy')
        
        spread630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_tmp2m_sprd.npy')  
        boxbin=np.empty((630*9,30,63))
        boxbin[:] = np.nan
        binnedsprd = np.empty((630))
        binnedmean = np.empty((630))
        zscore = np.empty((30,63))
        zscoresprd = np.empty((30,63))
        c10=0
        
        for y in range (0,30):
            for z in range (0,63):
        
                binnedsprd[:] = spread630[:,y,z]
        
                binnedmean[:] = mean630[:,y,z] 
                
                stdM = np.std(binnedmean)
                meanM = np.mean(binnedmean)
                stdS = np.std(binnedsprd)
                meanS = np.mean(binnedsprd)
                
                zscore[y,z] = (forecastMeanGrid[t,y,z] - meanM)/stdM
                zscoresprd[y,z] = (forecastSprdGrid[t,y,z] - meanS)/stdS
            
                
                for x in range (0,630):
                    
                    mCliZScore = ((mean630[x,y,z] - meanM)/stdM)
                    
                    
                    if ((zscore[y,z]-5) <= mCliZScore <= (zscore[y,z]+5)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                        boxbin[x,y,z] = spread630[x,y,z]
                        c10+=1
                    else:              
                        for q in range (-2,3):
                            for w in range (-2,3):
                                boxbin[(631+x*4),y,z] = boxfunct(x,y,z,q,mean630,meanM,stdM,zscore,boxbin,spread630,w)

                         
                            
                    
        mCliMean = np.nanmean(boxbin, axis=0)
        mCliStd =  np.nanstd(boxbin,axis=0)       
        
        for lat in range (0,30):
            for lon in range (0,63):
                if mCliStd[lat,lon] > 1000:
                    print lat, lon
                if mCliStd[lat,lon] <0.0005:
                   print lat, lon  
        
        confPlot = (forecastSprdGrid[t,:,:] - mCliMean)/mCliStd
        
        confPlot = np.array(confPlot)
        
        ticks = np.linspace(-4,4,num=8)
        tmpspace = range(-50,50,2)
        plt.figure(figsize=(15,10)) 
        m = Basemap(llcrnrlon=lons.min(),llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i')
        x,y = m(*np.meshgrid(lons,lats))

        z0 = m.contourf(x,y,confPlot,ticks,cmap=anomcmap,extend='both')
        cbar = m.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        
        cbar.set_label(r'$\frac{F_g-\mu_a}{\sigma}$')
        isoth=forecastMeanGrid[t,:,:]
        isoth=isoth - 273
        z1 = m.contour(x,y,isoth,levels=tmpspace,linestyle='--',colors='k',linewidths=0.2)
        z1.collections[25].set_linewidth(1)
        plt.clabel(z1, fontsize=6, inline_spacing=-1,fmt='%3.0f')
        m.drawcountries()
        m.drawcoastlines()
        m.drawstates()
        plt.title('GEFS SFC TEMP (C) and M-Climate Spread Anomaly \n HOUR %03d' % hour)# ++':'+strftime('%M')+ ' UTC '+ strftime('%a ') +strftime(' %b ') + strftime(' %d ') + strftime( '%Y'))
        plt.savefig('/media/taylor/Storage/mcli/realtime/tmp2m_'+str(month)+'_'+str(day)+'_'+str(run)+'z_run_'+str(hour)+'hr.png')
        print 'finished plot number ' +str(t+1)+'!'                                    
        plt.close()
                    
def confPm4(mslpMean, mslpStd, time, run, day):
    forecastSprdGrid = mslpStd
    forecastMeanGrid = mslpMean
    
    month = now.strftime('%m')
    
    for t in range (0,28):
        hour = time[t]
        hour = int(hour)
        print hour
        if hour > 168:
            break
        mean630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_mslp_mean.npy')

        spread630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_mslp_sprd.npy')
        boxbin = np.empty((630*6,30,63))
        gm = np.empty((630,30,63))
        gm2 = np.empty((630,30,63))
        rtgm = np.empty((30,63))
        boxbin[:] = np.nan
        gboxbin = np.empty((630*6,30,63))
        gboxbin[:] = np.nan
        binnedsprd = np.empty((630))
        testarray=np.empty((28,630,30,63))
        plt.figure(figsize=(15,10)) 
        ax1= np.empty((630))
        binnedmean = np.empty((630))
        zscore = np.empty((30,63))
        zscoresprd = np.empty((30,63))
        gradzscore = np.empty((30,63))
        c10=0
        
        for y in range (0,30):
            for z in range (0,63):
        
                binnedsprd[:] = spread630[:,y,z]
        
                binnedmean[:] = mean630[:,y,z] 
                for x in range(0,630):
                    gm[x,y,z] = gradFunc(x,y,z,mean630)
                
                stdG = np.std(gm[:,y,z])
                meanG = np.mean(gm[:,y,z])
                
                rtgm[y,z] = gfrt(y,z,forecastMeanGrid[t,:,:])
                
                stdM = np.std(binnedmean)
                meanM = np.mean(binnedmean)
                stdS = np.std(binnedsprd)
                meanS = np.mean(binnedsprd)
                
                gradzscore[y,z] = (rtgm[y,z] -meanG)/stdG
                zscore[y,z]= (forecastMeanGrid[t,y,z] - meanM)/stdM
                zscoresprd[y,z] = (forecastSprdGrid[t,y,z] - meanS)/stdS
             
                for x in range (0,630):
                    
                    mCliZScore = ((mean630[x,y,z] - meanM)/stdM)
                    gradMCli = ((gm[x,y,z] - meanG)/stdG)
                    
                    '''if ((gradzscore[y,z]-0.5) <= gradMCli <= (gradzscore[y,z]+0.5)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                        boxbin[x,y,z] = spread630[x,y,z]
                        c10+=1
                        if zscore[y,z] > 2:
                            testarray[t,x,y,z] = mean630[x,y,z]
                    else:
                        for q in range (-2,3):
                            for w in range (-2,3):
                                boxbin[(631+x*4),y,z] = boxfunc(x,y,z,q,mean630,meanM,stdM,zscore,boxbin,spread630,w)'''
                                
                    if ((gradzscore[y,z]-1) <= gradMCli <= (gradzscore[y,z]+1)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                        gboxbin[x,y,z] = gm[x,y,z]
                        boxbin[x,y,z] = spread630[x,y,z]
                    else:
                        for q in range (-2,3):
                            for w in range (-2,3):
                                
                                gboxbin[(631+x*4),y,z] = boxfuncg(x,y,z,q,mean630,meanG,stdG,gradzscore,gboxbin,gm,w,boxbin,spread630)         
                                boxbin[(631+x*4),y,z] = boxfuncgs(x,y,z,q,mean630,meanG,stdG,gradzscore,gboxbin,gm,w,boxbin,spread630)
        gradmean = np.mean(gm,axis=0)       
        mCliMean = np.nanmean(boxbin, axis=0)
        mCliStd =  np.nanstd(boxbin,axis=0)
        
        for lat in range (0,30):
            for lon in range (0,63):
                if mCliStd[lat,lon] > 1000:
                    print lat, lon
                if mCliStd[lat,lon] <0.0005:
                    print lat, lon
                
                    
        gradanomaly = (rtgm-np.nanmean(gboxbin,axis=0))/np.nanstd(gboxbin,axis=0)                       
        meananomaly = (forecastMeanGrid[t,:,:] - np.mean(mean630,axis=0))/np.std(mean630,axis=0)   
        spreadz = (forecastSprdGrid[t,:,:] - np.mean(spread630,axis=0))/np.std(spread630,axis=0)         
        confPlot = (forecastSprdGrid[t,:,:] - mCliMean)/mCliStd
        
        confPlot = np.array(confPlot)
        
        '''for l in range (0,30):
            for m in range (0,63):
                if np.isnan(confPlot[l,m]) == True:
                    confPlot[l,m] = 10'''
                    

        tick_nums_SA = np.linspace(-8,8,num=16)
        tick_nums = np.linspace(0,8,num=9)
        g_t = np.linspace(0,10,num=11)
        ticks = np.linspace(-4,4,num=8)
        mslspace = range(900,1100,2)

        plt.figure(figsize=(12,18)) 
        ax1 = plt.subplot2grid((3,2),(0, 0))
        ax2 = plt.subplot2grid((3,2),(0, 1))   
        ax3 = plt.subplot2grid((3,2),(1, 0))
        ax4 = plt.subplot2grid((3,2),(1, 1))
        ax5 = plt.subplot2grid((3,2),(2, 0))
        ax6 = plt.subplot2grid((3,2),(2, 1))
        ax1.set_title('Ensemble Mean and Spread')
        #pdb.set_trace()

        
        m0 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax1)
        x,y = m0(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,4)
        z0 = m0.contourf(x,y,forecastSprdGrid[t,:,:]/100,tick_nums,cmap=anomcmap2)
        cbar = m0.colorbar(z0,ticks=np.linspace(0,10,num=11),location='bottom',pad='5%')
        cbar.set_label('hPa')
       
        z1 = m0.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        m0.drawcountries()
        m0.drawcoastlines()
        m0.drawstates()
        
        ax2.set_title('Standardized Mean Anomaly')
        m1 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax2)
        
        x,y = m1(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,4)
        z0 = m1.contourf(x,y,meananomaly,tick_nums_SA,cmap=anomcmap)
        cbar = m1.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        z0.cmap.set_over('#551A8B') 
        cbar.set_label(r'$\frac{F_g-\mu}{\sigma}$')
        z1 = m1.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        plt.clabel(z1, fontsize=10, inline_spacing=-0.5,fmt='%3.0f')
        m1.drawcountries()
        m1.drawcoastlines()
        m1.drawstates()
        
        
        ax3.set_title('Standardized Spread Anomaly (Full M-Climate)')
        m2 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax3)
        ticks = np.linspace(-4,4,num=8)
        x,y = m2(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,2)
        
        z0 = m2.contourf(x,y,spreadz,ticks,cmap=anomcmap,extend='both')
        cbar = m2.colorbar(z0,ticks=np.linspace(-8,8, num=9),location='bottom',pad='5%')
        cbar.set_label(r'$\frac{F_g-\mu}{\sigma}$')
        z1 = m2.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=0.7)
        plt.clabel(z1, fontsize=10, inline_spacing=-1,fmt='%3.0f')
        m2.drawcountries()
        m2.drawcoastlines()
        m2.drawstates()
        
        ax4.set_title('Standardized Spread Anomaly (Anomaly M-Climate)')
        m3 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax4)
        ticks = np.linspace(-4,4,num=8)
        x,y = m3(*np.meshgrid(lons,lats))
        
        z0 = m3.contourf(x,y,confPlot,ticks,cmap=anomcmap,extend='both')
        cbar = m3.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        cbar.set_label(r'$\frac{F_g-\mu_a}{\sigma}$')
        z1 = m3.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=0.7)
        plt.clabel(z1, fontsize=10, inline_spacing=-1,fmt='%3.0f')
        m3.drawcountries()
        m3.drawcoastlines()
        m3.drawstates()
        
        ax5.set_title('Gradient')
        m4 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax5)
        x,y = m4(*np.meshgrid(lons,lats))
        
        z0 = m4.contourf(x,y,rtgm/100,np.linspace(0,20,num=20),cmap=anomcmap2)
        cbar=m4.colorbar(z0,ticks=np.linspace(0,20,num=11),location='bottom',pad='5%')
        z1 = m4.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        plt.clabel(z1, fontsize=10, inline_spacing=-0.5,fmt='%3.0f')
        cbar.set_label('hPa')
        m4.drawcountries()
        m4.drawcoastlines()
        m4.drawstates()
        
        ax6.set_title('Gradient Climatology')
        m5 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax6)
        x,y = m4(*np.meshgrid(lons,lats))
        
        z0 = m5.contourf(x,y,np.nanmean(gboxbin,axis=0)/100,np.linspace(0,20,num=20),cmap=anomcmap2)
        cbar=m5.colorbar(z0,ticks=np.linspace(0,20,num=11),location='bottom',pad='5%')
        cbar.set_label('hPa')
        m5.drawcountries()
        m5.drawcoastlines()
        m5.drawstates()  
        plt.savefig('/media/taylor/Storage/mcli/rt4p/mslp_'+str(month)+'_'+str(day)+'_'+str(run)+'z_run_'+str(hour)+'hr.png')
        print 'finished plot number ' +str(t+1)+'!'                                    
        plt.close('all')
        
def confPm4m(mslpMean, mslpStd, time, run, day):
    forecastSprdGrid = mslpStd
    forecastMeanGrid = mslpMean
    
    month = now.strftime('%m')
    
    for t in range (0,28):
        hour = time[t]
        hour = int(hour)
        print hour
        if hour > 168:
            break
        mean630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_mslp_mean.npy')

        spread630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_mslp_sprd.npy')
        boxbin = np.empty((630*6,30,63))
        gm = np.empty((630,30,63))
        gm2 = np.empty((630,30,63))
        rtgm = np.empty((30,63))
        boxbin[:] = np.nan
        gboxbin = np.empty((630*6,30,63))
        gboxbin[:] = np.nan
        binnedsprd = np.empty((630))
        testarray=np.empty((28,630,30,63))
        plt.figure(figsize=(15,10)) 
        ax1= np.empty((630))
        binnedmean = np.empty((630))
        zscore = np.empty((30,63))
        zscoresprd = np.empty((30,63))
        gradzscore = np.empty((30,63))
        c10=0
        
        for y in range (0,30):
            for z in range (0,63):
        
                binnedsprd[:] = spread630[:,y,z]
        
                binnedmean[:] = mean630[:,y,z] 
                for x in range(0,630):
                    gm[x,y,z] = gradFunc(x,y,z,mean630)
                
                stdG = np.std(gm[:,y,z])
                meanG = np.mean(gm[:,y,z])
                
                rtgm[y,z] = gfrt(y,z,forecastMeanGrid[t,:,:])
                
                stdM = np.std(binnedmean)
                meanM = np.mean(binnedmean)
                stdS = np.std(binnedsprd)
                meanS = np.mean(binnedsprd)
                
                gradzscore[y,z] = (rtgm[y,z] -meanG)/stdG
                zscore[y,z]= (forecastMeanGrid[t,y,z] - meanM)/stdM
                zscoresprd[y,z] = (forecastSprdGrid[t,y,z] - meanS)/stdS
             
                for x in range (0,630):
                    
                    mCliZScore = ((mean630[x,y,z] - meanM)/stdM)
                    
                                
                    if ((zscore[y,z]-1) <= mCliZScore <= (zscore[y,z]+1)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                        
                        boxbin[x,y,z] = spread630[x,y,z]
                    else:
                        for q in range (-2,3):
                            for w in range (-2,3):
                                boxbin[(631+x*4),y,z] = boxfunc(x,y,z,q,mean630,meanM,stdM,zscore,boxbin,spread630,w)
        gradmean = np.mean(gm,axis=0)       
        mCliMean = np.nanmean(boxbin, axis=0)
        mCliStd =  np.nanstd(boxbin,axis=0)
        
        for lat in range (0,30):
            for lon in range (0,63):
                if mCliStd[lat,lon] > 1000:
                    print lat, lon
                if mCliStd[lat,lon] <0.0005:
                    print lat, lon
                
                    
        gradanomaly = (rtgm-np.nanmean(gboxbin,axis=0))/np.nanstd(gboxbin,axis=0)                       
        meananomaly = (forecastMeanGrid[t,:,:] - np.mean(mean630,axis=0))/np.std(mean630,axis=0)   
        spreadz = (forecastSprdGrid[t,:,:] - np.mean(spread630,axis=0))/np.std(spread630,axis=0)         
        confPlot = (forecastSprdGrid[t,:,:] - mCliMean)/mCliStd
        
        confPlot = np.array(confPlot)
                    

        tick_nums_SA = np.linspace(-8,8,num=16)
        tick_nums = np.linspace(0,8,num=9)
        g_t = np.linspace(0,10,num=11)
        ticks = np.linspace(-4,4,num=8)
        mslspace = range(900,1100,2)

        plt.figure(figsize=(12,18)) 
        ax1 = plt.subplot2grid((3,2),(0, 0))
        ax2 = plt.subplot2grid((3,2),(0, 1))   
        ax3 = plt.subplot2grid((3,2),(1, 0))
        ax4 = plt.subplot2grid((3,2),(1, 1))
        ax5 = plt.subplot2grid((3,2),(2, 0))
        ax6 = plt.subplot2grid((3,2),(2, 1))
        ax1.set_title('Ensemble Mean and Spread')
        #pdb.set_trace()

        
        m0 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax1)
        x,y = m0(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,4)
        z0 = m0.contourf(x,y,forecastSprdGrid[t,:,:]/100,tick_nums,cmap=anomcmap2)
        cbar = m0.colorbar(z0,ticks=np.linspace(0,10,num=11),location='bottom',pad='5%')
        cbar.set_label('hPa')
       
        z1 = m0.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        m0.drawcountries()
        m0.drawcoastlines()
        m0.drawstates()
        
        ax2.set_title('Standardized Mean Anomaly')
        m1 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax2)
        
        x,y = m1(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,4)
        z0 = m1.contourf(x,y,meananomaly,tick_nums_SA,cmap=anomcmap)
        cbar = m1.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        z0.cmap.set_over('#551A8B') 
        cbar.set_label(r'$\frac{F_g-\mu}{\sigma}$')
        z1 = m1.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        plt.clabel(z1, fontsize=10, inline_spacing=-0.5,fmt='%3.0f')
        m1.drawcountries()
        m1.drawcoastlines()
        m1.drawstates()
        
        
        ax3.set_title('Standardized Spread Anomaly (Full M-Climate)')
        m2 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax3)
        ticks = np.linspace(-4,4,num=8)
        x,y = m2(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,2)
        
        z0 = m2.contourf(x,y,spreadz,ticks,cmap=anomcmap,extend='both')
        cbar = m2.colorbar(z0,ticks=np.linspace(-8,8, num=9),location='bottom',pad='5%')
        cbar.set_label(r'$\frac{F_g-\mu}{\sigma}$')
        z1 = m2.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=0.7)
        plt.clabel(z1, fontsize=10, inline_spacing=-1,fmt='%3.0f')
        m2.drawcountries()
        m2.drawcoastlines()
        m2.drawstates()
        
        ax4.set_title('Standardized Spread Anomaly (Mean Anomaly M-Climate)')
        m3 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax4)
        ticks = np.linspace(-4,4,num=8)
        x,y = m3(*np.meshgrid(lons,lats))
        
        z0 = m3.contourf(x,y,confPlot,ticks,cmap=anomcmap,extend='both')
        cbar = m3.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        cbar.set_label(r'$\frac{F_g-\mu_a}{\sigma}$')
        z1 = m3.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=0.7)
        plt.clabel(z1, fontsize=10, inline_spacing=-1,fmt='%3.0f')
        m3.drawcountries()
        m3.drawcoastlines()
        m3.drawstates()
        
        ax5.set_title('Gradient')
        m4 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax5)
        x,y = m4(*np.meshgrid(lons,lats))
        
        z0 = m4.contourf(x,y,rtgm/100,np.linspace(0,20,num=20),cmap=anomcmap2)
        cbar=m4.colorbar(z0,ticks=np.linspace(0,20,num=11),location='bottom',pad='5%')
        z1 = m4.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        plt.clabel(z1, fontsize=10, inline_spacing=-0.5,fmt='%3.0f')
        cbar.set_label('hPa')
        m4.drawcountries()
        m4.drawcoastlines()
        m4.drawstates()
        
        ax6.set_title('Gradient Climatology')
        m5 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax6)
        x,y = m4(*np.meshgrid(lons,lats))
        
        z0 = m5.contourf(x,y,np.nanmean(gboxbin,axis=0)/100,np.linspace(0,20,num=20),cmap=anomcmap2)
        cbar=m5.colorbar(z0,ticks=np.linspace(0,20,num=11),location='bottom',pad='5%')
        cbar.set_label('hPa')
        m5.drawcountries()
        m5.drawcoastlines()
        m5.drawstates()  
        plt.savefig('/media/taylor/Storage/mcli/rt4p/mslp_'+str(month)+'_'+str(day)+'_'+str(run)+'z_run_'+str(hour)+'_meananom_hr.png')
        print 'finished plot number ' +str(t+1)+'!'                                    
        plt.close('all')      
          
          




def confPm4lw(mslpMean, mslpStd, time, run, day):
    forecastSprdGrid = mslpStd
    forecastMeanGrid = mslpMean
    
    month = now.strftime('%m')
    
    for t in range (0,28):
        hour = time[t]
        hour = int(hour)
        print hour
        if hour > 168:
            break
        mean630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_mslp_mean.npy')
      
        spread630 = np.load('/media/taylor/Storage/mcli/'+str(month)+'_'+str(day)+'_/'+str(month)+'_'+str(day)+'_'+str(hour)+'z_mslp_sprd.npy')
        for x in range (0,630):
            for y in range (0,30):
                for z in range (0,63):
                    if mean630[x,y,z] < 89000:
                        mean630[x,y,z] = np.nan
                        spread630[x,y,z] = np.nan
        
        boxbin = np.empty((630*6,30,63))
        gm = np.empty((630,30,63))
        gm2 = np.empty((630,30,63))
        rtgm = np.empty((30,63))
        boxbin[:] = np.nan
        gboxbin = np.empty((630*6,30,63))
        gboxbin[:] = np.nan
        binnedsprd = np.empty((630))
        testarray=np.empty((28,630,30,63))
        gtestarray=np.empty((28,630,30,63))
        plt.figure(figsize=(15,10)) 
        ax1= np.empty((630))
        binnedmean = np.empty((630))
        zscore = np.empty((30,63))
        zscoresprd = np.empty((30,63))
        gradzscore = np.empty((30,63))
        c10=0
        
        for y in range (0,30):
            for z in range (0,63):
        
                binnedsprd[:] = spread630[:,y,z]
        
                binnedmean[:] = mean630[:,y,z] 
                for x in range(0,630):
                    gm[x,y,z] = gradFunc(x,y,z,mean630)
                
                stdG = np.nanstd(gm[:,y,z])
                meanG = np.nanmean(gm[:,y,z])
                
                rtgm[y,z] = gfrt(y,z,forecastMeanGrid[t,:,:])
                
                stdM = np.nanstd(binnedmean)
                meanM = np.nanmean(binnedmean)
                stdS = np.nanstd(binnedsprd)
                meanS = np.nanmean(binnedsprd)
                
                gradzscore[y,z] = (rtgm[y,z] -meanG)/stdG
                zscore[y,z]= (forecastMeanGrid[t,y,z] - meanM)/stdM
                zscoresprd[y,z] = (forecastSprdGrid[t,y,z] - meanS)/stdS
             
                for x in range (0,630):
                    
                    mCliZScore = ((mean630[x,y,z] - meanM)/stdM)
                    gradMCli = ((gm[x,y,z] - meanG)/stdG)
                    c=0
                    if gradMCli < 3:            
                        if ((gradzscore[y,z]-1) <= gradMCli <= (gradzscore[y,z]+1)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                            gboxbin[x,y,z] = spread630[x,y,z]
                            c+=1

                        else:
                            for q in range (-2,3):
                                for w in range (-2,3):

                                    gboxbin[(631+x*4),y,z] = boxfuncgs(x,y,z,q,mean630,meanG,stdG,gradzscore,gboxbin,gm,w,boxbin,spread630,c)
                                    if gboxbin[(631+x*4),y,z] > -5000:
                                        c+=1
                    elif gradMCli > 3:
                        if ((gradzscore[y,z]-1) <= gradMCli <= (gradzscore[y,z]+2)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                            gboxbin[x,y,z] = spread630[x,y,z]
                            c+=1

                        else:
                            for q in range (-2,3):
                                for w in range (-2,3):
                                    
                                    gboxbin[(631+x*4),y,z] = boxfuncgs(x,y,z,q,mean630,meanG,stdG,gradzscore,gboxbin,gm,w,boxbin,spread630,c)
                                    if gboxbin[(631+x*4),y,z] > -5000:
                                        c+=1
                    elif gradMCli > 5:
                        if ((gradzscore[y,z]-1) <= gradMCli <= (gradzscore[y,z]+50)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) for all cases with large anomaly (forecast, x, y), add to bin
                            gboxbin[x,y,z] = spread630[x,y,z]
                            c+=1
                        else:
                            for q in range (-2,3):
                                for w in range (-2,3):
                                    
                                    gboxbin[(631+x*4),y,z] = boxfuncgs(x,y,z,q,mean630,meanG,stdG,gradzscore,gboxbin,gm,w,boxbin,spread630,c)
                                    if gboxbin[(631+x*4),y,z] > -5000:
                                        c+=1
                if c <= 5:
                    for x in range (0,630):
                        mCliZScore = ((mean630[x,y,z] - meanM)/stdM)
                        gradMCli = ((gm[x,y,z] - meanG)/stdG)
                        c=0
                        for q in range (-4,5):
                            for w in range (-4,5):
                                if y in range (15,20):
                                    if z in range (44,49):
                                        gtestarray[t,x,:,:] = mean630[x,:,:]
                                gboxbin[(631+x*4),y,z] = boxfuncgs(x,y,z,q,mean630,meanG,stdG,gradzscore,gboxbin,gm,w,boxbin,spread630,c)
                                if gboxbin[(631+x*4),y,z] > -5000:
                                    c+=1
                                                        
        gradmean = np.nanmean(gm,axis=0)       
        mCliMean = np.nanmean(boxbin, axis=0)
        mCliStd =  np.nanstd(boxbin,axis=0)
        
        for lat in range (0,30):
            for lon in range (0,63):
                if mCliStd[lat,lon] > 1000:
                    print lat, lon
                if mCliStd[lat,lon] <0.0005:
                    print lat, lon
                
        for y in range (0,30):
            for z in range (0,63):
        
                binnedsprd[:] = spread630[:,y,z]
        
                binnedmean[:] = mean630[:,y,z] 
                for x in range(0,630):
                    gm[x,y,z] = gradFunc(x,y,z,mean630)
                
                stdG = np.nanstd(gm[:,y,z])
                meanG = np.nanmean(gm[:,y,z])
                
                rtgm[y,z] = gfrt(y,z,forecastMeanGrid[t,:,:])
                
                stdM = np.nanstd(binnedmean)
                meanM = np.nanmean(binnedmean)
                stdS = np.nanstd(binnedsprd)
                meanS = np.nanmean(binnedsprd)
                
                gradzscore[y,z] = (rtgm[y,z] -meanG)/stdG
                zscore[y,z]= (forecastMeanGrid[t,y,z] - meanM)/stdM
                zscoresprd[y,z] = (forecastSprdGrid[t,y,z] - meanS)/stdS
             
                for x in range (0,630):
                    
                    mCliZScore = ((mean630[x,y,z] - meanM)/stdM)
                    
                                
                    if ((zscore[y,z]-1) <= mCliZScore <= (zscore[y,z]+1)): #if the point of interest (case, x, y) has a zscore relative to all cases at that point (630, x, y) within 1 of the point on the forecast grid (forecast, x, y), add to bin
                        boxbin[x,y,z] = spread630[x,y,z]
        
                    else:
                        for q in range (-2,3):
                            for w in range (-2,3):                             
                                boxbin[(631+x*4),y,z] = boxfunc(x,y,z,q,mean630,meanM,stdM,zscore,boxbin,spread630,w)
                                
        gradmean = np.nanmean(gm,axis=0)       
        mCliMean = np.nanmean(boxbin, axis=0)
        mCliStd =  np.nanstd(boxbin,axis=0)
        gradMCliMean = np.nanmean(gboxbin,axis=0)
        gradMCliStd = np.nanstd(gboxbin,axis=0)
 
        for lat in range (0,30):
            for lon in range (0,63):
                if mCliStd[lat,lon] > 1000:
                    print lat, lon
                if mCliStd[lat,lon] <0.0005:
                    print lat, lon
                    
                    
        testCases(testarray,gtestarray,t)    
            
            
        forecastSprdGrid = np.array(forecastSprdGrid)            
        gradanomaly = (forecastSprdGrid[t,:,:]-gradMCliMean[:,:])/gradMCliStd[:,:]                  
        meananomaly = (forecastMeanGrid[t,:,:] - np.nanmean(mean630,axis=0))/np.nanstd(mean630,axis=0)
        gradientAnomaly = (rtgm-np.nanmean(gm,axis=0))/np.nanstd(gm,axis=0)   
        spreadz = (forecastSprdGrid[t,:,:] - np.nanmean(spread630,axis=0))/np.nanstd(spread630,axis=0)         
        confPlot = (forecastSprdGrid[t,:,:] - mCliMean)/mCliStd                                 
        lwMap = np.empty((30,63))
        lwStd = np.empty((30,63))
        confPlot = np.array(confPlot)
        for x in range (0,30):
            for y in range (0,63):
                lwMap[x,y] = gradMCliMean[x,y]*(gradMCliMean[x,y]/(gradMCliMean[x,y]+mCliMean[x,y])) + mCliMean[x,y]*(mCliMean[x,y]/(gradMCliMean[x,y]+mCliMean[x,y]))
                lwStd[x,y] = gradMCliStd[x,y]*(gradMCliStd[x,y]/(gradMCliStd[x,y]+mCliStd[x,y])) + mCliStd[x,y]*(mCliStd[x,y]/(gradMCliStd[x,y]+mCliStd[x,y]))
        LWAMAP = (forecastSprdGrid[t,:,:]-lwMap)/lwStd
                    
        tick_nums_SA = np.linspace(-8,8,num=16)
        tick_nums = np.linspace(0,8,num=9)
        g_t = np.linspace(0,10,num=11)
        ticks = np.linspace(-4,4,num=8)
        mslspace = range(900,1100,2)

        plt.figure(figsize=(12,18)) 
        ax1 = plt.subplot2grid((3,2),(0, 0))
        ax2 = plt.subplot2grid((3,2),(0, 1))   
        ax3 = plt.subplot2grid((3,2),(1, 0))
        ax4 = plt.subplot2grid((3,2),(1, 1))
        ax5 = plt.subplot2grid((3,2),(2, 0))
        ax6 = plt.subplot2grid((3,2),(2, 1))
        ax1.set_title('Ensemble Mean and Spread')


        
        m0 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax1)
        x,y = m0(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,4)
        z0 = m0.contourf(x,y,forecastSprdGrid[t,:,:]/100,tick_nums,cmap=anomcmap2)
        cbar = m0.colorbar(z0,ticks=np.linspace(0,10,num=11),location='bottom',pad='5%')
        cbar.set_label('hPa')
       
        z1 = m0.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        m0.drawcountries()
        m0.drawcoastlines()
        m0.drawstates()
        
        ax2.set_title('Standardized Mean Anomaly')
        m1 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax2)
        
        x,y = m1(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,4)
        z0 = m1.contourf(x,y,meananomaly,tick_nums_SA,cmap=anomcmap)
        cbar = m1.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        z0.cmap.set_over('#551A8B') 
        cbar.set_label(r'$\frac{F_g-\mu}{\sigma}$')
        z1 = m1.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        plt.clabel(z1, fontsize=10, inline_spacing=-0.5,fmt='%3.0f')
        m1.drawcountries()
        m1.drawcoastlines()
        m1.drawstates()
        
        
        ax3.set_title('Gradient M-Climatology')
        m2 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax3)
        ticks = np.linspace(-4,4,num=8)
        x,y = m2(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,2)
        
        z0 = m2.contourf(x,y,gradMCliMean,cmap=anomcmap,extend='both')
        cbar = m2.colorbar(z0,location='bottom',pad='5%')
        cbar.set_label(r'$\frac{F_g-\mu}{\sigma}$')
        z1 = m2.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=0.7)
        plt.clabel(z1, fontsize=10, inline_spacing=-1,fmt='%3.0f')
        m2.drawcountries()
        m2.drawcoastlines()
        m2.drawstates()
        
        ax4.set_title('Mean M-Climatology')
        m3 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax4)
        ticks = np.linspace(-4,4,num=8)
        x,y = m3(*np.meshgrid(lons,lats))
        
        z0 = m3.contourf(x,y,mCliMean,cmap=anomcmap,extend='both')
        cbar = m3.colorbar(z0,location='bottom',pad='5%')
        cbar.set_label(r'$\frac{F_g-\mu_a}{\sigma}$')
        z1 = m3.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=0.7)
        plt.clabel(z1, fontsize=10, inline_spacing=-1,fmt='%3.0f')
        m3.drawcountries()
        m3.drawcoastlines()
        m3.drawstates()
        
        ax5.set_title('Standardized Gradient Anomaly')
        m4 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax5)
        x,y = m4(*np.meshgrid(lons,lats))
        
        z0 = m4.contourf(x,y,gradientAnomaly,tick_nums_SA,cmap=anomcmap)
        cbar=m4.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        z1 = m4.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=1)
        plt.clabel(z1, fontsize=10, inline_spacing=-0.5,fmt='%3.0f')
        cbar.set_label('hPa')
        m4.drawcountries()
        m4.drawcoastlines()
        m4.drawstates()
        
        ax6.set_title('LWA of Gradient and Mean Standardized Anomaly')
        m5 = Basemap(llcrnrlon=265,llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='i',ax=ax6)
        x,y = m4(*np.meshgrid(lons,lats))
        
        z0 = m5.contourf(x,y,LWAMAP,ticks,cmap=anomcmap,extend='both')
        cbar=m5.colorbar(z0,ticks=np.linspace(-8,8,num=9),location='bottom',pad='5%')
        cbar.set_label('hPa')
        z1 = m5.contour(x,y,forecastMeanGrid[t,:,:]/100,levels=mslspace,colors='k',linewidths=0.7)
        plt.clabel(z1, fontsize=10, inline_spacing=-1,fmt='%3.0f')
        m5.drawcountries()
        m5.drawcoastlines()
        m5.drawstates()  
        plt.savefig('/media/taylor/Storage/mcli/rt4p/mslp_lwa_'+str(month)+'_'+str(day)+'_'+str(run)+'z_run_'+str(hour)+'hr.png')
        print 'finished plot number ' +str(t+1)+'!'                                    
        plt.close('all')
            
def stats(mslp,tmps):
    mslp =mslp[:,:,::-1,:]
    tmps = tmps[:,:,:,::-1,:]
    mslpMean = np.mean(mslp,axis=1)  
    mslpStd = np.std(mslp,axis=1)
    
    #gefsPlot(mslp,tmps)
    
    tmpMean = np.mean(tmps,axis=1)
    tmpStd = np.std(tmps,axis=1) 
    return mslpMean, mslpStd, tmpMean, tmpStd

#def gefsPlot(mslp, tmps):
    
        
def testCases(testarray,gtestarray,t):  
    for q in range(0,10): 
        rand=random.randint(0,629)
        randMap=testarray[t,rand,:,:]
        m = Basemap(llcrnrlon=lons.min(),llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='c')
        x,y = m(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,2)   
        z2 = m.contour(x,y,randMap/100,levels=mslspace,colors='k',linewidths=.7)
        plt.clabel(z2, fontsize=10, inline_spacing=-0.3,fmt='%3.0f')
        m.drawcountries()
        m.drawcoastlines()
        m.drawstates()
        plt.title('Random Mean Sample MSLP Map, '+str(rand))    
        plt.savefig('/media/taylor/Storage/mcli/rftest/mslp_meantest_'+str(t)+'_'+str(rand)+'.png')
        plt.close('all')
        
        randgMap=gtestarray[t,rand,:,:]
        m = Basemap(llcrnrlon=lons.min(),llcrnrlat=lats.min(),urcrnrlon=lons.max(),urcrnrlat=lats.max(),projection='cyl',resolution='c')
        x,y = m(*np.meshgrid(lons,lats))
        mslspace = range(900,1100,2)   
        z2 = m.contour(x,y,randgMap/100,levels=mslspace,colors='k',linewidths=.7)
        plt.clabel(z2, fontsize=10, inline_spacing=-0.3,fmt='%3.0f')
        m.drawcountries()
        m.drawcoastlines()
        m.drawstates()    
        plt.title('Random Gradient Sample MSLP Map, '+str(rand))  
        plt.savefig('/media/taylor/Storage/mcli/rftest/mslp_gradtest_'+str(t)+'_'+str(rand)+'.png')
        plt.close('all')
def main():
    mslp, tmps, time, run, day = gefsLoad()
    mslpMean, mslpStd, tmpMean, tmpStd = stats(mslp,tmps)
    #confPm(mslpMean, mslpStd, time, run, day)
    #confPm4m(mslpMean, mslpStd, time, run, day)
    confPm4lw(mslpMean, mslpStd, time, run, day)
    #testCases(testarray,mslpMean)
    #confPt(tmpMean, tmpStd, time, run, day)
    
if __name__ == "__main__":
    main()