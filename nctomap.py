#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Colormap


# In[2]:


cur_dir = "~/IMR/echo-test/hackathon/"
files = "OOI-D20170821-T163049.nc"


# In[3]:


# Open with Xarray, select the beam group
# if we use chunk, we will get almost functionality with numpy's memmap array
# i.e., data is loaded lazyly and per-chunk

ds = xr.open_dataset(cur_dir + files, group = "Beam", chunks={'ping_time': 100})
ds


# In[4]:


da = ds['backscatter_r']
print(da)
da


# In[5]:


# Getting the variables
ping_time = da.coords['ping_time']
print(ping_time)
range_bin = da.coords['range_bin']
print(range_bin)


# In[6]:


# Acccessing single data (wrong)
print(da[0,0,0])


# In[7]:


# Acccessing single data (right)
print(da[0,0,0].load())


# In[8]:


# Accessing using labels (freq= 38khz, time = ...)
da_sub = da.loc["3.8e+04", "2017-08-21T16:30":"2017-08-21T16:31"]
da_sub


# In[9]:


# Still traditional index works
da_sub[0,0]


# In[10]:


# Prepare simrad cmap
simrad_color_table = [(1, 1, 1),
                                    (0.6235, 0.6235, 0.6235),
                                    (0.3725, 0.3725, 0.3725),
                                    (0, 0, 1),
                                    (0, 0, 0.5),
                                    (0, 0.7490, 0),
                                    (0, 0.5, 0),
                                    (1, 1, 0),
                                    (1, 0.5, 0),
                                    (1, 0, 0.7490),
                                    (1, 0, 0),
                                    (0.6509, 0.3255, 0.2353),
                                    (0.4705, 0.2353, 0.1568)]
simrad_cmap = (LinearSegmentedColormap.from_list
                             ('Simrad', simrad_color_table))
simrad_cmap.set_bad(color='grey')


# In[12]:


# Plot it using simple contour
da_sub.plot.contourf(robust = True, cmap=simrad_cmap)


# In[ ]:




