#!/usr/bin/env python
# coding: utf-8

# In[1]:


from echolab2.instruments import EK80, EK60
from echolab2.plotting.matplotlib import echogram

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, ticker
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, Colormap
from matplotlib.pyplot import figure, show, subplots_adjust, get_cmap

import zarr
import os.path
import shutil
from scipy.signal import chirp, sweep_poly


# In[2]:


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


# In[3]:


# Simple plot function
def plot_sv(da):
    # Use 0.5 bins
    x_bins = np.arange(da.range.min().round(), da.range.max().round(), 0.5)
    # Plot using lower resolution
    da_1 = da.groupby_bins("range", x_bins).mean()
    da_1.plot( norm=colors.LogNorm(vmin=da_1.min(), vmax=da_1.max()), cmap=simrad_cmap)


# In[4]:


# Detect FileType
def ek_detect(fname):
    with open(fname, 'rb') as f:
        file_header = f.read(8)
        file_magic = file_header[-4:]
        if file_magic.startswith(b'XML'):
            return "EK80"
        elif file_magic.startswith(b'CON'):
            return "EK60"
        else:
            return None


# In[5]:


def ek_read(fname):
    if ek_detect(fname) == "EK80":
        ek80_obj = EK80.EK80()
        ek80_obj.read_raw(fname)
        return ek80_obj
    elif ek_detect(fname) == "EK60":
        ek60_obj = EK60.EK60()
        ek60_obj.read_raw(fname)
        return ek60_obj


# In[6]:


# File collections
dir = os.path.expanduser("~/IMR/echo-test/hackathon/")

files = ['CRIMAC_2020_EK80_CW_DemoFile_GOSars.raw',
'CRIMAC_2020_EK80_FM_DemoFile_GOSars.raw',
'2017843-D20170426-T115044.raw']

# Select a file
raw_fname = dir + files[0]
#raw_fname = dir + files[1]
#raw_fname = dir + files[2]

# Read
raw_obj = ek_read(raw_fname)


# In[7]:


print(raw_obj)


# In[8]:


# Switch for aiding the data storage
da_set = False

for chan in list(raw_obj.raw_data.keys()):
    # Getting raw data for a frequency
    raw_data = raw_obj.raw_data[chan][0]
    # Get calibration object
    cal_obj = raw_data.get_calibration()
    # Get sv values
    sv_obj = raw_data.get_sv(calibration = cal_obj)
    # Get sv as depth
    sv_obj_as_depth = raw_data.get_sv(calibration = cal_obj,
        return_depth=True)

    # Get frequency label
    freq = sv_obj.frequency

    # Expand sv values into a 3d object
    data3d = np.expand_dims(sv_obj.data, axis=0)

    # This is the sv data in 3d    
    sv = xr.DataArray(name="sv", data=data3d, dims=['frequency', 'ping_time', 'range'],
                           coords={ 'frequency': [freq],
                                    'ping_time': sv_obj.ping_time,
                                    'range': sv_obj.range,
                                   })
    # This is the depth data
    depth = xr.DataArray(name="depth", data=np.expand_dims(sv_obj_as_depth.depth, axis=0), dims=['frequency', 'range'],
                           coords={ 'frequency': [freq],
                                    'range': sv_obj.range,
                                   })

    # Create mask (for rapid nan filter later)
    ping_mask = xr.DataArray(name="ping_mask", data=True, dims=['frequency', 'ping_time'],
                           coords={ 'frequency': [freq],
                                    'ping_time': sv_obj.ping_time,
                                   })
    range_mask = xr.DataArray(name="range_mask", data=True, dims=['frequency', 'range'],
                           coords={ 'frequency': [freq],
                                    'range': sv_obj.range,
                                   })

    # Combine frequencies if not the first loop iteraton
    if da_set:
        da_sv = xr.concat([da_sv, sv], dim='frequency')
        da_depth = xr.concat([da_depth, depth], dim='frequency')
        da_ping_mask = xr.concat([da_ping_mask, ping_mask], dim='frequency')
        da_range_mask = xr.concat([da_range_mask, range_mask], dim='frequency')
    else:        
        da_sv = sv
        da_depth = depth
        da_ping_mask = ping_mask
        da_range_mask = range_mask
        da_set = True

# Getting motion data
obj_heave = raw_obj.motion_data.heave
obj_pitch = raw_obj.motion_data.pitch
obj_roll = raw_obj.motion_data.roll
obj_heading = raw_obj.motion_data.heading


# In[9]:


# Convert the masks as boolean
da_ping_mask = da_ping_mask.fillna(0).astype(bool)
da_range_mask = da_range_mask.fillna(0).astype(bool)


# In[10]:


# The resulting 3d array of sv
da_sv


# In[11]:


# Getting a frequency (note the NaNs)
da_sv.sel(frequency = "1.8e+04")


# In[12]:


plot_sv(da_sv.sel(frequency = "1.8e+04"))


# In[13]:


# Filtering one frequency back to the raw sv
da_sub_ping_mask = da_ping_mask.sel(frequency = "1.8e+04")
da_sub_range_mask = da_range_mask.sel(frequency = "1.8e+04")
da_sub_sv = da_sv.sel(frequency = "1.8e+04")[da_sub_ping_mask, da_sub_range_mask]
da_sub_sv


# In[14]:


plot_sv(da_sub_sv)


# In[15]:


# Try to interpolate NaNs
# Select the NaN coordinates
target = ~(da_sub_ping_mask & da_sub_range_mask)
nan_loc = da_sv.sel(frequency = "1.8e+04").where(target, drop = True)

# Interpolate based on the da_sub_sv (if there is any NaNs)
if nan_loc.size > 0:
    interpolated = da_sub_sv.interp(ping_time = nan_loc.ping_time, range = nan_loc.range)
    da_sub_sv_interpolated = da_sub_sv.combine_first(interpolated)
else:
    da_sub_sv_interpolated = da_sub_sv

da_sub_sv_interpolated


# In[16]:


plot_sv(da_sub_sv_interpolated)


# In[17]:


# Check to ensure the non-nan values (aka. all the values from raw file) are not replaced
da_sub_sv_interpolated[da_sub_ping_mask, da_sub_range_mask].identical(da_sub_sv)


# In[18]:


# Crate a dataset
ds = xr.Dataset(
    data_vars=dict(
        sv=(["frequency", "ping_time", "range"], da_sv),
        depth = (["frequency", "range"], da_depth),
        ping_mask=(["frequency", "ping_time"], da_ping_mask),
        range_mask=(["frequency", "range"], da_range_mask),
        heave=(["ping_time"], obj_heave),
        pitch=(["ping_time"], obj_pitch),
        roll=(["ping_time"], obj_roll),
        heading=(["ping_time"], obj_heading),
        ),
    coords=dict(
        frequency = da_sv.frequency,
        ping_time = da_sv.ping_time,
        range = da_sv.range,
        ),
    attrs=dict(description="Multi-frequency sv values from EK."),
)


# In[19]:


ds


# In[20]:


# Delete existing output files
if os.path.isfile(raw_fname + ".nc"):
    os.remove(raw_fname + ".nc")
if os.path.isdir(raw_fname + ".zarr"):
    shutil.rmtree(raw_fname + ".zarr")


# In[21]:


# Save into netcdf
comp = dict(zlib=True, complevel=5)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(raw_fname + ".nc", encoding=encoding)


# In[22]:


# Save into zarr
compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=2)
encoding = {var: {"compressor" : compressor} for var in ds.data_vars}
ds.to_zarr(raw_fname + ".zarr", encoding=encoding)


# In[23]:


# Try to open them
ds_nc = xr.open_dataset(raw_fname + ".nc")
ds_zr = xr.open_zarr(raw_fname + ".zarr")


# In[24]:


# Open NetCDF
ds_nc


# In[25]:


# Open Zarr
ds_zr

