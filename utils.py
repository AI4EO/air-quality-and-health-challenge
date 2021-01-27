import os
import zipfile
import itertools
from datetime import datetime
from matplotlib import pyplot as plt
from typing import List, Union, Tuple, Optional

import numpy as np
from matplotlib import patches, patheffects

import fiona
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon, MultiPolygon

from eolearn.core import EOPatch, FeatureType
from eolearn.io import ImportFromTiff

from logging import Logger
logger = Logger(__file__)


def get_extent(eopatch: EOPatch) -> Tuple[float, float, float, float]:
    """
    Calculate the extent (bounds) of the patch.

    Parameters
    ----------
    eopatch: EOPatch for which the extent is calculated.

    Returns The list of EOPatch bounds (min_x, max_x, min_y, max_y)
    -------
    """
    return eopatch.bbox.min_x, eopatch.bbox.max_x, eopatch.bbox.min_y, eopatch.bbox.max_y


def draw_outline(o, lw, foreground='black'):
    """
    Adds outline to the matplotlib patch.

    Parameters
    ----------
    o:
    lw: Linewidth
    foreground

    Returns
    -------
    """
    o.set_path_effects([patheffects.Stroke(linewidth=lw, foreground=foreground), patheffects.Normal()])


def draw_poly(ax, poly: Union[Polygon, MultiPolygon], color: str = 'r', lw: int = 2, outline: bool = True):
    """
    Draws a polygon or multipolygon onto an axes.

    Parameters
    ----------
    ax: Matplotlib Axes on which to plot on
    poly: Polygon or Multipolygons to plot
    color: Color of the plotted polygon
    lw: Line width of the plot
    outline: Should the polygon be outlined

    Returns None
    -------

    """
    if isinstance(poly, MultiPolygon):
        polys = list(poly)
    else:
        polys = [poly]
    for poly in polys:
        if poly is None:
            logger.warning("One of the polygons is None.")
            break
        if poly.exterior is None:
            logger.warning("One of the polygons has not exterior.")
            break

        x, y = poly.exterior.coords.xy
        xy = np.moveaxis(np.array([x, y]), 0, -1)
        patch = ax.add_patch(patches.Polygon(xy, closed=True, edgecolor=color, fill=False, lw=lw))

    if outline:
        draw_outline(patch, 4)


def draw_bbox(ax, eopatch: EOPatch, color: str = 'r', lw: int = 2, outline: bool = True):
    """
    Plots an EOPatch bounding box onto a matplotlib axes.
    Parameters
    ----------
    ax: Matplotlib axes on which to plot.
    eopatch: EOPatch with BBOx
    color: Color of the BBOX plot.
    lw: Line width.
    outline: Should the plot be additionally outlined.

    Returns None
    -------

    """
    bbox_poly = eopatch.bbox.get_polygon()
    draw_poly(ax, Polygon(bbox_poly), color=color, lw=lw, outline=outline)


def draw_feature(ax, eopatch: EOPatch, time_idx: Union[List[int], int, None], feature: tuple, grid: bool = True, band: int = None, interpolation: str = 'none',
              vmin: int = 0, vmax: int = 1, alpha: float = 1.0, cmap=plt.cm.viridis):
    """
    Draws an EOPatch feature.
    Parameters
    ----------
    ax: Matplotlib axes on which to plot on
    eopatch: EOPatch for which to plot the mask:
    time_idx: Time index of the mask. If int, single time index of the mask feature, if List[int] multiple masks for
    each time index. If None, plot mask_timeless.
    feature: Tuple defining feature to plot, e.g. (FeatureType.DATA, 'DATA').
    grid: Show grid on plot:
    interpolation: Interpolation used by imshow
    vmin: Minimum value (for mask visualization)
    vmax: Maximum value (for mask visualization)
    alpha: Transparency of the mask
    Returns
    -------

    """

    def _show_single_ts(axis, img, ts):
        fh = axis.imshow(img, extent=get_extent(eopatch), vmin=vmin, vmax=vmax, alpha=alpha, cmap=cmap, interpolation=interpolation)
        if grid:
            axis.grid()
        title = f'{feature[1]} {eopatch.timestamp[ts]}' if ts is not None else f'{feature[1]}'
        axis.set_title(title)
        return fh

    if time_idx is None:
        image = eopatch[feature][..., band] if band is not None else eopatch[feature].squeeze() 
        return _show_single_ts(ax, image, time_idx)
    elif isinstance(time_idx, int):
        image = eopatch[feature][time_idx][..., band] if band is not None else eopatch[feature][time_idx].squeeze()
        return _show_single_ts(ax, image, time_idx)
    elif isinstance(time_idx, list):
        for i, tidx in enumerate(time_idx):
            image = eopatch[feature][tidx][..., band] if band is not None else eopatch[feature][tidx].squeeze()
            fh = _show_single_ts(ax[i], image, tidx)
        return fh
    
    
def draw_true_color(ax: plt.axes, eopatch: EOPatch, time_idx: Union[List[int], int],
                    feature_name='BANDS-S2-L2A',
                    bands: Tuple[int] = (3, 2, 1),
                    factor: int = 3.5,
                    grid: bool = True):
    """
    Visualization of the bands in the EOPatch.
    Parameters
    ----------
    ax: Axis on which to plot
    eopatch: EOPatch to visualize.
    time_idx: Single timestamp or multiple timestamps.
    feature_name: Name of the feature to visualize.
    bands: Order of the bands.
    factor: Rescaling factor to
    grid: Show grid on visualization

    Returns None
    -------

    """
    def visualize_single_idx(axis, ts):
        axis.imshow(np.clip(eopatch.data[feature_name][ts][..., bands] * factor, 0, 1), extent=get_extent(eopatch))
        if grid:
            axis.grid()
            axis.set_title(f'{feature_name} {eopatch.timestamp[ts]}')

    if isinstance(time_idx, int):
        time_idx = [time_idx]
    if len(time_idx) == 1:
        visualize_single_idx(ax, time_idx[0])
    else:
        for i, tidx in enumerate(time_idx):
            visualize_single_idx(ax[i], tidx)

            
def ungz_file(gzfilename, filename, delete=False):
    """ Helper function to unzip .gz files and optionally delete them after decompression
    """
    if not os.path.exists(filename):
        with gzip.open(gzfilename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    if delete:
        os.remove(gzfilename)

        
def unzip_file(zipfilename, extract_to_dir):
    """ Helper function to unzip .zip files
    """
    with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
        
        
def days_to_datetimes(days: Union[int, List[int]], offset: int = 2100) -> Union[datetime, List[datetime]]:
    """ Converts Julian days to datetimes given an offset 
    """
    days = [days] if isinstance(days, int) else days
    return [datetime.strptime(f'{day-offset}', '%j') for day in days]
        
    
def datetimes_to_days(timestamps: Union[datetime, List[datetime]], offset: int = 2100) -> Union[float, List[float]]:
    """ Converts datetimes objects to julian days given an offset 
    """
    timestamps = [timestamps] if isinstance(timestamps, datetime) else timestamps
    return [timestamp.timetuple().tm_yday+offset+timestamp.hour/24.0 for timestamp in timestamps]
    
    
def _parse_s3_timestamps(tiles: List[str], offset: int = 2100) -> List[datetime]:
    timestamps = []
    for tile in sorted(tiles):
        day, hour = tile.split('_SYN_')[-1].split('_')
        day = int(day)-offset
        timestamps.append(datetime.strptime(f'{day}{hour[:3]}', '%jT%H'))
    return timestamps
        
    
def _parse_s5p_timestamps(tiles: List[str], offset: int = 2100) -> List[datetime]:
    timestamps = []
    for tile in sorted(tiles):
        day, hour = tile.split('_day')[-1].split('_')
        day = int(day)-offset
        timestamps.append(datetime.strptime(f'{day}{hour[:3]}', '%jT%H'))
    return timestamps


def _parse_modis_timestamps(tiles: List[str], offset: int = 2100) -> List[datetime]:
    timestamps = []
    for tile in sorted(tiles):
        day = tile.split('_day')[-1]
        day = day.split('_')[0].split('.')[0]
        day = int(day)-offset
        timestamps.append(datetime.strptime(f'{day}', '%j'))
    return timestamps


def _parse_era5_timestamps(tiles: List[str], offset: int = 2100) -> List[datetime]:
    timestamps = []
    for tile in sorted(tiles):
        day, hour = tile.split('_day')[-1].split('_')
        day = int(day)-offset
        timestamps.append(datetime.strptime(f'{day}{hour[:3]}', '%jh%H'))
    return timestamps


TIMESTAMP_PARSER = dict(s5p=_parse_s5p_timestamps,
                        modis=_parse_modis_timestamps,
                        era5=_parse_era5_timestamps,
                        cams=_parse_era5_timestamps,
                        s3=_parse_s3_timestamps)


def load_tiffs(datapath: str, 
               feature: Tuple[FeatureType, str], 
               filename: str = None,
               image_dtype: np.generic = np.float32, 
               no_data_value: float = np.nan, 
               data_source: str = 's5p', 
               offset: int = 2100):
    """ Helper function to load the data sources provided as tiffs """
    assert data_source in ['s5p', 'modis', 'era5', 'cams', 's3']
    
    tiles = sorted(os.listdir(datapath)) if filename is None else [filename]

    # only keep files
    tiles = [tile for tile in tiles if not os.path.isdir(datapath/tile)]

    # unzip tiffs if they have .gz extension and delete them
    compressed_tiles = [tile for tile in tiles if tile.endswith('.gz')]
    for ctile in compressed_tiles:
        uctile = ctile.split('.gz')[0]
        ungz_file(str(datapath/ctile), str(datapath/uctile), delete=True)
    
    tiles = [tile if not tile.endswith('.gz') else tile.split('.gz')[0]
             for tile in tiles]
    
    zipped_tiles = [tile for tile in tiles if tile.endswith('.zip')]
    for ztile in zipped_tiles:
        unzip_file(str(datapath/ztile), str(datapath))
        
    tiles = tiles + [tile.replace('.zip', '.tif') for tile in zipped_tiles]
    
    # remove files which don't have .tif extension
    tiles = [tile for tile in tiles if os.path.splitext(tile)[1] == '.tif']

    timestamp_size = len(tiles) if feature[0].is_time_dependent() else None
    
    import_task = ImportFromTiff(feature=feature, 
                                 folder=datapath,
                                 image_dtype=image_dtype,
                                 no_data_value=no_data_value,
                                 timestamp_size=len(tiles))
                                 
    if not feature[0].is_time_dependent():
        assert len(tiles) == 1
        return import_task.execute(filename=tiles[0])
    
    assert len(tiles) >= 1
    eop = import_task.execute(filename=(sorted(tiles)))
    eop.timestamp = TIMESTAMP_PARSER[data_source](sorted(tiles), offset)

    return eop


def reproject_tiff(src_tifname: str, dst_tifname: str, dst_crs: str):
    """ Reproject a tif file to WGS84. 
    
    Taken from https://rasterio.readthedocs.io/en/latest/topics/reproject.html
    """
    with rasterio.open(src_tifname) as src:
        print(*src.bounds)
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_tifname, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)
    
# #### Upsample S5P NO2 products to given resolution using bilinear interpolation

# The following code based on `rasterio` shows how to:

#  * reproject a S5P NO2 product tile to the target CRS;
#  * crop the tile to the target bounding box;
#  * upsample the tile to the target `500x500` dimension.
 
# This consistutes a mock upsmapling pipeline, and a very basic baseline.

def upscale_tiff(input_filename: str, output_filename: str, out_shape: Tuple[int, int]):
    """ Upscale a tiff file given a target shape. Writes out first channel only """
    
    with rasterio.open(input_filename) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                out_shape[0],
                out_shape[1]
            ),
            resampling=Resampling.bilinear)

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        
        out_meta = dataset.meta
        
    out_meta.update({"driver": "GTiff",
                     "height": data.shape[1],
                     "width": data.shape[2],
                     "transform": transform,
                     "count": 1})

    with rasterio.open(output_filename, 'w', **out_meta) as dest:
        dest.write(data[:1])

    
def mask_tiff(geo_filename: str, input_filename: str, output_filename: str):
    """ Mask a tiff file given a polygon geometry
    
    Taken from https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html?highlight=crop#masking-a-raster-using-a-shapefile
    """
    with fiona.open(geo_filename, 'r') as shapefile:
        shapes = [feature['geometry'] for feature in shapefile]
        
    with rasterio.open(input_filename) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, 
                                                      crop=True, 
                                                      nodata=np.nan)
        out_meta = src.meta
        
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(output_filename, 'w', **out_meta) as dest:
        dest.write(out_image)