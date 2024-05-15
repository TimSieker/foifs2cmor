"""ESMValTool CMORizer for ERA-Interim data.

Tier
    Tier 3: restricted datasets (i.e., dataset which requires a registration
 to be retrieved or provided upon request to the respective contact or PI).

Source
    http://apps.ecmwf.int/datasets/data/interim-full-moda/

Last access
    20190905

Download and processing instructions
    Select "ERA Interim Fields":
        Daily: for daily values
        Invariant: for time invariant variables (like land-sea mask)
        Monthly Means of Daily Means: for monthly values
        Monthly Means of Daily Forecast Accumulation: for accumulated variables
        like precipitation or radiation fluxes
    Select "Type of level" (Surface or Pressure levels)
    Download the data on a single variable and single year basis, and save
    them as ERA-Interim_<var>_<mean>_YYYY.nc, where <var> is the ERA-Interim
    variable name and <mean> is either monthly or daily. Further download
    "land-sea mask" from the "Invariant" data and save it in
    ERA-Interim_lsm.nc.
    It is also possible to download data in an automated way, see:
        https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets
        https://confluence.ecmwf.int/display/WEBAPI/Python+ERA-interim+examples
    A registration is required for downloading the data.
    It is also possible to use the script in:
    esmvaltool/cmorizers/data/download_scripts/download_era-interim.py
    This cmorization script currently supports daily and monthly data of
the following variables:
        10m u component of wind
        10m v component of wind
        2m dewpoint temperature
        2m temperature
        evaporation
        maximum 2m temperature since previous post processing
        mean sea level pressure
        minimum 2m temperature since previous post processing
        skin temperature
        snowfall
        surface net solar radiation
        surface solar radiation downwards
        temperature of snow layer
        toa incident solar radiation
        total cloud cover
        total precipitation
and daily, monthly (not invariant) data of:
        Geopotential

and monthly data of:
        Inst. eastward turbulent surface stress
        Inst. northward turbulent surface stress
        Sea surface temperature
        Surface net thermal radiation
        Surface latent heat flux
        Surface sensible heat flux
        Relative humidity
        Temperature
        U component of wind
        V component of wind
        Vertical velocity
        Specific humidity
        net top solar radiation
        net top solar radiation clear-sky
        top net thermal radiation
        top net thermal radiation clear-sky
        fraction of cloud cover (3-dim)
        vertical integral of condensed cloud water (ice and liquid)
        vertical integral of cloud liquid water
        vertical integral of cloud frozen water
        total column water vapour
        specific cloud liquid water content
        specific cloud ice water content

Caveats
    Make sure to select the right steps for accumulated fluxes, see:
        https://confluence.ecmwf.int/pages/viewpage.action?pageId=56658233
        https://confluence.ecmwf.int/display/CKB/ERA-Interim%3A+monthly+means
    for a detailed explanation.
    The data are updated regularly: recent years are added, but also the past
    years are sometimes corrected. To have a consistent timeseries, it is
    therefore recommended to download the full timeseries and not just add
    new years to a previous version of the data.

For further details on obtaining daily values from ERA-Interim,
    see:
    https://confluence.ecmwf.int/display/CKB/ERA-Interim
    https://confluence.ecmwf.int/display/CKB/ERA-Interim+documentation#ERA-Interimdocumentation-Monthlymeans
    https://confluence.ecmwf.int/display/CKB/ERA-Interim%3A+How+to+calculate+daily+total+precipitation
"""
import logging
import os
import uuid
import re
import sys
import yaml
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timedelta
from os import cpu_count
from pathlib import Path
from warnings import catch_warnings, filterwarnings
from netCDF4 import Dataset

import iris

import numpy as np
from esmvalcore.cmor.table import CMOR_TABLES
import esmvalcore.cmor.table
from esmvalcore.config import CFG
from esmvalcore.preprocessor import daily_statistics, monthly_statistics
from iris import NameConstraint

from esmvaltool.cmorizers.data import utilities as utils

logger = logging.getLogger(__name__)
iris.config.netcdf.conventions_override = True

def _fix_units(cube, definition, var_attr):
    """Fix issues with the units."""

    original_unit = cube.units
    unit_change=False

    if cube.var_name in {'evspsbl', 'pr', 'prc', 'prsn', 'mrro', 'mrros', 'sbl', 'sf'}:
        # Change units from meters of water per day
        # to kg of water per m2 per day
        #cube.units = 'm'  # fix invalid units
        cube.units = cube.units * 'kg m-3 s-1'
        cube.data = cube.core_data() * 1000 / 21600
        unit_change=True

    if cube.var_name in {'mrso'}:
        cube.units = 'kg m-2'
        cube.data = cube.core_data() * 1000
        unit_change=True

    if cube.var_name in {'hfss', 'hfls','rlus', 'rsus', 'rsut', 'rsutcs'}:
        # Add missing 'per day'
        cube.units = cube.units * 's-1'
        cube.data = cube.core_data() * (-1) / 21600
        unit_change=True

    if cube.var_name in {'rsds', 'rsdt', 'rlds', 'rss', 'rls'}:
        # Add missing 'per day'
        cube.units = cube.units * 's-1'
        cube.data = cube.core_data() / 21600
        unit_change=True

    if cube.var_name in {'rlut', 'rlutcs'}:
        # Add missing 'per day'
        cube.units = cube.units * 'day-1'
        # Radiation fluxes are positive in upward direction
        cube.attributes['positive'] = 'up'
        cube.data = cube.core_data() * -1.
        unit_change=True

    if cube.var_name in {'tauu', 'tauv'}:
        cube.attributes['positive'] = 'down'
    if cube.var_name in {'sftlf', 'clt', 'cl', 'clt-low', 'clt-med',
                         'clt-high'}:
        # Change units from fraction to percentage
        cube.units = definition.units
        cube.data = cube.core_data() * 100.
    if cube.var_name in {'zg', 'orog'}:
        # Divide by acceleration of gravity [m s-2],
        # required for geopotential height, see:
        # https://apps.ecmwf.int/codes/grib/param-db?id=129
        cube.units = cube.units / 'm s-2'
        cube.data = cube.core_data() / 9.80665
        unit_change=True

    if cube.var_name in {'wap'}:
        #### TODO
        cube.units = cube.units * 'Pa m-1'
        unit_change=True

    if cube.var_name in {'cli', 'clw'}:
        cube.units = 'kg kg-1'
        unit_change=True

    var_attr['unit_change'] = unit_change
    if unit_change:
        var_attr['original_unit'] = str(original_unit)

    return cube, var_attr


def _fix_coordinates(cube, definition, base_time="1850-01-01"):
    """Fix coordinates."""
    # Make latitude increasing
    cube = cube[..., ::-1, :]

    # Make pressure_level decreasing
    coord_long_name = [item.long_name for item in cube.coords()]
    if 'pressure_level' in coord_long_name:
        cube = cube[:, ::-1, ...]

    # Add scalar height coordinates
    if 'height2m' in definition.dimensions:
        utils.add_scalar_height_coord(cube, 2.)
    if 'height10m' in definition.dimensions:
        utils.add_scalar_height_coord(cube, 10.)

    for coord_def in definition.coordinates.values():
        axis = coord_def.axis

        # ERA-Interim cloud parameters are downloaded on pressure levels
        # (CMOR standard = generic (hybrid) levels, alevel)
        if axis == "" and coord_def.name == "alevel":
            axis = "Z"
            coord_def = CMOR_TABLES['CMIP6'].coords['plev19']

        coord = cube.coord(axis=axis)
        if axis == 'T':
            coord.convert_units('days since %s 00:00:00.0' % base_time)
        if axis == 'Z':
            coord.convert_units(coord_def.units)
        coord.standard_name = coord_def.standard_name
        coord.var_name = coord_def.out_name
        coord.long_name = coord_def.long_name
        coord.points = coord.core_points().astype('float64')

        if axis != 'T':
            coord.attributes['missing_values'] = 1e20


        if len(coord.points) > 1 and not coord.var_name == 'plev' and not coord.has_bounds(): ### adjusted
            coord.guess_bounds()
        if coord.var_name == 'plev':
            coord.attributes['positive'] = 'down'
    return cube


def _fix_aux_coords(cube):
    aux_coords = cube._aux_coords_and_dims
    aux_coords_new = []
    for aux_coord in aux_coords:
        if aux_coord[0].standard_name == 'time':
            logger.debug("Deleting auxiliary time coordinate to avoid duplication.")

        else:
            aux_coords_new.append(aux_coord)

    cube._aux_coords_and_dims = aux_coords_new
    return cube

def _load_cube(in_files, var, var_attr):
    """Load in_files into an iris cube."""
    ignore_warnings = (
        {
            'raw': 'cc',
            'units': '(0 - 1)',
        },
        {
            'raw': 'tcc',
            'units': '(0 - 1)',
        },
        {
            'raw': 'tciw',
            'units': 'kg m**-2',
        },
        {
            'raw': 'tclw',
            'units': 'kg m**-2',
        },
        {
            'raw': 'lsm',
            'units': '(0 - 1)',
        },
        {
            'raw': 'e',
            'units': 'm of water equivalent',
        },
        {
            'raw': 'sf',
            'units': 'm of water equivalent',
        },
        {
            'raw': 'tp',
            'units': 'm of water equivalent',
        },
    )

    with catch_warnings():
        msg = "Ignoring netCDF variable '{raw}' invalid units '{units}'"
        for warning in ignore_warnings:
            filterwarnings(action='ignore',
                           message=re.escape(msg.format(**warning)),
                           category=UserWarning,
                           module='iris')

        if len(in_files) == 1:
            cube = iris.load_cube(
                in_files[0],
                constraint=NameConstraint(var_name=var['raw']),
            )
            _fix_aux_coords(cube)
        elif var.get('operator', '') == 'sum':
            # Multiple variables case using sum operation
            cube = None
            for raw_name, filename in zip(var['raw'], in_files):
                in_cube = iris.load_cube(
                    filename,
                    constraint=NameConstraint(var_name=raw_name),
                )
                _fix_aux_coords(in_cube)
                if cube is None:
                    cube = in_cube
                else:

                    cube += in_cube
        elif var.get('operator', '') == 'diff':
            # two variables case using diff operation
            cube = None
            elements_var = len(var['raw'])
            elements_files = len(in_files)
            if (elements_var != 2) or (elements_files != 2):
                shortname = var.get('short_name')
                errmsg = (f'operator diff selected for variable {shortname} '
                          f'expects exactly two input variables and two input '
                          f'files')
                raise ValueError(errmsg)
            cube = iris.load_cube(
                in_files[0],
                constraint=NameConstraint(var_name=var['raw'][0]),
            )
            _fix_aux_coords(cube)
            cube2 = iris.load_cube(
                in_files[1],
                constraint=NameConstraint(var_name=var['raw'][1])
            )
            _fix_aux_coords(cube2)
            cube -= cube2
        else:
            raise ValueError(
                "Multiple input files found, with operator '{}' configured: {}"
                .format(var.get('operator'), ', '.join(in_files)))

    return cube, var_attr

def _get_table_header(table_id, mip_era):
    json_file = mip_era + '_' + table_id + '.json'
    json_filepath = esmvalcore.cmor.table.__file__
    json_filepath = os.path.dirname(json_filepath)
    json_filepath = os.path.join(json_filepath, 'tables')
    json_filepath = os.path.join(json_filepath, mip_era.lower())
    json_filepath = os.path.join(json_filepath, 'Tables')
    json_filepath = os.path.join(json_filepath, json_file)

    with open(json_filepath, encoding='utf-8') as inf:
            raw_data = json.loads(inf.read())
            header = raw_data['Header']
    return header

def _get_variable_attrs(table_id, mip_era, var_name, var_attr):
    ### looks for the meta data of the CMIP table
    ### also adds the the local comment attribute of the variable, since esmvaltool does not allow do read it otherwise
    json_file = mip_era + '_' + table_id + '.json'
    json_filepath = esmvalcore.cmor.table.__file__
    json_filepath = os.path.dirname(json_filepath)
    json_filepath = os.path.join(json_filepath, 'tables')
    json_filepath = os.path.join(json_filepath, mip_era.lower())
    json_filepath = os.path.join(json_filepath, 'Tables')
    json_filepath = os.path.join(json_filepath, json_file)

    with open(json_filepath, encoding='utf-8') as inf:
            raw_data = json.loads(inf.read())

            var_attr['comment'] = raw_data['variable_entry'][var_name]['comment']
            var_attr['cell_measures'] = raw_data['variable_entry'][var_name]['cell_measures']
    return var_attr

def _fix_global_metadata(cube, var, attrs, var_attr):
    """Complete the cmorized file with global metadata."""
    logger.debug("Setting global metadata...")
    attrs = dict(attrs)
    cube.attributes.clear()
    attrs.pop('project_id')
    attrs.pop('base_time')

    ###attr: creation_date
    timestamp = datetime.utcnow()
    timestamp_format = "%Y-%m-%dT%H:%M:%SZ"
    now_time = timestamp.strftime(timestamp_format)

    var_attr['timestamp'] = now_time

    ###attr: tracking_id
    x = uuid.uuid4()
    tracking_id = 'hdl:21.14100/' + str(x)

    ###attr: variant_id
    variant_label_indices = [
    'r', str(attrs['realization_index']),
    'i', str(attrs['initialization_index']),
    'p', str(attrs['physics_index']),
    'f', str(attrs['forcing_index'])]

    variant_label = ''.join(variant_label_indices)

    ###attr: realization_index
    attrs['realization_index'] = np.int32(attrs['realization_index'])

    ###attr: initialization_index
    attrs['initialization_index'] = np.int32(attrs['initialization_index'])

    ###attr: physics_index
    attrs['physics_index'] = np.int32(attrs['physics_index'])

    ###attr: forcing_index
    attrs['forcing_index'] = np.int32(attrs['forcing_index'])

    variant_label = ''.join(variant_label_indices)

    ###attr: member_id
    if attrs['sub_experiment_id'] == 'none':
        member_id = variant_label
    else:
        member_id = attrs['sub_experiment_id'] + '-' + variant_label

    ###attr:
    header = _get_table_header(var['table_id'], attrs['mip_era'])

    ###attr:
    if attrs['branch_method'] == 'no parent':
        attrs.pop('branch_method')
        attrs.pop('branch_time_in_child')
        attrs.pop('branch_time_in_parent')

    else:
        attrs['branch_time_in_child'] = np.float64(attrs['branch_time_in_child'])
        attrs['branch_time_in_parent'] = np.float64(attrs['branch_time_in_parent'])

    if attrs['parent_experiment_id'] == 'no parent':
        attrs.pop('parent_experiment_id')
        attrs.pop('parent_activity_id')
        attrs.pop('parent_mip_era')
        attrs.pop('parent_source_id')
        attrs.pop('parent_time_units')
        attrs.pop('parent_variant_label')


    attrs['table_id'] = var['table_id']
    attrs['variable_id'] = var['short_name']
    attrs['variant_label'] = variant_label
    attrs['member_id'] = member_id
    attrs['creation_date'] = now_time
    attrs['tracking_id'] = tracking_id
    attrs['data_specs_version'] = header['data_specs_version']
    attrs['cmor_version'] = np.float64(header['cmor_version'])
    attrs['Conventions'] = header['Conventions']

    cube.attributes = attrs

    return cube, var_attr


def _fix_variable(cube, definition, var_attr):
    # Set correct names
    original_name = cube.var_name
    if cube.var_name != definition.short_name:
        var_name_change = True

        cube.var_name = definition.short_name
        if definition.standard_name:
            cube.standard_name = definition.standard_name
        cube.long_name = definition.long_name
    else:
        var_name_change = False


    if original_name is None:
        var_name_change = False

    var_attr['var_name_change'] = var_name_change
    if var_name_change:
        var_attr['original_name'] = original_name


    #NOTE: Make sure dtype is float32
    #cube.data = cube.core_data().astype('float32')

    return cube, var_attr

def _extract_variable(in_files, var, cfg, out_struc):
    logger.info("CMORizing variable '%s' from input files '%s'",
                var['short_name'], ', '.join(in_files))
    attributes = deepcopy(cfg['attributes'])
    var_attr = {} #stores attributes specific to the variable

    cmor_table = CMOR_TABLES[attributes['mip_era']]
    definition = cmor_table.get_variable(var['table_id'], var['short_name'])


    cube, var_attr = _load_cube(in_files, var, var_attr)

    cube, var_attr = _fix_global_metadata(cube, var, attributes, var_attr)

    cube, var_attr = _fix_variable(cube, definition, var_attr)

    cube, var_attr = _fix_units(cube, definition, var_attr)

    cube = _fix_coordinates(cube, definition, base_time=attributes['base_time'])

    var_attr = _get_variable_attrs(var['table_id'], attributes['mip_era'], var['short_name'], var_attr)

    logger.debug("Saving cube\n%s", cube)
    logger.debug("Expected output size is %.1fGB",
                 np.prod(cube.shape) * 4 / 2**30)
    _save_variable(
        cube,
        cube.var_name,
        out_struc,
        var_attr,
        local_keys=['positive'],
    )
    logger.info("Finished CMORizing %s", ', '.join(in_files))


def _get_in_files_by_year(in_dir, var):
    """Find input files by year."""
    if 'file' in var:
        var['files'] = [var.pop('file')]

    in_files = defaultdict(list)
    for pattern in var['files']:
        for filename in Path(in_dir).glob(pattern):
            year = str(filename.stem).rsplit('_', maxsplit=1)[-1]
            in_files[year].append(str(filename))

    # Check if files are complete
    for year in in_files.copy():
        if len(in_files[year]) != len(var['files']):
            logger.warning(
                "Skipping CMORizing %s for year '%s', %s input files needed, "
                "but found only %s", var['short_name'], year,
                len(var['files']), ', '.join(in_files[year]))
            in_files.pop(year)

    return in_files.values()



def _save_variable(cube, var, out_struc, var_attr, **kwargs):
    """Saver function. Adapted to CMIP6 structure.

    Saves iris cubes (data variables) in CMOR-standard named files.

    Parameters
    ----------
    cube: iris.cube.Cube
        data cube to be saved.

    var: str
        Variable short_name e.g. ts or tas.

    outdir: str
        root directory where the file will be saved.

    attrs: dict
        dictionary holding cube metadata attributes like
        project_id, version etc.

    **kwargs: kwargs
        Keyword arguments to be passed to `iris.save`
    """

    out_filepath = out_struc['output_dir']

    if out_struc['custom_output_structure']:
        rs = ''
        attr = ''
        in_brackets=False

        for sstr in out_struc['output_dir_attr_list']:
            for t in sstr:
                if t == '{':
                    in_brackets=True
                    continue

                if t == '}':
                    in_brackets=False
                    rs = rs + str(cube.attributes[attr])
                    attr=''
                    continue
                if in_brackets:
                    attr = attr + t
                else:
                    rs = rs + t
            out_filepath = os.path.join(out_filepath, rs)
            rs=''

        if not os.path.exists(out_filepath):
            os.makedirs(out_filepath)

        rs = ''
        attr = ''
        in_brackets=False
        name_elements = []
        for sstr in out_struc['output_filename_attr_list']:
            for t in sstr:
                if t == '{':
                    in_brackets=True
                    continue

                if t == '}':
                    in_brackets=False
                    rs = rs + str(cube.attributes[attr])
                    attr=''
                    continue
                if in_brackets:
                    attr = attr + t
                else:
                    rs = rs + t
            name_elements.append(rs)
            rs=''
        out_filename = '_'.join(name_elements)
        out_filepath = os.path.join(out_filepath, out_filename)


    else:

        out_filepath = out_dir

        name_elements = [
            cfg['attributes']['mip_era'],
            cfg['attributes']['source_id'],
            cfg['attributes']['realm'],
            cfg['attributes']['version'],
            cfg['attributes']['table_id'],
            var,
        ]

        out_filename = '_'.join(name_elements)
        out_filepath = os.path.join(out_filepath, out_filename)

    # CMOR standard
    try:
        time = cube.coord('time')
    except iris.exceptions.CoordinateNotFoundError:
        time_suffix = None
    else:
        if len(time.points) == 1 and "mon" not in cube.attributes.get('mip'):
            year = str(time.cell(0).point.year)
            time_suffix = '-'.join([year + '01', year + '12'])
        else:
            date1 = (
                f"{time.cell(0).point.year:d}{time.cell(0).point.month:02d}"
            )
            date2 = (
                f"{time.cell(-1).point.year:d}{time.cell(-1).point.month:02d}"
            )
            time_suffix = '-'.join([date1, date2])

    out_filepath = out_filepath + '_' + time_suffix + '.nc'
    logger.info('Saving: %s', out_filepath)
    status = 'lazy' if cube.has_lazy_data() else 'realized'
    logger.info('Cube has %s data [lazy is preferred]', status)
    iris.save(cube, out_filepath, fill_value=1e20)

    ### run fixer, for all things that iris can't handle
    _fixer_script(out_filepath, var, var_attr)


    print('Finished cmorizing variable ' + var)


def _fixer_script(filepath, var_name, var_attr):
    ###functionality of iris doesn't allow to customize boundary variables, i.e. vertices have to be fixed
    with Dataset(filepath, 'r+') as ds:
        ## add local attributes
        ds[var_name].comment =  var_attr['comment']
        ds[var_name].cell_measures =  var_attr['cell_measures']

        var_change = False
        timestamp = var_attr['timestamp']
        history = timestamp + ' altered by CMOR: '

        if var_attr['var_name_change']:
            ds[var_name].original_name = var_attr['original_name']
            history = history + 'Renamed Variable. '
            var_change = True

        if var_attr['unit_change']:
            ds[var_name].original_unit = var_attr['original_unit']
            history = history + 'Unit converted. '
            var_change = True

        if var_change:
            ds[var_name].history = history


def _run(jobs, n_workers):
    """Run CMORization jobs using n_workers."""
    if n_workers == 1:
        for job in jobs:
            _extract_variable(*job)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for job in jobs:
                future = executor.submit(_extract_variable, *job)
                futures[future] = job[0]

            for future in as_completed(futures):
                try:
                    future.result()
                except:
                    logger.error("Failed to CMORize %s",
                                 ', '.join(futures[future]))
                    raise

def _read_config_file(config_file):
    """Read config user file and store settings in a dictionary."""
    config_file = Path(config_file)
    if not config_file.exists():
        raise IOError(f'Config file `{config_file}` does not exist.')

    with open(config_file, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    return cfg



def cmorization(cfg_user_filepath):
    """Run CMORizer for FOCI_OCEAN."""


    cfg_user = _read_config_file(cfg_user_filepath)

    in_dir = cfg_user['input_dir']

    cfg = _read_config_file(cfg_user['cmor_config'])

    cfg['attributes']['comment'] = cfg['attributes']['comment'].strip().format(
        year=datetime.now().year)

    project_id = cfg['attributes']['mip_era']
    #cfg_dev_path = cfg_user['config_developer_file']
    #cfg_dev = _read_config_file(cfg_dev_path)

    ### is a custom file structure given?
    out_struc = dict()
    try:
        output_dir_attr_list = cfg['output']['output_dir_structure'].split('/')
        output_filename_attr_list = cfg['output']['output_filename_structure'].split('/')
        out_struc['custom_output_structure'] = True
        out_struc['output_dir'] = cfg_user['output_dir']
        out_struc['output_dir_attr_list'] = output_dir_attr_list
        out_struc['output_filename_attr_list'] = output_filename_attr_list
    except:
        out_struc['custom_output_structure'] = False
        out_struc['output_dir'] = cfg_user['output_dir']
        out_struc['output_dir_attr_list'] = []
        out_struc['output_filename_attr_list'] = []

    n_workers = cfg_user.get('max_parallel_tasks')
    if n_workers is None:
        n_workers = int(cpu_count() / 1.5)
    logger.info("Using at most %s workers", n_workers)

    jobs = []


    for short_name, var in cfg['variables'].items():
        if 'short_name' not in var:
            var['short_name'] = short_name

        if 'vertices' in var:
            aux_dir = cfg_user['auxiliary_data_dir']
            gridvertices_filepath = os.path.join(aux_dir, var['vertices']['file'])
            var['vertices']['filepath'] = gridvertices_filepath

        for in_files in _get_in_files_by_year(in_dir, var):
            jobs.append([in_files, var, cfg, out_struc])

    _run(jobs, n_workers)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Incorrect number of arguments')
    cfg_user_filepath = sys.argv[1]

    cmorization(cfg_user_filepath)
