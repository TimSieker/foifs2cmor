
import logging
import os
import uuid
import re
import yaml
import json
import sys

from utils import _read_config_file, _get_in_files_by_year, _extract_from_curly_brackets, gen_dict_extract


from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timedelta
from os import cpu_count
from pathlib import Path
from warnings import catch_warnings, filterwarnings
from cf_units import Unit
import iris
import xarray as xr
from netCDF4 import Dataset
import numpy as np
from esmvalcore.cmor.table import CMOR_TABLES
import esmvalcore.cmor.table
from esmvalcore.config import CFG
from esmvalcore.preprocessor import daily_statistics, monthly_statistics
import dask

from iris import NameConstraint
from esmvaltool.cmorizers.data import utilities as utils

logger = logging.getLogger(__name__)
iris.config.netcdf.conventions_override = True
dask.config.set({'array.chunk-size': '250MiB'})

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

def _get_cell_vertices(vertices_config):
    filepath = vertices_config['filepath']
    lon_var_name = vertices_config['lon']
    lat_var_name = vertices_config['lat']

    vertices_cube_lon = iris.load_cube(
        filepath,
        constraint=NameConstraint(var_name=lon_var_name),
    )

    vertices_cube_lat = iris.load_cube(
        filepath,
        constraint=NameConstraint(var_name=lat_var_name),
    )
    vertices_lon = vertices_cube_lon.core_data().compute()
    vertices_lat = vertices_cube_lon.core_data().compute()
    vertices_lon = np.transpose(vertices_lon, (1,2,0))
    vertices_lat = np.transpose(vertices_lat, (1,2,0))
    return vertices_lon, vertices_lat

def _fix_coordinates(cube, var, definition, realm, base_time="1850-01-01"):
    """Fix coordinates."""
    # Make latitude increasing
    cube = cube[..., ::-1, :]

    if realm == 'ocean':
        if 'vertices' in var.keys():
            add_vertices = True
            vertices_lon, vertices_lat = _get_cell_vertices(var['vertices'])
        else:
            add_vertices = False

        for coord_def in definition.coordinates.values():
            axis = coord_def.axis

            if coord_def.name in ["typesi","iceband"]:
                continue


            if axis == "" and coord_def.name == "olevel":
                axis = "Z"

            coord = cube.coord(axis=axis)

            if axis == 'T':
                coord.convert_units('days since %s 00:00:00.0' % base_time)
                #coord.units = Unit(coord.units.origin, calendar='proleptic_gregorian')

                coord.attributes.pop('time_origin')
                coord.axis= 'T'

                if not coord.has_bounds(): ### adjusted
                    coord.guess_bounds()

            if axis == 'X' and add_vertices:
                coord.bounds = vertices_lon
                coord.units = 'degrees_east'
                coord.attributes['missing_values'] = 1e20


            if axis == 'Y' and add_vertices:
                coord.bounds = vertices_lat
                coord.units = 'degrees_north'
                coord.attributes['missing_values'] = 1e20

            if axis == 'Z':
                coord.attributes['missing_values'] = 1e20

            coord.standard_name = coord_def.standard_name

            coord.var_name = coord_def.out_name
            coord.long_name = coord_def.long_name
            coord.points = coord.core_points().astype('float64')


        ### add cell indices

        ndim = len(cube.shape)
        ipos = ndim -1
        jpos = ndim -2

        i_index = iris.coords.DimCoord(np.arange(cube.shape[ipos]).astype(np.int32), var_name='i', long_name='cell index along first dimension', units=Unit(1))
        j_index = iris.coords.DimCoord(np.arange(cube.shape[jpos]).astype(np.int32), var_name='j', long_name='cell index along second dimension', units=Unit(1))

        cube.add_dim_coord(i_index, ipos)
        cube.add_dim_coord(j_index, jpos)


    elif realm == 'atmos':
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

    if type(var['raw']) == str:
        cube = iris.load_cube(
            in_files[0],
            constraint=NameConstraint(var_name=var['raw']),
        )
        _fix_aux_coords(cube)

    elif var.get('operator', '') == 'sum':
        # Multiple variables case using sum operation
        cube = None
        for raw_name in var['raw']:
            in_cube = iris.load_cube(
                in_files[0],
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

        cube = iris.load_cube(
            in_files[0],
            constraint=NameConstraint(var_name=var['raw'][0]),
        )
        _fix_aux_coords(cube)

        cube2 = iris.load_cube(
            in_files[0],
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

def _fix_global_metadata(cube, var, attrs, experiment, var_attr):
    """Complete the cmorized file with global metadata."""
    logger.debug("Setting global metadata...")
    attrs = dict(attrs)
    cube.attributes.clear()
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

    ###attr:

    cfg_experiments = _read_config_file('cmor_config/experiments.yml')['experiment_id'][experiment]

    if type(cfg_experiments) == dict: ### switch this to prioritize experiments.yml
        for k, v in cfg_experiments.items():
            if k in attrs.keys():
                continue
            else:
                attrs[k] = v

    #attrs = {**cfg_experiments, **attrs}
        ###attr: member_id
    if attrs['sub_experiment_id'] == 'none':
        member_id = variant_label
    else:
        member_id = attrs['sub_experiment_id'] + '-' + variant_label

    ###attr:
    header = _get_table_header(var['table_id'], attrs['mip_era'])



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

    for k, v in list(attrs.items()):
        if v is None:
            del attrs[k]

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

def _extract_variable(in_files, var, cfg, out_struc, realm, experiment):

    attributes = deepcopy(cfg['attributes'])
    var_attr = {} #stores attributes specific to the variable

    cmor_table = CMOR_TABLES[attributes['mip_era']]

    definition = cmor_table.get_variable(var['table_id'], var['short_name'])

    if 'olevel' in definition.coordinates.keys():
        definition.coordinates.pop('olevel')
        definition.coordinates['depth_coord'] = cmor_table.coords['depth_coord']

    cube, var_attr = _load_cube(in_files, var, var_attr)

    cube, var_attr = _fix_global_metadata(cube, var, attributes, experiment, var_attr)

    cube, var_attr = _fix_variable(cube, definition, var_attr)

    cube, var_attr = _fix_units(cube, definition, var_attr)

    cube = _fix_coordinates(cube, var, definition, realm, base_time=attributes['base_time'])

    var_attr = _get_variable_attrs(var['table_id'], attributes['mip_era'], var['short_name'], var_attr)

    # Convert units if required
    #cube.convert_units(definition.units)

    output_size = np.prod(cube.shape) * 4 / 2**30
    logger.debug("Saving cube\n%s", cube)
    logger.info("Expected output size is %.1fGB",
                 output_size)

    '''if output_size > 2:
        chunksize = list(cube.shape)
        chunksize[1] = 1#int(chunksize[0] / 20)
    else:
        chunksize = list(cube.shape)
        chunksize[0] = int(chunksize[0] / 20)'''

    #chunk = np.prod(chunksize) * 4 / 2**30


    _save_variable(
        cube,
        cube.var_name,
        out_struc,
        var_attr,
        local_keys=['positive'],
    )

    logger.info("Finished CMORizing %s", ', '.join(in_files))

def _save_variable(cube, var_name, out_struc, var_attr, **kwargs):
    """Saver function. Adapted to CMIP6 structure.

    Saves iris cubes (data variables) in CMOR-standard named files.

    Parameters
    ----------
    cube: iris.cube.Cube
        data cube to be saved.

    var_name: str
        Variable short_name e.g. ts or tas.

    outdir: str
        root directory where the file will be saved.

    attrs: dict
        dictionary holding cube metadata attributes like
        project_id, version etc.

    **kwargs: kwargs
        Keyword arguments to be passed to `iris.save`
    """

    out_filepath = out_struc['output_basedir']

    for att in _extract_from_curly_brackets(cube.attributes, out_struc['output_dir_struc'], '/'):
        out_filepath = os.path.join(out_filepath, att)

    out_filename = '_'.join(_extract_from_curly_brackets(cube.attributes, out_struc['output_filename_struc'], '/'))

    out_filepath = out_filepath + out_filename

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


    iris.save(cube, out_filepath, fill_value=1e20, **kwargs)

    ### run fixer, for all things that iris can't handle
    _fixer_script(out_filepath, var_name, var_attr)

    print('Finished cmorizing variable ' + var_name + ' for ' + time_suffix)

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
                except:  # noqa
                    logger.error("Failed to CMORize %s",
                                 ', '.join(futures[future]))
                    raise

def cmorization(cfg_user_filepath):
    """Run CMORizer for FOCI_OCEAN."""

    cfg_user = _read_config_file(cfg_user_filepath)

    in_dir = cfg_user['input_dir']

    cfg = _read_config_file(cfg_user['cmor_config'])

    realm = cfg['realm']
    experiment = cfg['experiment_id']

    cfg['attributes']['realm'] =  realm
    cfg['attributes']['experiment_id'] =  experiment

    cfg.pop('realm')
    cfg.pop('experiment_id')


    out_struc = dict()
    out_struc['output_basedir'] = cfg_user['output_dir']
    out_struc['output_dir_struc'] = cfg['output']['output_dir_structure']
    out_struc['output_filename_struc'] = cfg['output']['output_filename_structure']

    aux_dir = cfg_user['auxiliary_data_dir']

    n_workers = cfg_user.get('max_parallel_tasks')
    if n_workers is None:
        n_workers = int(cpu_count() / 1.5)

    jobs = []


    input = cfg['input']

    for short_name, var in cfg['variables'].items():
        if 'file' not in var.keys():
            searchdict = {'a':cfg['attributes'], 'b': var}
            filename = '_'.join(_extract_from_curly_brackets(searchdict, cfg['input']['input_filename_structure'], '/')) + '.nc'
            var['file'] = filename

        if 'short_name' not in var:
            var['short_name'] = short_name

        if realm == 'ocean':
            gridtype = var['gridtype']

            var['vertices'] = cfg['input']['vertices'][gridtype]
            gridvertices_filepath = os.path.join(aux_dir, var['vertices']['file'])
            var['vertices']['filepath'] = gridvertices_filepath


        for in_files in _get_in_files_by_year(in_dir, var):
            jobs.append([in_files, var, cfg, out_struc, realm, experiment])

    _run(jobs, n_workers)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Incorrect number of arguments')
    cfg_user_filepath = sys.argv[1]

    cmorization(cfg_user_filepath)
