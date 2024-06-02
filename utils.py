import yaml
from pathlib import Path
from collections import defaultdict

def _read_config_file(config_file):
    """Read config user file and store settings in a dictionary."""
    config_file = Path(config_file)

    if not config_file.exists():
        raise IOError(f'Config file `{config_file}` does not exist.')

    with open(config_file, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    return cfg

def _get_in_files_by_year(in_dir, var):
    """Find input files by year."""

    if 'file' in var:
        var['files'] = [var.pop('file')]

    in_files = defaultdict(list)

    for pattern in var['files']:
        pos_of_asterisk = pattern.rsplit('_').index('*')
        for filename in Path(in_dir).glob(pattern):

            year = str(filename.stem).rsplit('_')[pos_of_asterisk]
            in_files[year].append(str(filename))

    # Check if files are complete
    for year in in_files.copy():
        if len(in_files[year]) != len(var['files']):
            print(
                "Skipping CMORizing %s for year '%s', %s input files needed, "
                "but found only %s", var['short_name'], year,
                len(var['files']), ', '.join(in_files[year]))
            in_files.pop(year)


    return in_files.values()

def _extract_from_curly_brackets(dict, struc, delimiter_in):
    items_list = struc.split(delimiter_in)

    rs = ''
    attr = ''

    in_brackets=False
    out_list = []


    for sstr in items_list:
        for t in sstr:
            if t == '{':
                in_brackets=True
                continue

            if t == '}':
                in_brackets=False
                vattrl = list(gen_dict_extract(attr, dict))

                if len(vattrl) > 1:
                    raise ValueError("Use of ... is ambigiuous")
                elif len(vattrl) == 0:
                    raise ValueError("Attribute ... not found")

                vattr = vattrl[0]

                rs = rs + vattr
                attr=''
                continue
            if in_brackets:
                attr = attr + t
            else:
                rs = rs + t

        out_list.append(rs)
        rs=''
    return out_list

def gen_dict_extract(key, var):
    """
    From https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists
    """
    if hasattr(var,'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result
