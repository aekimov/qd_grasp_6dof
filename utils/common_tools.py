
import pdb

from functools import partial, update_wrapper
import pathlib
from pathlib import Path

SUPPORTED_DUMP_PATH_TYPES = {str, pathlib.PosixPath}


def wrapped_partial(func, *args, **kwargs):
    """Taken from: http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/"""
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def arg_clean_str(x):
    return str(x).strip().lower()


def get_local_run_name(log_path, folder_name):
    Path(log_path).mkdir(exist_ok=True)
    id_run_export, valid_run_name_is_found = 0, False
    while not valid_run_name_is_found:
        run_name = Path(f"{log_path}/{folder_name}{id_run_export}")
        if not run_name.exists():
            valid_run_name_is_found = True
        else:
            id_run_export += 1

    return run_name


def get_new_run_name(log_path, folder_name, verbose=True):

    run_name = get_local_run_name(log_path, folder_name)

    run_name.mkdir(exist_ok=True)

    if verbose:
        print(f'Output folder (run_name={run_name}) has been successfully build.')

    return run_name


def is_export_path_type_valid(dump_path, attempted_export_str='data'):
    if type(dump_path) not in SUPPORTED_DUMP_PATH_TYPES:
        print(
            f'[plot_export_routine] Warning: dump_path not in {SUPPORTED_DUMP_PATH_TYPES} '
            f'(type={type(dump_path)}; cannot export {attempted_export_str}.'
        )
        return False

    return True


def get_export_path_root(dump_path):
    assert type(dump_path) in SUPPORTED_DUMP_PATH_TYPES
    return str(dump_path) if not isinstance(dump_path, str) else dump_path


def project_from_to_value(interval_from, interval_to, x_start):
    """ Apply the following projection:
    interval_start -> interval_stop
    x_start |---> x_stop
    :return:
    """
    coef = (x_start - interval_from[0]) / (interval_from[1] - interval_from[0])
    x_stop = interval_to[0] + coef * (interval_to[1] - interval_to[0])
    return x_stop

