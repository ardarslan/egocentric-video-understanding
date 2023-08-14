import io
import json
import pickle
import zipfile

from utils.unpicklers import CPUUnpickler


def read_json(path):
    if path.lower().endswith(".zip"):
        zip_obj = zipfile.ZipFile(path)
        json_bytes = zip_obj.read(zip_obj.namelist()[0])
        return json.load(io.BytesIO(json_bytes))
    else:
        with open(path, "r") as f:
            return json.load(f)


def read_pkl(path):
    if path.lower().endswith(".zip"):
        zip_obj = zipfile.ZipFile(path)
        pkl_bytes = zip_obj.read(zip_obj.namelist()[0])
    else:
        with open(path, "rb") as f:
            pkl_bytes = f.read()

    ret = CPUUnpickler(io.BytesIO(pkl_bytes)).load()
    return ret
