# -*- coding: utf-8 -*-

"""
@title: dstruct_dtype
@lastauthor: Rune Inglev
@description: This file contains the "dstruct"-structure which contains both spectral and meta-data.
"""

import numpy as np

# =========================
#  dstruct_dtype Class
# =========================

class dstruct_dtype:
    
    DEFAULT_VALUES = {'b': False, 'i': -1, 'u': 0, 'f': np.nan, 'U': '', 'M': 0, 'O': '', 'S': ''}
    
    def __init__(self, *fields):
        self.fields = {}
        self.filters = {}
        
        if len(fields) == 1 and isinstance(fields[0], np.dtype):
            self.from_np_dtype(fields[0])
        else:
            for field in fields:
                self.add_field(*field)

    def __getitem__(self, key):
        return self.fields[key]

    def __iter__(self):
        yield from self.fields.values()

    def add_field(self, field, dtype, shape=(), default=None, alias=None):
        dtype = np.dtype(dtype)
        default = dstruct_dtype.DEFAULT_VALUES[dtype.kind] if default is None else default
        alias, field = (field[0],field[1]) if isinstance(field,tuple) else (alias, field)
        name = (alias, field) if alias is not None else field
        
        self.fields[field] = {'name': name, 'dtype': dtype, 'shape': shape, 'default': default, 'alias': alias}

    def add_filter(self, name, filter_string):
        self.filters[name] = filter_string
            
    def get_dtype(self):
        return np.dtype([(field['name'], field['dtype'], field['shape']) for field in self], align=True)

    def from_np_dtype(self, np_dtype):
        """Convert a numpy structured dtype to dstruct_dtype fields."""
        for field in np_dtype.names:
            alias = np_dtype.fields[field][2] if len(np_dtype.fields[field]) > 2 else None
            dtype = np_dtype[field].base  # Get the dtype 
            shape = np_dtype[field].shape
            default = dstruct_dtype.DEFAULT_VALUES[dtype.kind]  # Default based on kind
            self.add_field(field, dtype, shape, default, alias)

    @staticmethod
    def get_default(dtype, default=None):
        return dstruct_dtype.DEFAULT_VALUES[dtype.kind] if default is None else default


