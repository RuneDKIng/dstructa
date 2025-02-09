# -*- coding: utf-8 -*-

"""
@title: dstruct_core
@lastauthor: Rune Inglev
@description: 

"""

import os
import collections

import numpy as np
import pandas as pd

from .dstruct_dtype import dstruct_dtype

# TODO: Create the filereaders!
    
# =====================
#  dstructBase Class
# =====================

class dstruct_core(np.ndarray):

    _dfilters = {}
    _dtype_descr = None
    
    def __new__(cls, data=None, *, dtype=None, size=0, loader=None, path=None, **kwargs):
        """ Constructs a new dstruct_core instance. """
        
        # Convert dtype to dstruct_dtype if possible
        if dtype is not None and not isinstance(dtype,dstruct_dtype):
            if isinstance(dtype, np.dtype):
                dtype = dstruct_dtype(dtype)
            elif isinstance(dtype, list):
                dtype = dstruct_dtype(*dtype)
            else:
                raise ValueError(f"dtype is unknown: {dtype}")
        
        # If only path is given, then replace data with path and set path to None
        if (data is None) and (path is not None):
            data, path = path, None
        
        if data is not None:
            if isinstance(data, str):
                # Join the path and the data to a single filepath
                data = os.path.join(path, data) if path is not None else data
                
                assert os.path.exists(data), f"The path given doesn't exist\n{data}"

                if os.path.isdir(data):
                    return cls.from_folder(data, dtype=dtype, loader=loader, **kwargs)
                
                return cls.from_files([data], None, dtype, loader=loader, **kwargs)
                    
            if isinstance(data, list):
                assert all((type(d) for d in data)), "All items in the list must be the same type!"
                
                if isinstance(data[0],str):
                    return cls.from_files(data, path, dtype, loader=loader)

                if isinstance(data[0],cls):
                    return cls(size=0,dtype=data[0].dtype).append(*data)
                    
                return cls.from_list(data, dtype=dtype)

            if isinstance(data, dict):
                return cls.from_dict(data, dtype=dtype, **kwargs)

            if isinstance(data, pd.DataFrame):
                return cls.from_pandas(data, dtype=dtype, **kwargs)

            if isinstance(data, np.ndarray):
                return cls.from_ndarray(data, dtype=dtype)

            if isinstance(data, cls):
                return cls.from_ndarray(data._array_, dtype=dtype)

        if (data is None) and (path is None) and (dtype is None):
            raise ValueError("Data, path and dtype are all 'None' - I dont know what you want.")
            
        # Create structured array with space for all the measurements
        rsp = np.ndarray.__new__(cls,(size,),dtype=dtype.get_dtype())

        # Populate with default values
        for name, field in dtype.fields.items():
            rsp.view(np.ndarray)[name][:] = field['default']

        # Set the filters from the dtype (there might not be any)
        rsp._dfilters = dtype.filters
            
        return rsp

    def __array_finalize__(self, obj):
        if isinstance(obj,dstruct_core):
            self._dfilters = obj._dfilters

    def __contains__(self,key):
        return key in self.dtype.names

    @property
    def dtypes(self):
        if self._dtype_descr is None:
            self._dtype_descr = [f for f in self.dtype.descr if f[0]]

        return self._dtype_descr
    
    @property
    def fields(self):
        return self.dtype.names
    
    @property
    def array(self):
        """ Return the object as an np.ndarray (copies the data) """
        return np.array(self)

    @property
    def _array_(self):
        """ Return the object as a view to an np.ndarray """
        return np.asarray(self)

    def is_same_type(self,obj):
        """
        Compare the current data type of the dstruct_core instance with a provided data type.

        :param dtype: The data type to compare against. 
        :type dtype: dstruct_dtype or equivalent type

        :return: True if the data types match, False otherwise.
        :rtype: bool

        :raises TypeError: If the provided dtype is not of the expected type.

        This method is useful for ensuring that the dstruct_core instance is compatible with
        expected data types for subsequent processing or analysis.
        """

        for field in self.dtype.names:
            if (field not in obj.dtype.fields) or (self.dtype.fields[field][0] != obj.dtype.fields[field][0]):
                return False

        for field in obj.dtype.names:
            if (field not in self.dtype.fields) or (self.dtype.fields[field][0] != obj.dtype.fields[field][0]):
                return False

        return True
    
    ########################
    # Basic selection stuff
    ########################
            
    def __getattr__(self, attr):
        """ Select data using a specific filter (if set through the dtype) """
        if attr in self._dfilters:
            return self._dfilters[attr](self)

        elif (self.fields is not None) and (attr in self.fields):
            return self[attr]

        raise AttributeError(attr)
    
    def __getitem__(self,key):
        """ Select data using a given key (indices or fieldnames)

        The extra stuff ensures that the returned object always has a
        connection to the original data, no matter how the data was selected
        (eg. with a boolean map or specific indices)
        """
        
        obj = np.ndarray.__getitem__(self, key)

        if isinstance(obj, dstruct_core):
            if hasattr(self , 'base_destruct') or ((obj.base is not None) and (obj.base is not self) and (obj.base is not self.base)):
                obj.base_rsp = self
                obj.base_map = key

        return obj
    
    def __setitem__(self,key,val):
        """ Insert data using the given key. Also inserts data into the original data, if connection exists """
        
        super().__setitem__(key,val)
        
        if hasattr(self, 'base_dstruct'):
            getattr(self,'base_dstruct')[getattr(self,'base_map')] = self
    
    # ======================================
    #  Converters to different object types
    # ======================================
    
    def to_dict(self):
        """ Returns the object as a dictionary """
        return {field: self[field].array for field in self.fields}

    
    def to_list(self):
        """ Returns the object as a list of dictionaries (each dictionary corresponding to a single row in the object) """
        return [dict(zip(self.fields, itm)) for itm in self.array]

    
    def to_ndarray(self):
        """ Returns the object as a np.ndarray (alias of self.array) """
        return self.array

    
    def to_pandas(self):
        """ Convert the object to a pandas dataframe"""

        data_dict = self.to_dict()
        pd_dict = {}
        
        for field,value in data_dict.items():
            if value.ndim > 1:
                for i in range(value.shape[1]):
                    pd_dict[f"{field}_{i}"] = value[:,i]
            else:
                pd_dict[field] = value

        return pd.DataFrame(pd_dict)

    # =============================================
    #  Factory methods from different object types
    # =============================================
    
    @classmethod
    def from_ndarray(cls, data, dtype=None):
        """ Creates an instance from a numpy ndarray with optional data-type given """
        
        if dtype is None:
            dtype = data.dtype

        rsp = cls(size=len(data),dtype=dtype)

        for field in rsp.fields:
            rsp[field] = data[field]

        return rsp

    
    @classmethod
    def from_pandas(cls, df, dtype=None, combine=None, underscore=True):
        """
        Converts a pandas DataFrame into an dstruct instance.

        Parameters:
        - df: pandas DataFrame to convert.
        - dtype: Optional data type for the dstruct fields. If None, the DataFrame's data types will be used.
        - combine: Optional parameter to define how to combine fields (specific usage to be determined by implementation).
        - underscore: If True, field names will be formatted to use underscores instead of spaces.

        Returns:
        - An instance of dstruct populated with the data from the DataFrame.
        """
        
        underscore = "_" if underscore else ""

        # Ensure combine is a list (if given)
        if combine is not None:
            combine = [combine] if not isinstance(combine,list) else combine

        # Create dtype if none is given
        if dtype is None:
            fields = []
            for col in df.columns:
                if combine and any(col.startswith(field) for field in combine):
                    base_field = next(field for field in combine if col.startswith(field))
                    if base_field not in (f[0] for f in fields):
                        num_fields = len([c for c in df.columns if c.startswith(base_field)])
                        fields.append((base_field, df[col].dtype, (num_fields,)))
                else:
                    fields.append((col, df[col].dtype))

            dtype = dstruct_dtype(*fields)

        # Create the placeholder object with the correct length
        rsp = cls(size=len(df),dtype=dtype)

        # Add all the columns to the dstruct object
        for field in rsp.fields:
            if rsp.dtype.fields[field][0].ndim == 0:
                rsp.view(np.ndarray)[field] = df[field].values
            else:
                start = 0 if f"{field}_0" in df.columns else 1
                for i in range(start, rsp.dtype.fields[field][0].shape[0]):
                    rsp.view(np.ndarray)[field][:, i] = df[f"{field}{underscore}{i}"].values
                    
        return rsp


    @classmethod
    def from_list(cls, datalist, dtype=None):
        """
        Converts a list of dictionaries into an dstruct instance.

        Parameters:
        - cls: The class of the dstruct to create.
        - datalist: List of dictionaries to convert, where each dictionary represents a row of data.
        - dtype: Optional custom data type for the dstruct fields. If None, will infer from the datalist.

        Returns:
        - An instance of dstruct populated with data from the list.
        """
        
        # Create dtype if none is given
        if dtype is None:
            fields = {}

            # Go through each item in the datalist and create the field in the dictionary
            # Items in datalist might have different meta-data, this ensures that the
            # final meta-data fields are the superset of all meta-data fields of all items
            for data in datalist:
                for field, value in data.items():
                    if field not in fields:
                        fields[field] = (field, np.array(value).dtype, np.array(value).shape)
                    elif np.array(value).dtype.kind != fields[field][1].kind:
                        raise TypeError(f"Two items in the list have the same fieldname ({field}) but different types ({np.array(value).dtype.kind},{fields[field][1]})!")

            # String fields are converted to "Object" fields
            for name, field in fields.items():
                if field[1].kind in ['U','S']:
                    fields[name] = (field[0], np.dtype("O")) + field[2:]

            dtype = dstruct_dtype(*fields.values())

        # Create the object and reserve memory space
        rsp = cls(size=len(datalist), dtype=dtype)

        # Move data from the list and into the object
        for i, data in enumerate(datalist):
            for field, value in data.items():
                if dtype[field]['dtype'].kind in ['U','S']:
                    value = str(value)
                rsp.view(np.ndarray)[i][field] = value if value is not None else dtype[field]['default']

        return rsp
    
    
    @classmethod
    def from_dict(cls, datadict, dtype=None, broadcast=()):
        """
        Converts a dictionary into an dstruct instance.

        Parameters:
        - cls: The class of the dstruct to create.
        - datadict: Dictionary containing data, where keys are field names and values are the corresponding data for those fields.
        - dtype: Optional custom data type for the dstruct fields. If None, will infer from the datadict.
        - broadcast: Optional parameter to specify how to handle broadcasting of the data across fields.

        Returns:
        - An instance of dstruct populated with data from the dictionary.
        """

        
        # Create dtype if none is given
        if dtype is None:
            fields = {}
            
            for field, value in datadict.items():
                if field not in fields:
                    # Broadcasting multi-dimensional items
                    # (eg. value might be 5 values, but we might want it replicated to each row)
                    if field in broadcast:
                        value = [value]
                    if np.array(value).ndim > 1:
                        fields[field] = (field, np.array(value).dtype, np.array(value).shape[1:])
                    elif np.array(value).ndim == 1:
                        fields[field] = (field, np.array(value).dtype, ())
                    else:
                        datadict[field] = [value]
                        fields[field] = (field, np.array(value).dtype, ())

            # String fields are converted to "Object" fields
            for name, field in fields.items():
                if field[1].kind in ['U','S','O']:
                    fields[name] = (field[0], np.dtype("O")) + field[2:]

                    # This ensures that 'S' type strings are not represented as "bytes" objects
                    datadict[field[0]] = np.array(datadict[field[0]],dtype=str)

            dtype = dstruct_dtype(*fields.values())
            
        else:
            for field, value in datadict.items():
                if value.dtype.kind in ["U","S","O"]:
                    datadict[field] = np.array(datadict[field],dtype=str)
            
        N = np.max([len(field) for field in datadict.values() if field is not None])
            
        rsp = cls(size=N, dtype=dtype)

        for field, value in datadict.items():
            rsp.view(np.ndarray)[field] = value if value is not None else dtype[field]['default']

        return rsp

    @classmethod
    def from_h5(cls, data, path=None, dtype=None):
        """
        Converts data from an H5 file into an dstruct instance.

        Parameters:
        - cls: The class of the dstruct to create.
        - data: H5 file object or path to the H5 file containing the data.
        - path: Optional. The specific path within the H5 file to read the data from.
        - dtype: Optional custom data type for the dstruct fields. If None, will infer from the data in the H5 file.

        Returns:
        - An instance of dstruct populated with data from the H5 file.
        """

        data = [data] if not isinstance(data, list) else data  # Ensure the data is given as a list of filenames

        assert all((d.endswith(".h5") for d in data)), "All files given must be h5 files"
        
        data = [os.path.join(path,d) for d in data] if path else data

        assert all((os.path.exists(d) for d in data)), f"Path to file given, but file does not exist - {data}"

        if len(data) == 1:
            return cls.from_dict(FileReaders.read_h5file(data[0]),dtype=dtype)

        return [cls.from_dict(FileReaders.read_h5file(d),dtype=dtype) for d in data]
    

    @classmethod
    def from_files(cls, data, path, dtype=None, loader=None, **kwargs):
        """
        Converts data from files into an dstruct instance.

        Parameters:
        - cls: The class of the dstruct to create.
        - data: The data source or identifier for the files to load.
        - path: The path to the directory or location of the files.
        - dtype: Optional custom data type for the dstruct fields. If None, will infer from the loaded data.
        - loader: Optional function to specify how to load the data from the files.
        - **kwargs: Additional keyword arguments passed to the loader function.

        Returns:
        - An instance of dstruct populated with data loaded from the specified files.
        """

        data = [data] if not isinstance(data, list) else data  # Ensure the data is a list of filenames
        data = [os.path.join(path,f) for f in data] if path is not None else data
        
        if loader is not None:
            if isinstance(loader,str):
                loader = getattr(FileReaders,loader)

            loaded_data = loader(data, **kwargs)
            
            if isinstance(loaded_data, list):
                if any(isinstance(item, list) for item in loaded_data):
                    loaded_data = [(item if isinstance(sublist,list) else sublist) for sublist in loaded_data for item in sublist]
                return cls.from_list(loaded_data, dtype=dtype)

            if isinstance(loaded_data,dict):
                return cls.from_dict(loaded_data, dtype=dtype)

            if isinstance(loaded_data, pd.DataFrame):
                return cls.from_pandas(loaded_data, dtype=dtype)

            if isinstance(loaded_data, cls):
                return loaded_data
            
            if isinstance(loaded_data, np.ndarray): # Must be after cls!!!
                return cls.from_ndarray(loaded_data, dtype=dtype)

            raise ValueError(f"Loader returned something unknown ({type(loaded_data)})")

        if data[0].endswith(".h5"):
            loaded = cls.from_h5(data, path, dtype, **kwargs)
            return loaded[0].append(*loaded[1:]) if isinstance(loaded, list) else loaded

        raise ValueError("Unknown file extension for loading files automatically")

    
    @classmethod
    def from_folder(cls, path, dtype=None, file_filter=None, **kwargs):
        """
        Converts data from multiple files in a specified folder into an dstruct instance.

        Parameters:
        - cls: The class of the dstruct to create.
        - path: The path to the folder containing the files.
        - dtype: Optional custom data type for the dstruct fields. If None, will infer from the loaded data.
        - file_filter: Optional function to filter which files to load based on their names or properties.
        - **kwargs: Additional keyword arguments passed to the loader function.

        Returns:
        - An instance of dstruct populated with data from the files in the specified folder.
        """
        
        files = os.listdir(path)

        if file_filter is not None:
            files = list(filter(file_filter,files))

        return cls.from_files(files, path, dtype=dtype, **kwargs)


    # ===============================
    #  Implementation of array_ufunc
    # ===============================
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handles universal functions (ufuncs) for element-wise operations on the dstruct instance.

        Parameters:
        - ufunc: The ufunc to apply.
        - method: The method of the ufunc to use (e.g., 'outer', 'reduce').
        - inputs: The input data for the ufunc.
        - kwargs: Additional keyword arguments for the ufunc.

        This method allows dstruct objects to participate in NumPy ufunc operations,
        ensuring that operations are applied correctly across the fields of the
        structured array based on the defined dtype.

        Returns:
        - Depending on the ufunc and method, returns a new dstruct instance or a
          modified version of the existing instance.
        """
        
        inputs = [inp._array_ if isinstance(inp,self.__class__) else inp for inp in inputs]
        output = kwargs.get('out',(None,))[0]
        
        if method == "__call__":
            if self.fields is None:
                kwargs['out'] = output._array_ if isinstance(output,self.__class__) else output
                result = ufunc(*inputs,**kwargs)
                if output is not None:
                    output[:] = result
                    return output
                return result
            
            result = self.copy() if output is None else output
            
            for field in result.fields:
                if result.dtype[field].base.kind in ['U','S','O','M']:
                    result[field] = self.view(np.ndarray)[field]
                else:
                    inpf = [inp[field] if inp.dtype.kind=='V' else inp for inp in inputs[1:]]
                    ufunc_res = ufunc(self.view(np.ndarray)[field],*inpf)
                    if result[field].dtype != ufunc_res.dtype:
                        result = result.reform_field(field, ufunc_res.dtype)
                    result[field] = ufunc_res

            return result

        if method == "reduce":
            if self.fields is None:
                return ufunc.reduce(*inputs, **kwargs)
            
            result = self.__class__(size=1,dtype=self.dtype)
            
            for field in result.fields:
                if result.dtype[field].base.kind in ['U','S','O']:
                    uniques = collections.Counter(self.view(np.ndarray)[field])
                    result[field] = "" if (len(uniques) > 1 or len(uniques) == 0) else next(iter(uniques))
                elif (result.dtype[field].base.kind == 'M'): #datetime
                    result[field] = self.view(np.ndarray)[field][0] # Takes the first value when reducing on datetimes
                else:
                    ufunc_res = ufunc.reduce(self.view(np.ndarray)[field],axis=0)
                    if result[field].dtype != ufunc_res.dtype:
                        result = result.reform_field(field, ufunc_res.dtype)
                    result[field] = ufunc_res
            return result
        
        return NotImplemented
