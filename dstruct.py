# -*- coding: utf-8 -*-

"""
@title: dstruct
@lastauthor: Rune Inglev
@description: This file contains the 'dstruct' class, which is the entry point for creating dstruct objects.
"""

import re
import json
import collections

import numpy as np

from .core.dstruct_core import dstruct_core
from .core.dstruct_dtype import dstruct_dtype
from .core.dstruct_print import NumpyPrettyPrint

# ================================================
#  dstruct Class
# ================================================

class dstruct(dstruct_core):

    @property
    def array(self):
        """ Return self as a view of normal np.ndarray """
        return np.array(self)

    def info(self):
        """ Print info about the rsp object """
        print("INFO ABOUT OBJECT", repr(self), f"(View: {self.is_view})")
        print(f"FIELDS: {np.array(self.fields)}")
        print("UNIQUE VALUES")
        for field in self.fields:
            fieldview = self[field]
            if fieldview.ndim > 1 and fieldview.shape[1] > 10:
                print(f"{field}: Many... (big matrix)")
                continue
            if fieldview.dtype.kind == 'O':
                coll = collections.Counter(fieldview.ravel())
                unique = list(coll.elements())
            else:
                unique = np.unique(self[field]) if self[field].ndim < 2 else np.unique(self[field],axis=0)
            print(f"{field}: {list(unique)}" if len(unique) < 11 else f"{field}: {len(unique)} unique values")

    def unique(self, fields):
        return np.unique(self[fields])

    def head(self,num_lines=20):
        """ Show the first {num_lines} items of the object """
        print(NumpyPrettyPrint(self[0:num_lines],limit=num_lines))
        
    def tail(self,num_lines=20):
        """ Show the first {num_lines} items of the object """
        print(NumpyPrettyPrint(self[-num_lines:],limit=num_lines))
        
    def show(self,limit=20):
        """ Show up to {num_lines} items from the object divided between tail and head """
        print(NumpyPrettyPrint(self,limit=limit))

    def __repr__(self):
        """ The string used for 'simple' display (the one that would be shown in a kernel) """
        if self.fields and ('Data' not in self.fields):
            return f"dstruct(size={len(self)})[{self.baseid}]" if self.dtype.names is not None else str(self)

        return f"dstruct(size={len(self)})[{self.baseid}]" if self.dtype.names is not None else str(self)
    
    def __str__(self):
        """ The string used in a 'print' display """
        return NumpyPrettyPrint(self,limit=20)
            
    @property
    def is_view(self):
        """ Return true or false if the object is a view """
        return hasattr(self,'base_rsp')
    
    @property
    def baseid(self):
        """ Return the id of the base data object """
        return getattr(self,'base_rsp').baseid if hasattr(self,'base_rsp') else hex(id(self))

    def cutout(self,field):
        class Cutout:
            def __init__(self, parent):
                self.parent = parent

            def __getitem__(self, index):
                if isinstance(index, tuple):
                    if len(index) == 2:
                        return self.parent.cutout_field("Data", self.parent[field][:,index[0],index[1]])
                    elif len(index) == 1:
                        return self.parent.cutout_field("Data", self.parent[field][:,index[0]])
                else:
                    return self.parent.cutout_field("Data", self.parent[field][:,index])

        return Cutout(self)

    ######################################################
    # Advanced selection (using underlying map function)
    ######################################################
            
    def select(self,where,index=False,inverse=False):
        """ Return where the comparison is true (either values or indices) """
        if (where is None):
            return np.arange(len(self))
        
        idxs = np.atleast_1d(np.squeeze(np.argwhere(self.mask(where,inverse=inverse))))
        return idxs if index else self[idxs]
    
    def where(self,where,index=False,inverse=False):
        """ Same as select """
        return self.select(where,index=index,inverse=inverse)
    
    def index(self,where,inverse=False):
        """ Return where the comparison is true (as indices) Alias for self.where(..,..,..,index=True) """
        return self.select(where,index=True,inverse=inverse)
    
    def set(self,where,values):
        """ Set the values at the indices that fulfill 'where' """
        self[self.mask(where)] = values
        
    def groupby(self,*by,where=None):
        """ Return groups

        what: Fields to be grouped by (in order)
        mask: Whether the grouper should return a mask instead of dstruct objects
        where: Extra conditions to be fulfilled by the groups

        return: grouper (function) that takes index and group list
        
        """
                
        return dstructGroupBy(self, *by, where=where)

    def iterate(self,*what,where=None,mask=False):
        """ Iterate over specified items, may return mask if mask is set to True. Where is extra conditionals """

        grouper = self.groupby(*what,where=where)
        
        for group in grouper:
            yield group if mask is False else group.group_mask

    ##############################
    # Data-object manipulation
    ##############################
            
    def append(self,*data):
        """ Append other dstruct objects to the end """
        
        if not all(self.is_same_type(instance) for instance in data):
            raise ValueError("All instances in data must be the same dtype as the object appending to")

        sizes = [len(item) for item in data]
        
        total_size = sum(sizes) + len(self)

        new_rsp = dstruct(size=total_size, dtype=self.dtype)

        idx = len(self)

        new_rsp[:idx] = self
        
        for size, item in zip(sizes,data):
            for field in self.fields:
                new_rsp.view(np.ndarray)[field][idx:idx+size] = item[field]

            idx = idx+size
            
        return new_rsp
    
    def add_field(self, fieldname, dtype, shape=(), default=None, loc=-1):
        """ Create a new field for dstruct object """
        
        dtype_old = self.dtypes

        new_field = (fieldname, dtype, shape) if shape else (fieldname, dtype)
        
        if loc == -1:
            dtype_new = dtype_old + [new_field]
        elif loc == 0:
            dtype_new = [new_field] + dtype_old
        else:
            dtype_new = dtype_old[:loc] + [new_field] + dtype_old[loc:]
            
        new_rsp = dstruct(size=len(self), dtype=np.dtype(dtype_new, align=True))
        new_rsp._dtype_descr = dtype_new

        for field in self.fields:  
            new_rsp[field] = self[field]
            
        new_rsp[fieldname] = dstruct_dtype.get_default(np.dtype(dtype), default)

        return new_rsp

    def remove_field(self, fields):
        """ Remove a given field """
        fields = [fields] if not isinstance(fields, list) else fields
        
        dtype_new = [f[1] for f in zip(self.fields,self.dtypes) if f[0] not in fields]

        new_rsp = dstruct(size=len(self), dtype=np.dtype(dtype_new),align=True)

        for field in new_rsp.fields:
            new_rsp[field] = self[field]

        return new_rsp

    def replace_field(self, fieldname, dtype, shape=(), default=None):
        """ Replace a field with a new description """

        assert fieldname in self.fields, "Field doesn't exist!"

        loc = self.fields.index(fieldname)

        dtype_new = self.dtypes.copy() 
        dtype_new[loc] = (fieldname, dtype, shape) if shape else (fieldname, dtype)

        new_rsp = dstruct(size=len(self), dtype=np.dtype(dtype_new,align=True))

        new_rsp._dtype_descr = dtype_new
        
        for field in self.fields:
            if field != fieldname:
                new_rsp.view(np.ndarray)[field] = self.view(np.ndarray)[field]

        if default is not None:
            new_rsp.view(np.ndarray)[fieldname] = default

        return new_rsp

    def rename_field(self, fieldname, newname):
        """ Replace a field with a new description """

        assert fieldname in self.fields, "Field doesn't exist!"

        loc = self.fields.index(fieldname)
        
        dtype_new = self.dtypes.copy()
        dtype_new[loc] = (newname,) + dtype_new[loc][1:]
            
        new_rsp = dstruct(size=len(self), dtype=np.dtype(dtype_new,align=True))
        
        for field in self.fields:
            if field != fieldname:
                new_rsp[field] = self[field]
            else:
                new_rsp[newname] = self[field]

        return new_rsp

    def reform_field(self, fieldname, dtype):
        """ Reform the field into a new dtype (often used to change floats into int or vice versa) """

        new_rsp = self.replace_field(fieldname, dtype, shape=self.dtype[fieldname].shape) # pylint: disable=unsubscriptable-object

        new_rsp[fieldname] = self[fieldname]

        return new_rsp

    def reshape_field(self, fieldname, shape):
        """ Reshape the field into a new dtype """

        new_rsp = self.replace_field(fieldname, dtype=self.dtype[fieldname].base, shape=shape) # pylint: disable=unsubscriptable-object

        return new_rsp

    def cutout_field(self, fieldname, cutout):
        """ Replace a field with a slice of itself """
        
        new_rsp = self.reshape_field(fieldname, np.squeeze(cutout).shape[1:])

        new_rsp[fieldname] = np.squeeze(cutout)
        
        return new_rsp

    def slice_field(self, fieldname, cutout):
        return self.cutout_field(fieldname, cutout)
    
    def remove_where(self, where):
        return np.delete(self, self.index(where))

    def delete_where(self, where):
        return self.remove_where(where)

    def reduce_where(self, where, func, fields=None):
        idxs = self.index(where) if where else np.arange(0,len(self))

        result = np.delete(self, idxs[1:])
        result = result if fields is None else result[fields].copy()
                
        data = self[idxs] if fields is None else self[fields][idxs]
        data = func(data, axis=0) if callable(func) else func
    
        if data.dtype.kind == 'V':
            for field in data.fields:
                if data.dtype[field] != result.dtype[field]:
                    result = result.reform_field(field, data[field].dtype)

        elif data.dtype != result.dtype:
            result = np.array(result,dtype=data.dtype)
        
        result[idxs[0]] = data

        return result
    
    def reduceat(self, where, func, fields=None):
        return self.reduce_where(where, func, fields=fields)

    def reduce_rolling(self, func, window, step=1, where=None, **kwargs):
        if where is not None:
            data_roll = self.where(where).reduce_rolling(func, window, step, **kwargs)
            data_noroll = self.where(where,inverse=True)

            return data_roll.append(data_noroll)

        indices = []
        idx = 0
        max_idx = len(self)

        while idx < max_idx:
            indices.append(idx)
            if callable(step):
                idx = next((idx + j for j in range(1, max_idx - idx) if step(self[idx:idx + j + 1], **kwargs)), max_idx)
            else:
                idx += step  # Assuming step is an integer for non-callable case

        
        res = dstruct(size=len(indices),dtype=self.dtypes)

        for i,idx in enumerate(indices):
            if callable(window):
                j = next((j for j in range(1,len(self[idx:])) if window(self[idx:idx+j+1],**kwargs)),-1)
            else:
                j = window

            data = func(self[idx:]) if j == -1 else func(self[idx:idx+j])
            for field in res.fields:
                if data.dtype.fields[field][0] != res.dtype.fields[field][0]:
                    res = res.reform_field(field, data[field].dtype)

            res[i] = data

        return res

    def combine_fields(self, fields, func, to_field=None, dtype=None, **kwargs):

        res = np.array([func(*r[fields], **kwargs) for r in self])

        if dtype is not None:
            res = res.astype(dtype)
        
        if to_field is None:
            to_field = "+".join(fields)
            
        obj = self.add_field(to_field, res.dtype, res.shape[1:])

        obj[to_field] = res

        return obj
    
    def reduce_field(self, fields, insert, axis=None, to_field=None, **kwargs):
        assert axis != 0, "Axis cannot be 0, use 'reduceat' if you want to reduce that direction"

        fields = [fields] if not isinstance(fields, list) else fields

        if to_field is not None:
            to_fields = [to_field] if not isinstance(to_field, list) else to_field
            assert len(fields)==len(to_fields), f"Different number of fields to be reduced, and their destination\n{fields} vs {to_fields}"
        else:
            to_fields = fields
            
        obj = self
        
        for field, to_field in zip(fields,to_fields):
            field_data = self.view(np.ndarray)[field]

            if field_data.ndim == 1:
                continue
            
            if axis is None:                
                for _ in range(field_data.ndim-1):
                    field_data = insert(field_data, axis=-1, **kwargs) if callable(insert) else insert
            else:
                field_data = insert(field_data, axis=axis, **kwargs) if callable(insert) else insert

            if to_field not in obj:
                obj = obj.add_field(to_field, field_data.dtype, field_data.shape[1:])
            else:
                obj = obj.replace_field(to_field,field_data.dtype,field_data.shape[1:])
                
            obj[to_field] = field_data
                
        return obj

    def applyat(self, fields, func, *args, where=None, copy=False, **kwargs):
        result = self.copy() if copy else self
        if where is not None:
            result.where(where)[fields] = func(result.where(where)[fields], *args, **kwargs)
        else:
            result[fields] = func(result[fields],*args,**kwargs)

        return result
                    
    def mask(self,where,*args,masks=None,outer=True,inverse=False,literals=None):
        """ Return the mask where the thing is true """

        if callable(where):
            return np.fromiter((bool(where(row)) for row in self), dtype=bool, count=len(self))
        
        masks = [] if masks is None else masks
        
        if outer:
            literals = {}
            for idx, literal in enumerate(re.findall(r"(\"[^\"]*\"|'[^']*')",where)):
                placeholder = f"__STR{idx}__"
                where = where.replace(literal, placeholder)
                literals[placeholder] = literal[1:-1]
        
        while '(' in where:
            where = re.sub(r'\(([^()]+)\)',lambda m: f'{self.mask(m[1],*args,masks=masks,outer=False, literals=literals)}',where)
            
        where = where.replace("and","&&").replace("or", '||')
        where = re.split(r"\s(&&|\|\|)\s",where)
        
        where_ops = where[1::2]
        where_exp = where[::2]
        
        # Evaluate simple expressions or get the mask from inner parentheses
        for i, exp in enumerate(where_exp):
            if re.match(r"#\d+#",exp.strip()) is not None:
                where_exp[i] = masks.pop(0)
            else:
                match = re.match(r"(\w+)\s*(!=|==|<=|>=|<|>|in|not in)\s*([\s\-\_\[\]\w,\.%\'\"]+)",exp.strip())
                if arg := re.match(r"\%(\d+)",match[3]):
                    where_exp[i] = self._compare(match[1],args[int(arg[1])],match[2])
                else:
                    value = match[3]
                    for plch,literal in literals.items():
                        value = value.replace(plch, f"'{literal}'" if match[2] in ['in','not in'] else literal)
                    where_exp[i] = self._compare(match[1],value,match[2])
    
        # Successivly join expressions with booleans (AND then OR)        
        for operation, func in {'&&': np.logical_and, '||': np.logical_or}.items():
            while operation in where_ops:
                for i, bool_op in enumerate(where_ops):
                    if isinstance(bool_op,str) and (bool_op == operation):
                        where_exp[i+1] = func(where_exp[i],where_exp[i+1])
                        where_exp[i] = where_ops[i] = None
                
                where_exp = [exp for exp in where_exp if exp is not None]
                where_ops = [bop for bop in where_ops if bop is not None]

        if outer:
            return ~where_exp[0] if inverse else where_exp[0]
                
        mask_name = f'#{len(masks)}#'
        masks.append(where_exp[0])
        
        return mask_name

    def _compare(self,where,value,comparator):
        
        """ Return a mask which can be used to access the specific elements requested """
        
        comp = {'==': {'X': np.equal,         'U': np.char.equal},
                '<=': {'X': np.less_equal,    'U': np.char.less_equal},
                '<' : {'X': np.less,          'U': np.char.less},
                '>' : {'X': np.greater,       'U': np.char.greater},
                '>=': {'X': np.greater_equal, 'U': np.char.greater_equal},
                '!=': {'X': np.not_equal,     'U': np.char.not_equal},
                'in': {'X': np.isin,          'U': np.isin},
                'not in': {'X': lambda r,t: np.isin(r,t,invert=True), 'U': lambda r,t: np.isin(r,t,invert=True)}}
        
        assert comparator in comp, "Unknown comparison operator!"
        
        where = self.view(np.ndarray)[where]
        
        compare = comp[comparator]['U'] if where.dtype.kind in ['U','S','O'] else comp[comparator]['X']

        if where.dtype.kind == 'O':
            where = np.array(where,dtype=str)
        
        if comparator in ['in', 'not in']:
            value = json.loads(value.replace("'",'"')) if isinstance(value,str) else value
        else:
            value = json.loads(value) if (isinstance(value,str) and where.dtype.kind!='U') else value
        
        return compare(where,value)


# ===========================
#  GROUP BY IMPLEMENTATION
# ===========================

class dstructGroupBy:
    def __init__(self, rsp, *by, where=None):
        self.rsp = rsp
        self.by = list(by)

        self.group_list, idxs, self.group_map = np.unique(rsp.view(np.ndarray)[self.by], return_inverse=True, return_index=True)

        argsort = np.argsort(idxs)

        self.group_list = self.group_list[argsort]
        
        self.group_masks = [self.group_map==idx for idx in argsort]

        if where:
            where_mask = rsp.mask(where)
            self.group_masks = [(mask & where_mask) for mask in self.group_masks]

        self.group_indices = [np.atleast_1d(np.squeeze(np.argwhere(mask))) for mask in self.group_masks]
        self.group_indices = [idxs for idxs in self.group_indices if len(idxs)]

        self.length = len(self.group_indices)
        
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.groups[idx]
        
        rsp = self.rsp[self.group_indices[idx]]
        rsp.group = self.group(idx)
        rsp.group_mask = self.mask(idx)
        rsp.indices = self.indices(idx)
        for by in self.by:
            setattr(rsp,by,rsp.group[by])
            
        return rsp

    def __iter__(self):
        for idx in range(self.length):
            yield self[idx]

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"{object.__repr__(self)}\nGroups: {len(self)}"

    @property
    def groups(self):
        return self.group_list

    def group(self, idx):
        return self.group_list[idx]

    def mask(self, idx):
        return self.group_masks[idx]

    def indices(self, idx):
        return self.group_indices[idx]

    def combine(self, func):
        """ Combine the groups based on some function """
        combine_groups = [group for group in self if func(group)]

        return combine_groups[0].append(*combine_groups[1:])

    def reduceat(self, where, func, fields=None):
        if where is not None:
            length = len(self.groups) + len(self.rsp.index(where,inverse=True))
        else:
            length = len(self.groups)

        dtype = self.rsp.dtype if fields is None else self.rsp.dtype[fields]
        rsp = dstruct(size=length,dtype=dtype)            
        idx = 0
        
        for group in self:
            result = group.reduceat(where, func, fields=fields)
            end = idx+len(result)

            if result.fields:
                for field in result.fields:
                    rsp[idx:end][field] = result[field]
            else:
                rsp[idx:end] = result

            idx = end

        return rsp

    def mean(self):
        return self.reduceat(None, np.mean)
