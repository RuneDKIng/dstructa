# -*- coding: utf-8 -*-

"""
@title: dstruct_print
@lastauthor: Rune Inglev
@description: Printing functions for the numpy structured array.
"""

import numpy as np

def NumpyPrettyPrint(obj,limit=10):
    """
    Convert a structured numpy array to a human-readable string format.

    Parameters:
    obj (NumpyPrettyPrint): The structured numpy array to be converted.
    limit (int): The maximum number of elements to display from the array.

    Returns:
    str: A formatted string representation of the structured array.

    Notes:
    The output will be truncated to the specified limit if the array exceeds this size.
    """    

    with np.printoptions(threshold=15,linewidth=100):
        if not obj.dtype.names:
            return repr(obj.view(np.ndarray))
            
        head_rows = [list(row) for row in obj[0:limit//2]] if len(obj) > limit else list(obj)
        tail_rows = [list(row) for row in obj[-(limit//2):]] if len(obj) > limit else []

        types = [f"{obj.dtype.base[f]}" for f in obj.fields]

        # Expand multi-dimensional fields
        if any(obj.dtype[f].ndim > 1 for f in obj.fields):
            exp_rows = ExpandSubarrayRows(head_rows+tail_rows)
            tail_rows = exp_rows[len(head_rows)*3:]
            head_rows = exp_rows[:len(head_rows)*3]

        rows = [obj.fields] + [types] + head_rows + tail_rows

        widths = np.max([[max(map(len,s.split("\n"))) for s in list(map(str,row))] for row in rows],axis=0)

        string = RowToStr(obj.fields,widths) + "\n"
        string += RowToStr(types,widths) + "\n"
        string += "\n".join([RowToStr(row,widths) for row in head_rows])
        
        if tail_rows:
            string += f"\n\n .... [{len(obj)-limit} more] .... \n\n"
            string += "\n".join([RowToStr(row,widths) for row in tail_rows])

    return string

def ExpandSubarrayRows(rows):
    
    transposed = []
    shapes = []

    # Trans pose the rows and fields
    for row in rows:
        for i,elm in enumerate(row):
            if len(transposed) == i:
                transposed.append([])
            if hasattr(elm,'ndim') and (elm.ndim > 1):
                transposed[i].append(elm[[0,-1]])
                shapes.append(elm.shape)
            else:
                transposed[i].append(elm)
    
    for i,elm in enumerate(transposed):
        elm = np.array(elm)
        
        if elm.ndim > 2:
            elm = np.concatenate([elm[...,0:3],elm[...,-3:]],axis=-1) # Grab the end points of the array
            elm_str = np.array2string(elm, edgeitems=len(rows), precision=3)[1:-1] # Print it
            elm_rows = elm_str.split("\n\n") # Split into each row again
            transposed[i] = [[e[0].strip(), shapes.pop(), e[1].strip()] for e in [r.split("\n") for r in elm_rows]]
            
    rows = []

    for elm in transposed:
        for i, row in enumerate(elm):
            if len(rows) == i:
                rows.append([])
            rows[i].append(row)
    
    exp_rows = []
    for row in rows:
        new_rows = ExpandRow(row)

        exp_rows.append(new_rows[0])
        exp_rows.append(new_rows[1])
        exp_rows.append(new_rows[2])

    return exp_rows

def ExpandRow(row):
    new_rows = [[],[],[]]
    
    for elm in row:
        if isinstance(elm,list):
            line0 = f"{elm[0][0]} {elm[0][1:len(elm[0])//2]} ... {elm[0][len(elm[0])//2:]} "
            line1 = f"{elm[1]}"
            line2 = f"  {elm[2][:len(elm[2])//2]} ... {elm[2][len(elm[2])//2:-1]} {elm[2][-1]}"
            width = max([len(line0), len(line2)])
            cent_width = len(f"{elm[1]}")
            padding = max((width-cent_width-6)//2, 0)
            new_rows[0].append(line0)
            new_rows[1].append(f"  [{'':<{padding}}{line1}{'':<{padding+1}}]")
            new_rows[2].append(line2)
        else:
            new_rows[0].append("")
            new_rows[1].append(elm)
            new_rows[2].append("")

    return new_rows

def RowToStr(elm,widths=None):
    """ Create a string representation of a single element """
    if widths is not None:
        return "  ".join([f"{str(v):<{w}}" for v,w in zip(elm,widths)])

    return "  ".join([str(v) for v in elm])
