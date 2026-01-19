# -*- coding: utf-8 -*-
"""
@author: Jonathan E. Robinson

read_jsonl.py - Utilities to read and chunk JSONL (newline-delimited JSON) files.

Functions:
- read_jsonl_with_pandas(path): load into a pandas DataFrame (if pandas installed).
- save_working(path, pd, suff) # saves current version (<path>, <pandas dataFrame>, <suffice for file>)
- check_save_integrity(name, df, path) # optional step to make sure saved file can be save and loaded without data lose
- read(sample, chunks=None) # top level command to read datafile and chunk for processing if needed.

"""

import pandas as pd
from datetime import date


def read_jsonl_with_pandas(path: str, chunks=None):
    """
    Read a .jsonl file into a pandas DataFrame.
    
    Parameters
    ----------
    name
        file name string
    chunk
        integer of the size of chunk (if non file will just be read)

    """

    if chunks:
        print("Chunking into "+str(chunks)+" width chunks.")
        # pandas handles compression based on filename extension
        return pd.read_json(path, lines=True, chunksize=chunks)
    else:
        # pandas handles compression based on filename extension
        return pd.read_json(path, lines=True)


def save_working(path: str, pd: pd.DataFrame, suff: str):
    """
    Save pandas dataframe to working version
    
    Parameters
    ----------
    path
        string of save path
    pd
        pandas DataFrame to save
    suffix
        suffix to add to file 

    """
    split_str = path.split('.')
    pre_fn = ''.join(split_str[0:len(split_str)-1])
    day = "_"+''.join(str(date.today()).split('-'))
    ftype = '.'+split_str[-1]    
    fname = pre_fn+suff+day+ftype
    pd.to_json(fname,orient='records',lines=True)
    
    
def check_save_integrity(name: str, df: pd.DataFrame, path: str):
    """
    Check save integrity of save by reloading to check for read and data lose
    
    Parameters
    ----------
    name
        file name string
    df
        pandas DataFrame to save
    path
        string of save path
    
    """
    try:
        save_df = read_jsonl_with_pandas(path)
        if save_df.shpae == df.shape():
            print('Data saved: intergrity confirmed')
        else:
            raise Exception('Data Integrity failure')
    except:
        print('Saving error: Retrying')
        save_working(name, df, '_mod.')
    


def read(filename: str, chunks=None):
    """
    Read in a data file and chunk ready for parallelised analysis if required
    
    Parameters
    ----------
    name
        file name string
    chunk
        integer of the size of chunk (if non file will just be read)
    
    """
    
    try:
        if chunks:
            df = read_jsonl_with_pandas(filename,chunks)
            print("\n"+filename)
            print("Loaded into pandas DataFrame.")
            
        else:
            df = read_jsonl_with_pandas(filename)
            print("\n"+filename)
            print("Loaded into pandas DataFrame.")
            save_working(filename, df, '_mod.')
            
            
        # print(df)

    except RuntimeError:
        print("\nPandas not installed; skipping DataFrame example.")
    
    return df

