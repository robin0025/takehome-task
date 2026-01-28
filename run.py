# -*- coding: utf-8 -*-
"""
@author: Jonathan E. Robinson

run.py - Set of functions to run data cleaning pipeline ('clean_text.py') 
            - allows for testing without parallel processing
            - run different types of chunking and parallelisation to suit system
            - run.py requires analysis files to be placed in 'data' directory 
            - run.py output will be place in new or existing 'result' directory

Functions:
    run_file(file, inp_path, out_path, test, inp_chunksize)
    
"""


import pandas as pd
import multiprocessing as mp
import os
import getpass
import numpy as np
import platform
from datetime import datetime, date

# get base location for relatives
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Own scripts
import src.read_jsonl as rj
from src.clean_text import Cleaner


### !!! Useful sub functions ###
# Function to make topline report data
def _audit_dict(cleaner):
    audit_log = {}
    # Add items to audit report
    audit_log.update({'USER':getpass.getuser()})
    audit_log.update({'PLATFORM':platform.platform()})
    audit_log.update({'TIME_STARTED':str(datetime.now())})
    audit_log.update({'CONFIG_PARAMETERS':cleaner.cfg})
    
    return audit_log
 
# Makes an output directory
def _mk_out(out_name='result',output="DATA", suf=False):
   if suf:
       out_name = out_name+date.today().strftime("%Y%m%d")
   if os.path.isdir(out_name):
       print('OUTPUT %s WILL BE PLACED IN PREXISTING FOLDER: /%s'%(output,out_name))
   else:
       os.mkdir(out_name)  
       print('OUTPUT %s WILL BE PLACED IN NEWLY CREATER FOLDER: /%s'%(output,out_name))  
   return out_name 
   
### !!! Main functions for running data ###
def run_file(file, inp_path, out_path, rep_path='report', test=False, inp_chunksize = 100):
    """
    Main analysis loop for data pipeline
    
    Parameters
    ----------
    file
        file name string
    inp_path
        path for data input
    out_path
        path for data outputs
    test
        logical for testing
    inp_chunksize
        integer for size of chunks data should be broken into
    """
    # Ensure correct directory
    os.chdir(_thisDir)
    os.chdir(inp_path)
    
    # Initiate Cleaner
    proc = Cleaner()
    
    # Prepare a df for text and all reasons for changes  
    text_df = pd.DataFrame(columns=['text'])
    report_df = pd.DataFrame([str({item:_audit_dict(proc)[item]}) for item in _audit_dict(proc)] , columns=['report'])
    # test=True To run testing on individual level
    if test:

        # Read Data
        data = rj.read(file)
        Text = data['text']
        
        # Go to output directory ready to save
        os.chdir(_thisDir)
        os.chdir(out_path)
        # Warn users whats happening
        if len(Text) > 100:
            print("WARNING: Running data processing in serial. There are %d lines of text to process. We recommend chunked parallel pool for full scale processing."%(len(Text)))
        else:
            print("WARNING: Running data processing in serial.")
            
        # Iterate through strings
        for ind, item in enumerate(Text):
            # Run analysis an inform user
            results = []
            results = proc.clean(item)
            print("Text Item "+ str(ind+1).zfill(4)+" processed.")
            print(results[0])
            # Place iteams in data frames 
            tmp_pd = pd.DataFrame([results[0]], columns=['text'])
            text_df = pd.concat([text_df, tmp_pd], ignore_index=True)
            tmp_pd = pd.DataFrame([str({result:results[1][result]}) for result in results[1]], columns=['report'])
            report_df = pd.concat([report_df, tmp_pd], ignore_index=True)
            
        ### Save out data
        rj.save_working(file, text_df, "_text")
        rj.save_working(file, report_df, "_report")
    
    # test=False to run Chunked data for full run
    else:
        ### For real runs
        workers  = mp.cpu_count()-2 # Workers available for processing (CPU cores on single machine)
        # Chunked parrallel pooling runs if chunk size is defined
        if inp_chunksize:
            #worker_chunksize = int(np.ceil(inp_chunksize/(workers)))
            # Chunk Data
            chunks = rj.read(file,inp_chunksize)
            # Go to output directory ready to save
            os.chdir(_thisDir)
            os.chdir(out_path)
            os.chdir(rep_path)
            print("Initialising chunked parallel pool with %d workers"%workers)
            print("Workers receive %d lines per run."%(inp_chunksize))
            #print("Workers receive %d per run, divided over the workers, each processing %d text lines at a time."%(inp_chunksize,worker_chunksize))
            # Iterate through Chunks
            for ind, chunk in enumerate(chunks):
                chText = chunk['text']
                # Run parallel pooling
                results = []
                with mp.Pool(workers) as pool:
                    results = pool.map(proc.clean,chText) 
                    #results = pool.map(proc.clean,chText,chunksize=worker_chunksize) 
                print("\nChunk "+ str(ind+1).zfill(4)+" processed.\n")
                # Place items in data frames 
                tmp_pd = pd.DataFrame([result[0] for result in results], columns=['text'])
                text_df = pd.concat([text_df, tmp_pd], ignore_index=True)
                tmp_pd = pd.DataFrame([str({item:result[1][item]}) for result in results for item in result[1]], columns=['report'])
                report_df = pd.concat([report_df, tmp_pd], ignore_index=True)

                ### Save out report for each chunk
                rj.save_working(file, report_df,"_ch" + str(ind+1).zfill(4)+"_report")
                report_df = pd.DataFrame(columns=['report'])
                
                ### Re-initiate empty cleaner
                proc = Cleaner()
            
            # ### Save out data
            os.chdir(_thisDir)
            os.chdir(out_path)
            rj.save_working(file, text_df, "_text")
        
        # Run parallel processing but not chunking
        else:
            # Read file and put in variable for analysis
            df = rj.read(file)
            # Go to output directory ready to save
            os.chdir(_thisDir)
            os.chdir(out_path)
            
            allText = df['text']
            # Prepare a df for text and all reasons for changes
            text_df = pd.DataFrame(columns=['text'])
            report_df = pd.DataFrame(columns=['report'])

            # run parallel pooling
            results = []
            print("Initialising chunked parallel pool with %d workers"%workers)
            print("Workers receive all %d to process at once."%(len(df)))
            with mp.Pool(processes= workers, initializer=worker_init) as pool:
                results = pool.map(proc.clean,allText) 
            
            # Place iteams in data frames 
            tmp_pd = pd.DataFrame([result[0] for result in results], columns=['text'])
            text_df = pd.concat([text_df, tmp_pd], ignore_index=True)
            tmp_pd = pd.DataFrame([str({item:result[1][item]}) for result in results for item in result[1]], columns=['report'])
            report_df = pd.concat([report_df, tmp_pd], ignore_index=True)
            
            ### Save out data
            rj.save_working(file, text_df, "_text")
            rj.save_working(file, report_df, "_report")

         

### !!! DATA PROCESSING CALLS ###
testIndividual = False
if __name__ == '__main__':
    # reset base location
    os.chdir(_thisDir)
    # input file location
    data_path = 'data'
    # output file location
    output_path = 'result'
    report_path = 'report'
    output_path = _mk_out(output_path) #make sure one if creater
    # set extension
    ext = '.jsonl'
    # get all files if it exists
    try:
        dirs = os.listdir(data_path)
    except:
        raise Exception("FOLDER ERROR: THE DEFINE FILE FOR INPUT DATA: '/%s' DOES NOT EXIST IN PACKAGE DIRECTORY"%data_path)
    os.chdir(output_path)
    report_path = _mk_out(report_path,'REPORT', True) #make sure one if creater
    # Cut down to extension only
    files = [file for file in dirs if ext in file]
    if files:
        for file in dirs:
             run_file(file, data_path, output_path, report_path, testIndividual, 100)
    else:
        raise Exception("DATA ERROR: NO FILES FOUND FOR PROCESSING. DATAFILE MUST BE IN '/%s' DIRECTORY"%data_path)
