import pandas as pd
import numpy as np
"""
This file is responsible for handling data exploration and auditing. 
The benefit is that it will reduce all the "clutter" on the main jupyter notebook. 
This will help the reader engage with the content without looking at the excrutiating detail
"""


"""
change = 
{
same: Let the index be the original values of the dataset
fill: Let the index be the range


}
"""
# Data Cleaning Methods
"""
Input: Columns
Lower the cases in the string 
"""
def lower_columns(*data):
    frames = []
    for frame in range(len(data)):
        frames.append(data[frame])
        frames[frame].columns = map(str.lower,frames[frame].columns)
    return frames


"""
Return a sub-dataset pertaining to the given columns
"""
def select_columns(data,Label_Change,Columns = ['Q814Exp','Q58Val','head_sex','head_age','totmhinc','Q812Netincome']): # Select Variables For A Dataset
    ghs = data.copy()
    ghs = ghs[Columns] # Select The Chosen Variables
    for col,lbl in zip(ghs.columns,Label_Change):
        ghs.rename(columns={col:lbl},inplace=True)
        ghs
    return ghs

"""
Loop through all the list of datasets and select the given columns for each
"""
def prune_datasets(Label_Change,Columns,*ghs): # Return A List of Pruned Datasets
    prune_dataset = []
    for prune in range(len(ghs)):
        ghs_i = select_columns(ghs[prune],Label_Change,Columns)
        prune_dataset.append(ghs_i)
    return prune_dataset


"""
Select the index of the dataset
"""
def set_index(data_frame,index_name="ID",change="fill"): # This method handles indices
    data = data_frame.copy()
    data.rename(columns={index_name:index_name.lower()},inplace=True)
    if change == "fill":
        data[index_name.lower()] = range(len(data))
    data.set_index(index_name.lower(),inplace=True,verify_integrity=True)
    return data


def swap_columns(data,i,j):
    cols = list(data.columns.values)
    cols[i],cols[j] = cols[j],cols[i]
    return data[cols]

def display_records_per_year(data):
    pass
