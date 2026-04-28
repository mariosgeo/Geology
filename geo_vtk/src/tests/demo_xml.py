# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:08:21 2020

@author: karaouli
"""

import xml.etree.ElementTree as et 
from vtkclass import VtkClass
import numpy as np
import glob
import pandas as pd
# Intiallize our class
int1=VtkClass()





xtree = et.parse("..\\data\\arcgis\\noordzee_vak_S7.xml")


doc_df = pd.DataFrame(list(iter_docs(xtree.getroot())))


xroot = xtree.getroot() 


number_of_boreholes=len(xroot)
number_of_attributes=len(xroot[0])

for elem in xroot:
    for subelem in elem:
        for subelem2 in subelem:
            for subelem3 in subelem2:
                print(subelem3.attrib)




def parse_XML(xml_file, df_cols): 
    """Parse the input XML file and store the result in a pandas 
    DataFrame with the given columns. 
    
    The first element of df_cols is supposed to be the identifier 
    variable, which is an attribute of each node element in the 
    XML data; other features will be parsed from the text content 
    of each sub-element. 
    """
    
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    rows = []
    
    for node in xroot: 
        res = []
        res.append(node.attrib.get(df_cols[0]))
        for el in df_cols[1:]: 
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else: 
                res.append(None)
        rows.append({df_cols[i]: res[i] 
                     for i, _ in enumerate(df_cols)})
    
    out_df = pd.DataFrame(rows, columns=df_cols)
        
    return out_df






df=parse_XML('..\\data\\arcgis\\noordzee_vak_S7.xml',[""])