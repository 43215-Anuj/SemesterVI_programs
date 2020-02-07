# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:18:05 2020

@author: anujk
"""

from pandas import read_csv
import pandas as pd

path = r"C:\Users\anujk\Documents\tennis.csv"
dataframe = read_csv(path)
yes=dataframe[dataframe['play']=='Yes']
no=dataframe[dataframe['play']=='No']

psunny_yes=len(yes[yes['outlook']=="Sunny"])/len(yes)
phot_yes = len(yes[yes['temp']=="Hot"])/len(yes)
phumidity_yes = len(yes[yes['humidity']=="High"])/len(yes)
pwind_yes = len(yes[yes['wind']=="Strong"])/len(yes)

psunny_no=len(no[no['outlook']=="Sunny"])/len(no)
phot_no = len(no[no['temp']=="Hot"])/len(no)
phumidity_no = len(no[no['humidity']=="High"])/len(no)
pwind_no = len(no[no['wind']=="Strong"])/len(no)

(psunny_yes + phot_yes + phumidity_yes + pwind_yes )