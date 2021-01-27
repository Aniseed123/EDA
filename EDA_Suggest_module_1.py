# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:07:08 2019

@author: 510571
"""

import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  
from sklearn import svm
import matplotlib.pyplot as plot
import pandas
import numpy as np
import copy
from copy import deepcopy
import seaborn as sns

from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from keras.callbacks import History 

from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from functools import partial
import pickle

class suggestions():
    
    def __init__(self,dataset):
        self.dataset = dataset
        #self.width = 2
        #self.height = 5
        #self.scaler = StandardScaler()
        self.colname = self.dataset.columns.values
        self.dup_dataset = copy.copy(self.dataset)
        self.ticks = [' ' for self.i in range(len(self.dataset.columns.values))]
        self.tickmark = [0 for self.i in range(len(self.dataset.columns.values))]
        self.alpha_data_process()
    
    def alpha_data_process(self):
        
        for i in range(len(self.dataset.columns.values)):
            if( (self.dataset.iloc[:,i].dtype != np.int64) and (self.dataset.iloc[:,i].dtype != np.float64)):
                    self.ticks[i] = np.unique(self.dataset.iloc[:,i])
                    self.tickmark[i] = 1
                    self.tempcol = pandas.factorize(self.dataset.iloc[:,i])
                    self.dataset.iloc[:,i] = self.tempcol[0]
        
    def myfunction(self,event):
                self.canvas.configure(scrollregion=self.canvas.bbox("all"),width=self.widh,height=self.heig)
        
    def scrollwindow(self,widh,heig):
        self.widh = widh
        self.heig = heig
        self.myframe=tk.Frame(self.scrollwin,width=100,height=100,bd=1)
        self.myframe.place(x=10,y=10)
        
        self.canvas=tk.Canvas(self.myframe)
        self.frame=tk.Frame(self.canvas)
        self.myscrollbar=tk.Scrollbar(self.myframe,orient="vertical",command=self.canvas.yview)
        self.myscrollbar2=tk.Scrollbar(self.myframe,orient="horizontal",command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.myscrollbar.set)
        self.canvas.configure(xscrollcommand=self.myscrollbar2.set)
        
        self.myscrollbar.pack(side="right",fill="y")
        self.myscrollbar2.pack(side="bottom",fill="x")
        self.canvas.pack(side="left")
        self.canvas.create_window((0,0),window=self.frame,anchor='nw')
        self.frame.bind("<Configure>",self.myfunction)
        
    def labeldecide(self):
        self.labeldecidewindow = tk.Toplevel()
        self.labeldecidewindow.geometry("700x500")
        self.labeldecidewindow.title('Labels')
        val = []        
        
        skew = self.dataset.skew()
        
        for i in range(len(self.dataset.columns.values)):
            val.append(len(self.dataset.groupby(self.dataset.iloc[:,i])))
        uniqval = pandas.Series(val, index = self.dataset.columns.values) 

        ranges = self.dataset.max() - self.dataset.min()
        
        data_cmpl = pandas.concat([skew,uniqval,ranges], axis = 1)

        filename_model = 'D:/ML_data/labelpred_trial3.sav'
        #filename_model = 'D:/Saved_models/85_0.2__1024_512_256_128_0.2_64_ep100_sur2.sav'
        loaded_model = pickle.load(open(filename_model, 'rb'))
        loaded_model.summary()
        
        x = data_cmpl
        #print(x)

        print(loaded_model.predict(x.values)*100,'%')
        self.percentval = loaded_model.predict(x.values)
        #self.percentval = self.percentval.reshape(-1,1)
        #print(type(self.percentval))
        #self.percentval = np.array(self.percentval)
        #print((self.percentval))
        #self.percentval = dict(zip(self.dataset.columns.names,self.percentval))
        self.percentval = pandas.Series(self.percentval.ravel(), index = self.dataset.columns.values)
        print(self.percentval)
        self.percentval = pandas.concat([pandas.Series(self.dataset.columns.values,index = self.dataset.columns.values),self.percentval], axis=1)
        print(self.percentval)
        #svcscore = float("{0:.3f}".format(svcscore))
        self.percentval.rename(columns={0: 'Columns', 1: 'Percentage'}, inplace=True)
        print(self.percentval)
        self.showarray()
    
    '''
     for i in range(len(self.dataset.columns.values)):
            self.percentval[i] = float("{0:.4f}".format(float(self.percentval[i])))*100
            #tk.Label(self.frame, text = self.DecideLabel(self.colname[i],i) +' : ', font=("Courier", 15)).grid(row = i+1, column = 0, sticky = 'w')
            tk.Label(self.frame, text = str(self.percentval[i]) + '%', font=("Courier", 15)).grid(row = i+1, column = 1, sticky = 'w')
        tk.Button(self.frame, text = 'Descending order').grid(row = 0, column = 2)
    
    '''
    def showarray(self):
        self.scrollwin = self.labeldecidewindow
        self.scrollwindow(650,450)
        tk.Label(self.frame, text = 'Column',font=("Courier", 15)).grid(row=0, column = 0, sticky = 'w')
        tk.Label(self.frame, text = 'P of being label', font=("Courier", 15)).grid(row = 0, column =1, sticky = 'w')
        for i in range(len(self.dataset.columns.values)):
            #self.percentval.iloc[i][1] = (self.percentval.iloc[i][1])*100
            tk.Label(self.frame, text = self.DecideLabel(self.percentval.iloc[i][0],i) +' : ', font=("Courier", 15)).grid(row = i+1, column = 0, sticky = 'w')
            tk.Label(self.frame, text = str(self.percentval.iloc[i][1]*100) + '%', font=("Courier", 15)).grid(row = i+1, column = 1, sticky = 'w')
        tk.Button(self.frame, text = 'Descending order', command = self.desorder).grid(row = 0, column = 2)
        
    def desorder(self):
        #list.sort(key=..., reverse=...)
        #np.sort(self.percentval)[::-1]
        '''
        import operator
        self.percentval = sorted(self.percentval.items(), key=operator.itemgetter(1), reverse = True)
        '''
        self.percentval = self.percentval.sort_values(by = ['Percentage'], ascending = False)
        print(self.percentval)
        self.corrdecide()
        self.showarray()
    
    def corrdecide(self):
        #self.corrdecidewindow = tk.Toplevel()
        #self.corrdecidewindow.geometry("600x500")
        #self.corrdecidewindow.title('Corelations')
        
        #self.scrollwin = self.corrdecidewindow
        #self.scrollwindow(550,450)
        
        corrs = self.dataset.corr(method='pearson')
        print(corrs)
        
        
    def is_number(self,testn):  #Useful in function DecideLabel. Tells if value passed is a number or a string
        #self.testn = self.name
        try:
            float(testn)   # Type-casting the string to `float`.
                                # If string is not a valid `float`, 
                                # it'll raise `ValueError` exception
        except ValueError:
            return testn
        return float(testn)
    
    def DecideLabel(self,name,i):       #Used in naming columns for selections in menu. If columns have string heading, displays string. If numeric value/ no proper column heading, returns 'Column n'
        #self.i = self.val
        #self.name = self.colname[self.i]
        tempname = self.is_number(name)
        if type(tempname) is str:
            return name
        elif type(tempname) is not str :
            return "column " + str(i+1)
        