# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:18:24 2019
Update:
        trial2_class:
            >12/3/19: Added fractorization of non-numeical data
            >13/3/19: Made DecideLabel and is_number independent of val (using local variables now)
            >Added experimental hotencoding function; not in use as of now
@author: 510571
"""
import tkinter as tk
import pandas
import numpy as np
import copy
from tkinter import filedialog
from tkinter.filedialog   import askopenfilename
from functools import partial

class analysis():
    
    def __init__(self,dataset):#constructor for class
        self.dataset = dataset
        self.scrollwin = 'NONE'
        self.widh = 0
        self.heig = 0
        self.dup_dataset = copy.copy(self.dataset)
        self.ticks = [' ' for self.i in range(len(self.dataset.columns.values))]
        self.tickmark = [0 for self.i in range(len(self.dataset.columns.values))]
        self.alpha_data_process()
        
        
    def alpha_data_process(self): #label encode categorical data and save this data to be used as names of axes, etc.
        
        
        for i in range(len(self.dataset.columns.values)):
            if( (self.dataset.iloc[:,i].dtype != np.int64) and (self.dataset.iloc[:,i].dtype != np.float64)):
                    self.ticks[i] = np.unique(self.dataset.iloc[:,i])
                    self.tickmark[i] = 1
                    self.tempcol = pandas.factorize(self.dataset.iloc[:,i])
                    self.dataset.iloc[:,i] = self.tempcol[0]
        #self.viewencodedvalues()
                    
    def hotencoding(self): #One-Hot encoding; experimental function; not in use yet
        
        for i in range(len(self.dataset.columns.values)):
            if( (self.dataset.iloc[:,i].dtype != np.number) or (self.dataset.iloc[:,i].dtype != np.floating)):
                    self.ticks[i] = np.unique(self.dataset.iloc[:,i])
                    self.tickmark[i] = 1
                    
                    #self.dataset.iloc[:,i] = pandas.get_dummies(self.dataset.iloc[:,i])
                    #self.dataset.iloc[:,i] = self.dataset.iloc[:,i].values
                    
                    self.dataset = pandas.concat([self.dataset,pandas.get_dummies(self.dataset.iloc[:,i])],axis=1)
                    print(self.dataset.shape)
                    print(type(self.dataset))
                    print(self.dataset.columns.values[i])
                    self.dataset.drop(self.dataset.columns.values[i], axis = 1, inplace = True)
                    print(self.dataset.shape)
                    print(type(self.dataset))
                
    def Dimension(self): #Function to give dimensions of the dataset (rows and columns)
        #nonlocal dataset
        print('Inside Dimension', self.dataset)
        self.dimwindow= tk.Toplevel()
        self.dimwindow.geometry("500x500")
        self.dimwindow.title('Dataset Dim')
        
        self.scrollwin = self.dimwindow
        self.scrollwindow(450,450)
        
        self.shape = self.dataset.shape 
        print(self.shape)
        self.T = tk.Label(self.frame, text = 'Rows: ' + str(self.shape[0]) + ' \n Cols: ' + str(self.shape[1]), font=("Courier", 15))
        self.T.grid(row=0, column=0)
        #tk.Button(self.frame,text = 'Save', command = partial(self.saveinfo,self.shape)).grid(row=0,column=0, sticky = 'w')
        
    def myfunction(self,event): #vital function in scrollwindow()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"),width=self.widh,height=self.heig)
                
    def Peek(self): #Function to see at most first 30 entries of the dataset 
        self.peekwindow= tk.Toplevel()
        self.peekwindow.geometry("700x700")
        self.peekwindow.title('Raw Data')
        self.peekwindow.title('Dataset Head')
        
        #if (self.tickmark[self.v.get()] == 1):
         #       plot.xticks(np.unique(self.x), self.ticks[self.v.get()])
        #peek = dataset.head(50)  
        print(self.dup_dataset)
        print(self.dataset)
        self.colname = self.dup_dataset.columns.values
        datasetshape = self.dup_dataset.shape
        self.scrollwin = self.peekwindow
        self.scrollwindow(650,650)
        
        for self.val in range(len(self.colname)):
            tk.Label(self.frame, text = 'Sr. Num', font=("Courier", 15)).grid(row = 0, column = 0)
            tk.Label(self.frame, text = '|'+self.DecideLabel(self.colname[self.val],self.val)+'|', font=("Courier", 15), bd=1).grid(row = 0, column = self.val+1)
        for k in range(min(30,datasetshape[0])):
            tk.Label(self.frame, text = k+1, font=("Courier", 15)).grid(row = k+1, column = 0)
            for m in range(len(self.colname)):
                tk.Label(self.frame, text = self.dup_dataset.iloc[k,m], font=("Courier", 15)).grid(row = k+1, column = m+1, sticky = 'w')
        
    
        
    def Datatype(self): #Function to see the datatype of columns
        self.typewindow= tk.Toplevel()
        self.typewindow.geometry("500x500")
        self.typewindow.title('Attribute Datatype')
        self.types = self.dup_dataset.dtypes
        print(self.types)
        self.scrollwin = self.typewindow
        self.scrollwindow(450,450)
        self.T = tk.Label(self.frame, text = self.types, font=("Courier", 15), anchor = 'nw')
        self.T.grid(row=1, column = 0)
        tk.Button(self.frame,text = 'Save', command = partial(self.saveinfo,self.types)).grid(row=0,column=0, sticky = 'w')
       
    def Descstats(self): #Descriptive statistics: count, mean, stanard deviation, min, 25th, 50th (aka median) and 75th percentile, max 
        self.descwindow= tk.Toplevel()
        self.descwindow.geometry("700x700")
        self.descwindow.title('Descriptive Statistics')
        self.description = self.dup_dataset.describe().transpose() #!!!Made it transpose
        
        self.scrollwin = self.descwindow
        self.scrollwindow(650,650)
        
        descshape = self.description.shape
        print(self.description)
        #for i in range(descshape[0]):
         #   tk.Label(self.frame, text = self.description.iloc[i],font=("Courier", 12)).grid(row=i+1, column =0)
        #partial(self.selectallboxes,self.col1)
        tk.Label(self.frame, text = self.description.iloc[np.arange(descshape[0])],font=("Courier", 12)).grid(row=1, column =0)
        '''
        self.T = tk.Label(self.frame, text = self.description,font=("Courier", 12))
        self.T.grid(row=1, column=0)
        '''
        tk.Button(self.frame,text = 'Save', command = partial(self.saveinfo,self.description)).grid(row=0,column=0, sticky = 'w')
        
    def saveinfo(self, data):
        self.filename =  filedialog.asksaveasfilename(title = "Save As",defaultextension= ".csv")
        with open(self.filename, 'w') as f:
                data.to_csv(f, header=True, sep = ',')
        
    def clickclassdist(self): #Function executed after button pressed in Classdist() function
            self.class2dwindow = tk.Toplevel()
            self.class2dwindow.geometry("500x500")
            self.class2dwindow.title('Class Distribution')
            
            #self.secondframe = 1
            self.scrollwin = self.class2dwindow
            self.scrollwindow(450,450)
            
            self.x = self.dup_dataset.iloc[:,self.v.get()]
            self.class_counts = self.dup_dataset.groupby(self.x).size()
            print(self.class_counts)   
            self.b1 = tk.Label(self.frame, text = 'Class' + str(self.class_counts),font=("Courier", 12) )
            self.b1.grid(row=1, column=0)
            tk.Button(self.frame, text = 'Save', command = partial(self.saveinfo,self.class_counts)).grid(row=0,column=0, sticky = 'w')
            
    def Classdist(self): #Distinct entities in a column
        self.classdwindow= tk.Toplevel()
        self.classdwindow.geometry("500x500")
        self.classdwindow.title('Class Distribution')
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        self.scrollwin = self.classdwindow
        self.scrollwindow(450,450)
        self.b1 = tk.Label(self.frame, text = 'Select column')
        self.b1.grid(row=0, column=0)
        self.v = tk.IntVar()
        self.v.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
                      text= self.DecideLabel(self.colname[self.val],self.val),
                      variable=self.v, 
                      value=self.val)
            self.lft.grid(row = self.val+1 ,column = 0, sticky = 'w',ipadx = 0, ipady =0)
        self.clickbutton = tk.Button(self.frame, text = 'Enter', command = self.clickclassdist)
        self.clickbutton.grid(row=0, column=1)
        
    def Correlation(self): #Givex correlation matrix
        self.corrwindow= tk.Toplevel()
        self.corrwindow.geometry("700x700")
        self.corrwindow.title('Correlation')
        self.correlations = self.dataset.corr(method='pearson')
        self.scrollwin = self.corrwindow
        self.scrollwindow(650,650)
        corshape = self.correlations.shape
        tk.Label(self.frame, text = self.correlations.iloc[np.arange(corshape[0])],font=("Courier", 12)).grid(row=1, column =0)
        print(self.correlations)
        tk.Button(self.frame,text = 'Save', command = partial(self.saveinfo,self.correlations)).grid(row=0,column=0, sticky = 'w')
        #self.T = tk.Label(self.frame, text = self.correlations, height=100, width=200,font=("Courier", 12), anchor = 'nw')
        #self.T.pack()
        
    def Skew(self): #Gives skew of the columns
        self.skewwindow= tk.Toplevel()
        self.skewwindow.geometry("500x500")
        self.skewwindow.title('Skew')
        self.skew = self.dataset.skew()
        self.scrollwin = self.skewwindow
        self.scrollwindow(450,450)
        skewshape = self.skew.shape
        print(self.skew)
        tk.Label(self.frame, text = self.skew.iloc[np.arange(skewshape[0])],font=("Courier", 12)).grid(row=1, column =0)
        tk.Button(self.frame, text = 'Save', command = partial(self.saveinfo,self.skew)).grid(row=0,column=0, sticky = 'w')
        #self.T = tk.Label(self.frame, text = self.skew, height=100, width=100, font=("Courier", 15), anchor = 'nw')
        #self.T.pack()
    
    def scrollwindow(self,widh,heig): #put scrollbars in any window of choice
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
    
    def viewencodedvalues(self): #Display Label encoded values in dataset if present
        print('INSIDE VIEWENCODEDVALUES FUNCTION')
        self.encodedvalueswindow = tk.Toplevel()
        self.encodedvalueswindow.geometry("700x500")
        self.encodedvalueswindow.title('Encoded Values')
        self.colname = self.dataset.columns.values
        
        self.scrollwin = self.encodedvalueswindow
        self.scrollwindow(650,450)
        
        for self.val in range(len(self.colname)):
            if(self.tickmark[self.val] == 1): 
                for self.k in range(len(np.unique(self.dataset.iloc[:,self.val]))):
                    print(self.ticks[self.val][self.k] , '=>' , np.unique(self.dataset.iloc[:,self.val])[self.k])       
                    tk.Label(self.frame, text = self.DecideLabel(self.colname[self.val],self.val)+ ' |', font=("Courier", 15) ).grid(row = 0, column = self.val, sticky = 'w',ipadx = 10)
                    tk.Label(self.frame, text = str(self.ticks[self.val][self.k]) + '=>' + str(np.unique(self.dataset.iloc[:,self.val])[self.k]), font=("Courier", 15) ).grid(row=self.k+1, column = self.val, sticky = 'w', ipadx = 10)
  
    def is_number(self,testn): #Useful in function DecideLabel. Tells if value passed is a number or a string
        #self.testn = self.name
        try:
            float(testn)   # Type-casting the string to `float`.
               # If string is not a valid `float`, 
               # it'll raise `ValueError` exception
        except ValueError:
            return testn
        return float(testn)
    
    def DecideLabel(self,name,i): #Used in naming columns for selections in menu. If columns have string heading, displays string. If numeric value/ no proper column heading, returns 'Column n'
        #self.i = self.val
        #self.i = self.val
        #self.name = self.colname[self.i]
        tempname = self.is_number(name)
        if type(tempname) is str:
            return name
        elif type(tempname) is not str :
            return "column " + str(i+1)
'''    
if __name__ == "__main__": #Main function to call the class
    #main()
    tes = analysis() #(x,y,z) = (analysis, visualization, algorithm)
           #Calling the menu from the classx
    #cols = x.getcolvalues()
    #print(cols)
'''