# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:30:17 2019

@author: 510571
Update:     trial2:
            >Used nonlocal instead of clobal for col, coltemp, colnum, columnlist, opts
            trial3:
            >using classes
            >13/3/19: Made DecideLabel and is_number independent of val (using local variables now)
            >15/2/19: Added Kmeans clustering
            trial4:
            >Multivariable LinRegr with added model evaluations
            trial5:
            >22/3: Added dropout layer option in nnlist 
            >nnfunc shows both test and train accuracy/loss
            >Lower limit set to minuscol and minusrow for nnlist and nngrid
            >Epoch entry box added inside nnlistwindow and nngridwindow
            >Attempted to save selected values before minus or plus cols in nnlist
            trial7:
            >Linregr made independent variable multivariable
            >Made Load Model work with NN, LinRegr and SVM models
            trial8:
            >Added polynomial regression to lingrgr
            >figure out using plot.plot instead of plot.scatter for clean graphs
"""

import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog   import askopenfilename
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
import itertools

from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
from keras.callbacks import History 

from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures #For polynomial regression

from functools import partial #To pass paramenters in functons via command parameter in Tkinter buttons
import pickle #saving and loading ML models

class algorithm():
    
    def __init__(self,dataset): #constructor for the class
        self.dataset = dataset
        self.width = 2 #For NN
        self.height = 5 #For NN
        self.scaler = StandardScaler() #For normalizing values
        self.colname = self.dataset.columns.values
        self.dup_dataset = copy.copy(self.dataset) #Copy of dataset
        self.ticks = [' ' for self.i in range(len(self.dataset.columns.values))] #x and y ticks in graphs
        self.tickmark = [0 for self.i in range(len(self.dataset.columns.values))] #content of x and y ticks in graphs
        self.alpha_data_process() #To process categorical data
    
    def alpha_data_process(self): #For categorical/non-numeric data
        
        for i in range(len(self.dataset.columns.values)):#Finding which column is non-numeric and saving their labels to plot on graphs
            if( (self.dataset.iloc[:,i].dtype != np.int64) and (self.dataset.iloc[:,i].dtype != np.float64)):
                    self.ticks[i] = np.unique(self.dataset.iloc[:,i])
                    self.tickmark[i] = 1
                    self.tempcol = pandas.factorize(self.dataset.iloc[:,i])
                    self.dataset.iloc[:,i] = self.tempcol[0]
        
    def myfunction(self,event): #used in scroll function
                self.canvas.configure(scrollregion=self.canvas.bbox("all"),width=self.widh,height=self.heig)
        
    def scrollwindow(self,widh,heig): #creates a scroll window for desired TK frame
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
        
    def selectallboxes(self,col):   #To select all columns at once in OpenFileSelectiven for column 1
        lft = {}
       #print(col)
        for val in range(self.shape[1]):
            col[val] = tk.IntVar()
            lft[val] = tk.Checkbutton(self.frame, text = self.DecideLabel(self.colname[val],val), variable = col[val])
            lft[val].select()
            lft[val].grid(row=val+2, column=0, sticky = 'w')
            
    def selectallboxes2(self,col):   #To select all columns at once in OpenFileSelective for column 2
        lft = {}
        #print(col)
        for val in range(self.shape[1]):
            col[val] = tk.IntVar()
            lft[val] = tk.Checkbutton(self.frame, text = self.DecideLabel(self.colname[val],val), variable = col[val])
            lft[val].select()
            lft[val].grid(row=val+2, column=1, sticky = 'w')
            
    def LinRegr(self): #Regression function
        
        self.linregrwindow = tk.Toplevel()
        self.linregrwindow.geometry("800x500")
        self.linregrwindow.title('Linear Regression')
        #self.colname = self.dataset.columns.values
        self.col1 = {}
        self.col2 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.linregrwindow #making scrollbars on the window
        self.scrollwindow(750,450)
        
        self.w1 = tk.Scale(self.frame, from_=0, to=1, orient= 'horizontal', length = 200, label = 'Select Test:Training ratio', resolution = 0.05)
        self.w1.set(0.3)
        self.w1.grid(row = 0, column = 3, sticky = 'w',ipadx = 5)
        
    
        self.b1 = tk.Label(self.frame, text = 'Independent variable(s)')
        self.b2 = tk.Label(self.frame, text = 'Dependent Variable(s)') 
        self.b1.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        #button = Tk.Button(master=frame, text='press', command=partial(action, arg)
        tk.Checkbutton(self.frame, text = 'Select All',command = partial(self.selectallboxes,self.col1)).grid(row=1,column=0, sticky = 'w')
        self.ck = tk.IntVar()
        self.ck.set(0)
        tk.Checkbutton(self.frame, text = 'Select All',command = partial(self.selectallboxes2,self.col2)).grid(row=1,column=1, sticky = 'w')
        #self.ck1 = tk.Checkbutton(self.frame, text= 'Normalize values', variable = self.ck)
        #self.ck1.grid(row=1, column = 3)
        
        '''
        self.v1 = tk.IntVar()
        #self.v2 = tk.IntVar()
        self.v1.set(0)
        #self.v2.set(0)
        for self.i in range(self.shape[1]):
            self.col2[self.i] = [self.colname[self.i],self.i]
        
        for self.val, self.col2 in enumerate(self.col2):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+2 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
        '''
            #self.rgt = tk.Radiobutton(self.frame, 
             #     text= self.DecideLabel(self.colname[self.val],self.val),
              #    variable=self.v2, 
               #   value=self.val)
            #self.rgt.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
        
        for self.val in range(self.shape[1]): #Selecting options with checkbuttons
            self.col1[self.val] = tk.IntVar()
            self.lft = tk.Checkbutton(self.frame, text = self.DecideLabel(self.colname[self.val],self.val), variable = self.col1[self.val])
            self.lft.grid(row=self.val+2, column=0, sticky = 'w')
            
        for self.val in range(self.shape[1]): #Selecting options with checkbuttons
            self.col2[self.val] = tk.IntVar()
            self.lft = tk.Checkbutton(self.frame, text = self.DecideLabel(self.colname[self.val],self.val), variable = self.col2[self.val])
            self.lft.grid(row=self.val+2, column=1, sticky = 'w')
            
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clicklinregr)
        self.clickbutton.grid(row=0, column=2, ipadx = 0)
        tk.Button(self.frame, text = 'Save model', command = self.savemodel_linregr).grid(row =1, column =2)
        
        self.regtype = tk.IntVar() #Choosing regression type
        self.regtype.set(0)
        self.n = tk.StringVar()
        self.n.set('2')
        tk.Label(self.frame, text = 'Regression Type:').grid(row = 2, column = 2, sticky = 'w')
        tk.Radiobutton(self.frame, text = 'Linear', variable = self.regtype, value = 0).grid(row = 3, column = 2, sticky = 'w')
        tk.Radiobutton(self.frame, text = 'Polynomial', variable = self.regtype, value = 1).grid(row = 4, column = 2, sticky = 'w')
        tk.Label(self.frame, text="Enter polynomial degree:").grid(row = 5, column = 2)
        tk.Entry(self.frame, textvariable = self.n).grid(row = 6, column = 2)
        
    def clicklinregr(self): #After clicking 'Plot' button
            self.key_save1 = []
            self.key_save2 = []
            for self.key, self.value in self.col1.items(): #Finding 'checked' or ticked options
                if self.value.get() > 0:
                    #print('selected column Y axis',key+1)
                    self.key_save1.append(self.key)
            
            for self.key, self.value in self.col2.items():
                if self.value.get() > 0:
                    #print('selected column Y axis',key+1)
                    self.key_save2.append(self.key)
                    
            self.depvar = self.dataset.iloc[:,self.key_save1]
            
            self.indepvar = self.dataset.iloc[:,self.key_save2]
            print(self.key_save2)
            self.depTrain, self.depTest, self.indepTrain, self.indepTest  = train_test_split(self.depvar,self.indepvar,test_size = self.w1.get(), random_state = 0)
            '''
            if(self.ck.get() == 1):
                self.scaler.fit(self.indepTrain)
                self.indepTrain = self.scaler.transform(self.indepTrain)
                self.scaler.fit(self.indepTest)
                self.indepTest = self.scaler.transform(self.indepTest)
           '''
           
           #mean_absolute_error(y_true, y_pred)
            if(self.regtype.get() is 0): #If linear regression
                if(len(self.key_save1) is 1 and len(self.key_save2) is 1): #For univariate analysis - draw a graph
                    self.indepTrain = self.indepTrain.values.reshape(-1,1) #Reshaped required
                    self.depTrain = self.depTrain.values.reshape(-1,1)
                    self.indepTest = self.indepTest.values.reshape(-1,1)
                    self.depTest = self.depTest.values.reshape(-1,1)
                    
                    self.scaler.fit(self.indepTrain) #normalize values
                    self.indepTrain = self.scaler.transform(self.indepTrain)
                    self.scaler.fit(self.depTrain)
                    self.depTrain = self.scaler.transform(self.depTrain)
                    self.scaler.fit(self.indepTest)
                    self.indepTest = self.scaler.transform(self.indepTest)
                    self.scaler.fit(self.depTest)
                    self.depTest = self.scaler.transform(self.depTest)
                    
                    
                    self.linearRegressor = LinearRegression()
                    self.linearRegressor.fit(self.depTrain,self.indepTrain)
                    
                    #print('printing indepTrain',self.indepTrain)
                    #print('printing depTrain',self.depTrain)
                    
                    plot.figure(figsize = (10,10)) #Plot graph
                    plot.subplot(221)
                    #if (self.tickmark[self.v2.get()] == 1):
                     #   plot.xticks(np.unique(self.x), self.ticks[self.v2.get()])
                    if (self.tickmark[self.key_save2[0]] == 1):
                        plot.xticks(np.unique(self.indepvar), self.ticks[self.key_save2[0]])
                    plot.scatter(self.depTrain, self.indepTrain, color = 'red')
                    plot.plot(self.depTrain, self.linearRegressor.predict(self.depTrain), color = 'blue')
                    plot.title('Training set')
                    plot.xlabel(self.colname[self.key_save1[0]])
                    plot.ylabel(self.colname[self.key_save2[0]])
                    #plot.show()
                    plot.subplot(222)
                    if (self.tickmark[self.key_save2[0]] == 1):
                        print(np.unique(self.indepvar), self.ticks[self.key_save2[0]])
                        plot.xticks(np.unique(self.indepvar), self.ticks[self.key_save2[0]])
                    plot.scatter(self.depTest,self.indepTest, color = 'red')
                    plot.plot(self.depTrain, self.linearRegressor.predict(self.depTrain), color = 'blue')
                    plot.title('Test set')
                    plot.xlabel(self.colname[self.key_save1[0]])
                    plot.ylabel(self.colname[self.key_save2[0]])
                    #plot.xlabel(self.colname[self.v2.get()])
                    #plot.ylabel(self.colname[self.v1.get()])
                    plot.show()
                    mae = mean_absolute_error(self.indepTest, self.linearRegressor.predict(self.depTest))
                    mae = float("{0:.3f}".format(mae))
                    print('MAE', mae)
                    mse = mean_squared_error(self.indepTest, self.linearRegressor.predict(self.depTest))
                    mse = float("{0:.3f}".format(mse))
                    print('MSE', mse)
                    r2score = r2_score(self.indepTest, self.linearRegressor.predict(self.depTest))
                    r2score = float("{0:.3f}".format(r2score))
                    print('R2 score',r2score)
                    tk.Label(self.frame, text = 'Mean Absolute Error: ' + str(mae)).grid(row = 1, column = 3, sticky = 'w')
                    tk.Label(self.frame, text = 'Mean Square Error: ' + str(mse)).grid(row = 2, column = 3, sticky = 'w')
                    tk.Label(self.frame, text = 'R2 score: ' + str(r2score)).grid(row = 3, column = 3, sticky = 'w')
                else: #multivariate analysis
                    print('multiple var')
                    self.linearRegressor = LinearRegression()
                    self.linearRegressor.fit(self.depTrain,self.indepTrain)
                    #print('MAE', mean_absolute_error(self.indepTest, self.linearRegressor.predict(self.depTest)))
                    #print('MSE',mean_squared_error(self.indepTest, self.linearRegressor.predict(self.depTest)))
                    #print('R2 score',r2_score(self.indepTest, self.linearRegressor.predict(self.depTest)))
                    
                    mae = mean_absolute_error(self.indepTest, self.linearRegressor.predict(self.depTest))
                    mae = float("{0:.3f}".format(mae))
                    print('MAE', mae)
                    mse = mean_squared_error(self.indepTest, self.linearRegressor.predict(self.depTest))
                    mse = float("{0:.3f}".format(mse))
                    print('MSE', mse)
                    r2score = r2_score(self.indepTest, self.linearRegressor.predict(self.depTest))
                    r2score = float("{0:.3f}".format(r2score))
                    print('R2 score',r2score)
                    tk.Label(self.frame, text = 'Mean Absolute Error: ' + str(mae)).grid(row = 1, column = 3, sticky = 'w')
                    tk.Label(self.frame, text = 'Mean Square Error: ' + str(mse)).grid(row = 2, column = 3, sticky = 'w')
                    tk.Label(self.frame, text = 'R2 score: ' + str(r2score)).grid(row = 3, column = 3, sticky = 'w')
            
            elif(self.regtype.get() is 1): #If polynomial regression
                print('polynomial')
                if(len(self.key_save1) is 1 and len(self.key_save2) is 1): #univariate analysis - plot graph
                    self.scaler.fit(self.indepTrain)
                    self.indepTrain = self.scaler.transform(self.indepTrain)
                    self.scaler.fit(self.depTrain)
                    self.depTrain = self.scaler.transform(self.depTrain)
                    self.scaler.fit(self.indepTest)
                    self.indepTest = self.scaler.transform(self.indepTest)
                    self.scaler.fit(self.depTest)
                    self.depTest = self.scaler.transform(self.depTest)
                    
                    poly = PolynomialFeatures(degree= int(self.n.get()))
                    X_ = poly.fit_transform(self.depTrain)
                    self.linearRegressor = LinearRegression()
                    self.linearRegressor.fit(X_, self.indepTrain)
                    
                    mae = mean_absolute_error(self.indepTest, self.linearRegressor.predict(poly.fit_transform(self.depTest)))
                    mae = float("{0:.3f}".format(mae))
                    print('MAE', mae)
                    mse = mean_squared_error(self.indepTest, self.linearRegressor.predict(poly.fit_transform(self.depTest)))
                    mse = float("{0:.3f}".format(mse))
                    print('MSE', mse)
                    r2score = r2_score(self.indepTest, self.linearRegressor.predict(poly.fit_transform(self.depTest)))
                    r2score = float("{0:.3f}".format(r2score))
                    print('R2 score',r2score)
                    tk.Label(self.frame, text = 'Mean Absolute Error: ' + str(mae)+'   ').grid(row = 1, column = 3, sticky = 'w')
                    tk.Label(self.frame, text = 'Mean Square Error: ' + str(mse)+'   ').grid(row = 2, column = 3, sticky = 'w')
                    tk.Label(self.frame, text = 'R2 score: ' + str(r2score)+'   ').grid(row = 3, column = 3, sticky = 'w')
                    
                    lists = sorted(zip(*[self.indepTrain, self.linearRegressor.predict(poly.fit_transform(self.indepTrain))]))
                    new_x, new_y = list(zip(*lists))

                    plot.figure(figsize = (10,10))
                    plot.subplot(221)
                    if (self.tickmark[self.key_save2[0]] == 1):
                        plot.xticks(np.unique(self.depvar), self.ticks[self.key_save2[0]])
                    plot.scatter(self.depTrain,self.indepTrain, color = 'red')
                    plot.plot(new_y, new_x, color = 'blue')
                    plot.title('Training set')
                    plot.xlabel(self.colname[self.key_save1[0]])
                    plot.ylabel(self.colname[self.key_save2[0]])
                    #plot.show()
                    plot.subplot(222)
                    if (self.tickmark[self.key_save2[0]] == 1):
                        print(np.unique(self.depvar), self.ticks[self.key_save2[0]])
                        plot.xticks(np.unique(self.depvar), self.ticks[self.key_save2[0]])
                    plot.scatter(self.depTest,self.indepTest, color = 'red')
                    plot.plot(new_y, new_x, color = 'blue')
                    plot.title('Test set')
                    plot.xlabel(self.colname[self.key_save1[0]])
                    plot.ylabel(self.colname[self.key_save2[0]])
                    plot.show()
                else: #multivariate analysis
                    print('multiple var')
                    poly = PolynomialFeatures(degree= int(self.n.get()))
                    X_ = poly.fit_transform(self.depTrain)
                    self.linearRegressor = LinearRegression()
                    self.linearRegressor.fit(X_, self.indepTrain)
                    
                    mae = mean_absolute_error(self.indepTest, self.linearRegressor.predict(poly.fit_transform(self.depTest)))
                    mae = float("{0:.3f}".format(mae))
                    print('MAE', mae)
                    mse = mean_squared_error(self.indepTest, self.linearRegressor.predict(poly.fit_transform(self.depTest)))
                    mse = float("{0:.3f}".format(mse))
                    print('MSE', mse)
                    r2score = r2_score(self.indepTest, self.linearRegressor.predict(poly.fit_transform(self.depTest)))
                    r2score = float("{0:.3f}".format(r2score))
                    print('R2 score',r2score)
                    tk.Label(self.frame, text = 'Mean Absolute Error: ' + str(mae) +'   ').grid(row = 1, column = 3, sticky = 'w')
                    tk.Label(self.frame, text = 'Mean Square Error: ' + str(mse) +'   ').grid(row = 2, column = 3, sticky = 'w')
                    tk.Label(self.frame, text = 'R2 score: ' + str(r2score) +'   ').grid(row = 3, column = 3, sticky = 'w')

    def savemodel_linregr(self): #save regression model
        #import pickle
        self.filename =  filedialog.asksaveasfilename(title = "Save As", defaultextension=".sav")
        print (self.filename)
        pickle.dump(self.linearRegressor, open(self.filename, 'wb'))
            
    def svmfunc(self): #Support Vector Machine
        
        self.svmrwindow = tk.Toplevel()
        self.svmrwindow.geometry("800x500")
        self.svmrwindow.title('SVM')
        #self.colname = self.dataset.columns.values
        self.col1 = {}
        self.col2 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.svmrwindow #scroll window
        self.scrollwindow(750,450)
        
        self.w1 = tk.Scale(self.frame, from_=0.05, to=0.95, orient= 'horizontal', length = 200, label = 'Select Test:Training ratio', resolution = 0.05)
        self.w1.set(0.30)
        self.w1.grid(row = 0, column = 2, sticky = 'w',ipadx = 5)
        
    
        self.b1 = tk.Label(self.frame, text = 'Select attributes')
        self.b2 = tk.Label(self.frame, text = 'Select label') 
        self.b1.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        tk.Checkbutton(self.frame, text = 'Select All',command = partial(self.selectallboxes,self.col1)).grid(row=1,column=0, sticky = 'w')
        self.svmtype = tk.IntVar()
        self.svmtype.set(0)
        self.n = tk.StringVar()
        self.n.set('3')
        tk.Radiobutton(self.frame, text = 'SVC with linear kernel', variable = self.svmtype, value = 0).grid(row = 1, column = 2, sticky = 'w')
        tk.Radiobutton(self.frame, text = 'LinearSVC (linear kernel)', variable = self.svmtype, value = 1).grid(row = 2, column = 2, sticky = 'w')
        tk.Radiobutton(self.frame, text = 'SVC with RBF kernel', variable = self.svmtype, value = 2).grid(row = 3, column = 2, sticky = 'w')
        tk.Radiobutton(self.frame, text = 'SVC with polynomial kernel', variable = self.svmtype, value = 3).grid(row = 4, column = 2, sticky = 'w')
        tk.Label(self.frame, text="Enter polynomial degree:" ).grid(row = 5, column = 2)
        tk.Entry(self.frame, textvariable = self.n).grid(row = 6, column = 2)
        tk.Label(self.frame, text = 'Accuracy :' ).grid(row = 7, column = 2)
        tk.Label(self.frame, text = '0%' ).grid(row = 8, column = 2)
        
            
        for self.val in range(self.shape[1]):
            self.col1[self.val] = tk.IntVar()
            self.lft = tk.Checkbutton(self.frame, text = self.DecideLabel(self.colname[self.val],self.val), variable = self.col1[self.val])
            self.lft.grid(row=self.val+2, column=0, sticky = 'w')
        self.v2 = tk.IntVar()     
        self.v2.set(0)
        
        for self.i in range(self.shape[1]):
            self.col2[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col2 in enumerate(self.col2):
            self.rgt = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+2 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
    
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clicksvm)
        self.clickbutton.grid(row=0, column=3, ipadx = 0)
        tk.Button(self.frame, text = 'Save model', command = self.savemodel_svm).grid(row = 1, column = 3)
        
    def clicksvm(self): #After clicking the 'Plot' button
            #print (svmtype.get())
            self.key_save = []
            for self.key, self.value in self.col1.items():
                if self.value.get() > 0:
                    #print('selected column Y axis',key+1)
                    self.key_save.append(self.key)
    
            self.x = self.dataset.iloc[:,self.key_save]
            self.y = self.dataset.iloc[:,self.v2.get()]

            self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x,self.y,test_size = self.w1.get(), random_state = 0)

            
            if (self.svmtype.get() is 0): #Different funcions for different types of kernels
                self.svc = svm.SVC(kernel='linear', C=1.0).fit(self.xTrain, self.yTrain)
                self.kerntype = 'Linear'
            elif (self.svmtype.get() is 1):
                self.svc = svm.LinearSVC(C=1.0).fit(self.xTrain, self.yTrain)
                self.kerntype = 'LinearSVC'
            elif (self.svmtype.get() is 2):
                self.svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0).fit(self.xTrain, self.yTrain)
                self.kerntype = 'RBF'
            elif (self.svmtype.get() is 3):
                self.svc =  svm.SVC(kernel='poly', degree=int(self.n.get()), C=1.0).fit(self.xTrain, self.yTrain)
                self.kerntype = 'Polynomial (n = ' + self.n.get() + ' )'
                
            
            if(len(self.key_save) is 2): #To plot a graph if 2D data
                self.h = .02
                self.fig = plot.figure(figsize=(10, 10))
                self.ax = self.fig.add_subplot(111)
                self.x_min, self.x_max = self.xTest.iloc[:, 0].min() - 1, self.xTest.iloc[:, 0].max() + 1
                self.y_min, self.y_max = self.xTest.iloc[:, 1].min() - 1, self.xTest.iloc[:, 1].max() + 1
                self.xx, self.yy = np.meshgrid(np.arange(self.x_min, self.x_max, self.h),
                	            np.arange(self.y_min, self.y_max, self.h))
                self.Z = self.svc.predict(np.c_[self.xx.ravel(), self.yy.ravel()])
                self.Z = self.Z.reshape(self.xx.shape)
                plot.contourf(self.xx, self.yy, self.Z, cmap=plot.cm.coolwarm, alpha=0.8)
                plot.scatter(self.xTest.iloc[:, 0], self.xTest.iloc[:, 1], c=self.yTest, cmap=plot.cm.coolwarm,edgecolors='black')
                self.ax.set_xlabel(self.colname[self.key_save[0]])
                self.ax.set_ylabel(self.colname[self.key_save[1]])
                #plot.xlabel(colname[key_save[0]])
                #plot.ylabel(colname[key_save[1]])
            svcscore = self.svc.score(self.xTest,self.yTest)*100
            svcscore = float("{0:.3f}".format(svcscore))
            print('Accuracy for', self.kerntype, ':' )
            print(svcscore)
            #g1.destroy()
            #g2.destroy()
            tk.Label(self.frame, text = '        Accuracy for '+self.kerntype+':               ' ).grid(row = 7, column = 2)
            tk.Label(self.frame, text = ' '+str(svcscore)+'% ' ).grid(row = 8, column = 2)
            
    def savemodel_svm(self):  #saving SVM model
        #import pickle
        self.filename =  filedialog.asksaveasfilename(initialdir = "/",title = "Save As", defaultextension=".sav")
        print (self.filename)
        pickle.dump(self.svc, open(self.filename, 'wb'))
        
    def nnfunc(self): #Neural networks
        self.nnwindow = tk.Toplevel()
        self.nnwindow.geometry("800x500")
        self.nnwindow.title('Neural Networks')
        
        self.col1 = {}
        self.col2 = {}
        self.shape = self.dataset.shape
        self.opts = {}
        self.layertype = {}
        self.colnumlist = {}
        self.col = [[0 for self.xt in range(self.width)] for self.yt in range(self.height)]
        self.coltemp = [[0 for self.xt in range(self.width)] for self.yt in range(self.height)]
        self.colnum = [0 for self.xt in range(self.width)]
        #self.flag_nnlist = 0
        #self.flag_nngrid = 0
        self.losstype =' '
        
        self.scrollwin = self.nnwindow
        self.scrollwindow(750,450)
   
        self.b1 = tk.Label(self.frame, text = 'Select attributes')
        self.b2 = tk.Label(self.frame, text = 'Select label') 
        self.b1.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        tk.Checkbutton(self.frame, text = 'Select All',command = partial(self.selectallboxes,self.col1)).grid(row=1,column=0, sticky = 'w')
        self.b1 = tk.Button(self.frame, text = 'Create NN (grid)', command = self.clicknngrid)
        self.b1.grid(row=2, column=3, ipadx = 0)
        self.b2 = tk.Button(self.frame, text = 'Create NN (list)', command = self.clicknnlist)
        self.b2.grid(row=3, column=3, ipadx = 0)
        self.w1 = tk.Scale(self.frame, from_=0.05, to=0.95, orient= 'horizontal', length = 200, label = 'Select Test:Training ratio', resolution = 0.05)
        self.w1.set(0.30)
        self.w1.grid(row = 0, column = 3, sticky = 'w',ipadx = 5)
        self.w2 = tk.Scale(self.frame, from_=10, to=1000, orient= 'horizontal', length = 200, label = 'Select Epochs', resolution = 10)
        self.w2.set(100)
        self.w2.grid(row = 0, column = 4, sticky = 'w',ipadx = 5)
        
        for self.val in range(self.shape[1]):
            self.col1[self.val] = tk.IntVar()
            self.lft = tk.Checkbutton(self.frame, text = self.DecideLabel(self.colname[self.val],self.val), variable = self.col1[self.val])
            self.lft.grid(row=self.val+2, column=0, sticky = 'w')
        self.v2 = tk.IntVar()     
        self.v2.set(0)
        
        for self.i in range(self.shape[1]):
            self.col2[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col2 in enumerate(self.col2):
            self.rgt = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+2 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
    
    def clicknngrid(self): #For grid
            self.manualnndwindow = tk.Toplevel()
            self.manualnndwindow.geometry("500x500")
            self.manualnndwindow.title('Enter NN manually')
             
            #self.predictnngrid()
            self.scrollwin = self.manualnndwindow
            self.scrollwindow(450,450)
            self.gridcreatenngrid()
  
    def predictnngrid(self): #Running predction algorithms
                self.history = History()
                self.key_save = []
            
                for self.key, self.value in self.col1.items():
                    if self.value.get() > 0:
                        self.key_save.append(self.key)
                self.x = self.dataset.iloc[:,self.key_save]
                self.y = self.dataset.iloc[:,self.v2.get()]
                '''
                if( (self.y.dtype != np.int64) or (self.y.dtype != np.float64)):
                        self.y = pandas.factorize(self.y)
                        self.y = self.y[0]
                '''
                #print (x,y)
                self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x,self.y,test_size =self.w1.get(), random_state = 0)
                self.scaler.fit(self.xTrain)
                self.xTrain = self.scaler.transform(self.xTrain)
                self.scaler.fit(self.xTest)
                self.xTest = self.scaler.transform(self.xTest)
                self.yTrain=np_utils.to_categorical(self.yTrain,num_classes=len(np.unique(self.y)))
                self.yTest=np_utils.to_categorical(self.yTest,num_classes=len(np.unique(self.y)))
                print("Shape of y_train",self.yTrain.shape)
                print("Shape of y_test",self.yTest.shape)
                
                self.model=Sequential()
                self.model.add(Dense(self.colnum[0],input_dim=len(self.key_save),activation='relu'))
                for self.i in range (1,self.width-1):
                   self.model.add(Dense(self.colnum[self.i],activation='relu'))
                #for i in range (2):
                #    model.add(Dropout(0.2))
                self.model.add(Dense(len(np.unique(self.y)),activation='softmax'))
                self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
                self.model.summary()
                self.epochs=int(self.epochsnngrid.get())
                self.history = self.model.fit(self.xTrain,self.yTrain,validation_data=(self.xTest,self.yTest),batch_size=20,epochs=int(self.epochsnngrid.get()),verbose=1, callbacks=[self.history])
                
                self.prediction=self.model.predict(self.xTest)
                self.length=len(self.prediction)
                self.y_label=np.argmax(self.yTest,axis=1)
                self.predict_label=np.argmax(self.prediction,axis=1)
    
                self.accuracy=np.sum(self.y_label==self.predict_label)/self.length * 100 
                print("Accuracy of the dataset",self.accuracy )
                print(self.history.history.keys())
                
                self.val_loss_arr= self.history.history['val_loss']
                #print(self.history.history['val_loss'])
                self.loss_arr= self.history.history['loss']
                #print(self.history.history['val_loss'])
                self.val_acc_arr= self.history.history['val_acc']
                self.acc_arr= self.history.history['acc']
                
                self.f = plot.figure(figsize=(10,10))
                self.ax = self.f.add_subplot(2,1,1)
                self.ax2 = self.f.add_subplot(2,1,2)
                
                self.ax.set_title('Loss (Red: Train, Blue:Test)')
                for self.k in range(0,self.epochs):
                    self.ax.scatter(self.k, self.val_loss_arr[self.k], color = 'blue')
                    self.ax.scatter(self.k, self.loss_arr[self.k], color = 'red')
                #print(self.history.history['val_loss'])
                
                self.ax2.set_title('Accuracy (Red: Train, Blue:Test)')
                for self.k in range(0,self.epochs):
                    self.ax2.scatter(self.k, self.val_acc_arr[self.k], color = 'blue')
                    self.ax2.scatter(self.k, self.acc_arr[self.k], color = 'red')
            
    def gridcreatenngrid(self): #creating grid
                
                #print ('w=',width,'h=',height)
                self.col = [[0 for self.xt in range(self.width)] for self.yt in range(self.height)]
                self.coltemp = [[0 for self.xt2 in range(self.width)] for self.yt2 in range(self.height)]
                #SCROLL
                self.scrollwin = self.manualnndwindow
                self.scrollwindow(450,450)
                
                
                for self.i in range(self.height):
                    for self.k in range(self.width-1):
                        tk.Label(self.frame, text = 'Layer' + str(self.k+1)).grid(row = 0, column = self.k+1)
                        self.col[self.i][self.k] = tk.IntVar()
                        self.lft = tk.Checkbutton(self.frame, variable = self.col[self.i][self.k])
                        self.lft.grid(row=self.i+2, column=self.k+1, sticky = 'w')
                self.plusbuttoncol = tk.Button(self.frame, text = '+', command = self.Addcolmnngrid)
                self.plusbuttoncol.grid(row = 2, column = self.width+2)
                self.minusbuttoncol = tk.Button(self.frame, text = '-', command = self.Minuscolmnngrid)
                self.minusbuttoncol.grid(row = 2, column = self.width+3)
                self.plusbuttonr= tk.Button(self.frame, text = '+', command = self.Addronngrid)
                self.plusbuttonr.grid(row = self.height+2, column = 1)
                self.minusbuttonr= tk.Button(self.frame, text = '-', command =  self.Minusronngrid)
                self.minusbuttonr.grid(row = self.height+2, column = 2)
                tk.Button(self.frame, text = 'Enter', command = self.shownngrid).grid(row = self.height+2, column = 0)
                self.coltemp = list(map(list, zip(*self.coltemp)))
                #self.dup_x = self.x
                self.colnum = [0 for self.dup_x in range(self.width)]
                tk.Label(self.frame, text = 'Layer' + str(self.width)).grid(row = 0, column = self.width)
                for self.p in range (self.height):
                    self.bu1 = tk.Checkbutton(self.frame, state = 'disabled')
                    if(self.tickon(self.p) is 1):
                        self.bu1.select()
                    else:
                        self.bu1.deselect()
                    self.bu1.grid(row = self.p+2, column = self.width)
                    
                self.epochsnngrid = tk.StringVar()
                self.epochsnngrid.set(self.w2.get())
                tk.Label(self.frame, text = 'Epochs').grid(row=0, column=0)
                tk.Entry(self.frame, textvariable = self.epochsnngrid).grid(row = 1, column = 0)
        
                    
    def tickon(self,p): #To grey out last column
        self.y = self.dataset.iloc[:,self.v2.get()]
        #print(len(np.unique(self.y)), 'this?')
        if (self.p < len(np.unique(self.y))):
            return 1
        else:
            return 0
           
    def Addronngrid(self): #Function to increase gridsize after clicking '+' rows
              
        self.height = self.height+1
        self.frame.destroy()
        self.gridcreatenngrid()
            
    def Addcolmnngrid(self): #Function to increase gridsize after clicking '+' columns
        
        self.width = self.width+1
        self.frame.destroy()
        self.gridcreatenngrid()
    
    def Minuscolmnngrid(self): #Function to decrease gridsize after clicking '-' columns
        
        if(self.width > 2): #Limit how much the grid of list can decrease
            self.width = self.width-1
            self.frame.destroy()
            self.gridcreatenngrid()
        else:
            pass
        
    def Minusronngrid(self): #Function to decrease gridsize after clicking '-' rows
        
        if(self.height > len(np.unique(self.y)) ): 
            self.height = self.height-1
            self.frame.destroy()
            self.gridcreatenngrid()
        else:
            pass
        
    def shownngrid(self): #Calculate the NN shape from input to the grid
        self.coltemp = list(map(list, zip(*self.coltemp)))
        for self.i in range(self.height):
            for self.k in range(self.width-1):
                print(self.col[self.i][self.k].get(),' ',end="")
                self.coltemp[self.i][self.k] = self.col[self.i][self.k].get()
            print('\n')
        self.coltemp = list(map(list, zip(*self.coltemp))) #transpose
        
        self.nums1()
        self.predictnngrid()
        
    def nums1(self): #accessory function to showgrid
        for self.i in range(self.width):
            self.colnum[self.i] = self.coltemp[self.i].count(1)
    
    def shownnlist(self):          #Calculate the NN shape from input to the list     
        for self.i in range(self.height):            
           print(self.colnumlist[self.i].get(),' ',end="")
        self.predictnnlist()    
            
    def Addronnlist(self): #Function to increase list size after clicking '+' 
        #self.height_save = copy.copy(self.height)
        self.height = self.height+1
        #self.colnumlist_save = copy.copy(self.colnumlist)
        #self.layertype_save = copy.copy(self.layertype)
        #self.opts_save = copy.copy(self.opts)
        #self.flag_nnlist = 1
        self.frame.destroy()
        self.gridcreatennlist()
        
    def Minusronnlist(self): #Function to increase list size after clicking '-' 
        if(self.height > 2): #Limit how much the size of list can decrease
            #self.height_save = copy.copy(self.height)
            self.height = self.height-1
            #self.colnumlist_save = copy.copy(self.colnumlist)
            #self.layertype_save = copy.copy(self.layertype)
            #self.opts_save = copy.copy(self.opts)
            #self.flag_nnlist = 1
            self.frame.destroy()
            self.gridcreatennlist()
        else:
            pass
            
    def gridcreatennlist(self): #Function to create List type NN window
        self.y = self.dataset.iloc[:,self.v2.get()]
        #print ('w=',width,'h=',height)
        ######SCROLLNEEDED
        self.scrollwin = self.listnndwindow
        self.scrollwindow(650,450)
        tk.Label(self.frame, text = 'Layer Number').grid(row = 0, column = 0)
        tk.Label(self.frame, text = 'Neurons/Dropout ratio').grid(row = 0, column = 1)
        tk.Label(self.frame, text = 'Activation').grid(row = 0, column = 2)
        tk.Label(self.frame, text = 'Layer type').grid(row = 0, column = 3)
        for self.i in range(self.height-1):
            self.opts[self.i] = tk.StringVar()
            #self.opts[self.i].set('relu')
            self.layertype[self.i] = tk.StringVar()
            #self.layertype[self.i].set('Dense')
            self.colnumlist[self.i] = tk.StringVar()
            #self.colnumlist[self.i].set('10')
            '''
            if(self.flag_nnlist == 1):
                self.opts[self.i].set('relu')
                self.layertype[self.i].set('Dense')
                self.colnumlist[self.i].set('10')
                if(self.i < self.height_save):
                    self.opts[self.i].set(self.opts_save[self.i].get())
                    self.layertype[self.i].set(self.layertype_save[self.i].get())
                    self.colnumlist[self.i].set(self.colnumlist_save[self.i].get())
            '''
            #elif(self.flag_nnlist == 0):
            self.opts[self.i].set('relu')
            self.layertype[self.i].set('Dense')
            self.colnumlist[self.i].set('10')
                
            tk.OptionMenu(self.frame, self.opts[self.i], 'relu', 'tanh', 'sigmoid', 'softmax').grid(row = self.i+1, column = 2)
            if(self.i != 0):
                tk.OptionMenu(self.frame, self.layertype[self.i], 'Dense', 'Dropout').grid(row = self.i+1, column = 3)
            tk.Label(self.frame, text = 'Layer ' + str(self.i+1) ).grid(row = self.i+1, column = 0)
            tk.Entry(self.frame, textvariable = self.colnumlist[self.i]).grid(row = self.i+1, column = 1)
            self.plusbuttonr= tk.Button(self.frame, text = '+', command = self.Addronnlist)
            self.plusbuttonr.grid(row = self.height+2, column = 0)
            self.minusbuttonr= tk.Button(self.frame, text = '-', command =  self.Minusronnlist)
            self.minusbuttonr.grid(row = self.height+2, column = 1)
            tk.Button(self.frame, text = 'Enter', command = self.shownnlist).grid(row = 0, column = 4)
            
        self.colnumlist[self.height-1] = tk.StringVar()
        self.opts[self.height-1] = tk.StringVar()
        self.opts[self.height-1].set('softmax')
        self.colnumlist[self.height-1].set(len(np.unique(self.y)))
        tk.OptionMenu(self.frame, self.opts[self.height-1], 'softmax', 'sigmoid').grid(row = self.height, column = 2)
        tk.Label(self.frame, text = 'Layer ' + str(self.height) ).grid(row = self.height, column = 0)
        
        temp_text = tk.StringVar()
        temp_text.set(self.colnumlist[self.height-1].get()+' or 1')
        self.bu1 = tk.Entry(self.frame, state = 'disabled', textvariable = temp_text)
        self.bu1.grid(row = self.height, column =1)
        
        self.epochsnnlist = tk.StringVar()
        self.epochsnnlist.set(self.w2.get())
        tk.Label(self.frame, text = 'Epochs').grid(row=1, column=4)
        tk.Entry(self.frame, textvariable = self.epochsnnlist).grid(row = 1, column = 5)
        tk.Button(self.frame, text = 'Save model', command = self.savemodel).grid(row = 0, column = 5)
        
        self.losstype = tk.StringVar()
        self.losstype.set('categorical_crossentropy')
        tk.Label(self.frame, text = 'Loss Type:').grid(row = 2, column = 4)
        tk.OptionMenu(self.frame, self.losstype, 'categorical_crossentropy', 'binary_crossentropy').grid(row = 2, column = 5)
        #self.hot_enc = tk.IntVar()
        #tk.Checkbutton(self.frame, text = 'One-Hot-Encode label', variable = self.hot_enc).grid(row=3, column = 4)
    def predictnnlist(self): #Predicting with List type
        self.history = History()
        self.key_save = []
    
        for self.key, self.value in self.col1.items():
            if self.value.get() > 0:
                self.key_save.append(self.key)
        self.x = self.dataset.iloc[:,self.key_save]
        self.y = self.dataset.iloc[:,self.v2.get()]
        print('Y',self.y)
        '''
        if( (self.y.dtype != np.int64) or (self.y.dtype != np.float64)):
                self.y = pandas.factorize(self.y)
                self.y = self.y[0]
        '''
        #print (x,y)
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x,self.y,test_size =self.w1.get(), random_state = 0)
        self.scaler.fit(self.xTrain)
        self.xTrain = self.scaler.transform(self.xTrain)
        self.scaler.fit(self.xTest)
        self.xTest = self.scaler.transform(self.xTest)
        #print('click status', self.hot_enc.get(), type(self.hot_enc.get()))
        
        #if self.hot_enc.get() > 0 or self.losstype == 'categorical_crossentropy':
        print(self.losstype, type(self.losstype))
        if self.losstype.get() == 'categorical_crossentropy':
            #print('inside func')
            self.yTrain=np_utils.to_categorical(self.yTrain,num_classes=len(np.unique(self.y)))
            self.yTest=np_utils.to_categorical(self.yTest,num_classes=len(np.unique(self.y)))
            op_layer_shape = int(self.colnumlist[self.height-1].get())
        elif self.losstype.get() == 'binary_crossentropy':
            op_layer_shape = 1
            
        print("Shape of y_train",self.yTrain.shape)
        print("Shape of y_test",self.yTest.shape)
        self.model=Sequential()
        print(self.yTrain)
        print('input dim:',len(self.key_save))
        self.model.add(Dense(int(self.colnumlist[0].get()),input_dim=len(self.key_save),activation=self.opts[0].get()))
       
        for self.i in range (1,self.height-1):
            print(self.layertype[self.i].get())
            if(self.layertype[self.i].get() == 'Dense'):
                self.model.add(Dense(int(self.colnumlist[self.i].get()),activation=self.opts[self.i].get()))
            elif(self.layertype[self.i].get() == 'Dropout'):
                print(float(self.colnumlist[self.i].get()))
                self.model.add(Dropout(float(self.colnumlist[self.i].get())))
                
        self.model.add(Dense(op_layer_shape,activation=self.opts[self.height-1].get()))
        ##self.model.add(Dense(1,activation='sigmoid'))
        
        self.model.compile(loss=self.losstype.get(),optimizer='adam',metrics=['accuracy'])
        ##self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.summary()
        self.epochs=int(self.epochsnnlist.get())
        self.history = self.model.fit(self.xTrain,self.yTrain,validation_data=(self.xTest,self.yTest),epochs=int(self.epochsnnlist.get()),verbose=1, callbacks=[self.history])
        
        self.prediction=self.model.predict(self.xTest)
        self.length=len(self.prediction)
        self.y_label=np.argmax(self.yTest,axis=1)
        self.predict_label=np.argmax(self.prediction,axis=1)

        self.accuracy=np.sum(self.y_label==self.predict_label)/self.length * 100 
        print("Accuracy of the dataset",self.accuracy )
        print(self.history.history.keys())
        self.val_loss_arr= self.history.history['val_loss']
        #print(self.history.history['val_loss'])
        self.loss_arr= self.history.history['loss']
        #print(self.history.history['val_loss'])
        self.val_acc_arr= self.history.history['val_acc']
        self.acc_arr= self.history.history['acc']
        
        self.f = plot.figure(figsize=(10,10))
        self.ax = self.f.add_subplot(2,1,1)
        self.ax2 = self.f.add_subplot(2,1,2)
        
        self.ax.set_title('Loss (Red: Train, Blue:Test)')
        for self.k in range(0,self.epochs):
            self.ax.scatter(self.k, self.val_loss_arr[self.k], color = 'blue')
            self.ax.scatter(self.k, self.loss_arr[self.k], color = 'red')
        #print(self.history.history['val_loss'])
        
        self.ax2.set_title('Accuracy (Red: Train, Blue:Test)')
        for self.k in range(0,self.epochs):
            self.ax2.scatter(self.k, self.val_acc_arr[self.k], color = 'blue')
            self.ax2.scatter(self.k, self.acc_arr[self.k], color = 'red')  

    def clicknnlist(self): #Create List type NN
        #print ('Need to make')
        self.listnndwindow = tk.Toplevel()
        self.listnndwindow.geometry("700x500")
        self.listnndwindow.title('Enter NN manually')
        #################SCROLLNEEDED
        self.scrollwin = self.listnndwindow
        self.scrollwindow(650,450)
        
        self.gridcreatennlist()
            
    def savemodel(self): #Save NN model in .sav format
        self.filename =  filedialog.asksaveasfilename(initialdir = "/",title = "Save As", defaultextension=".sav")
        print (self.filename)
        pickle.dump(self.model, open(self.filename, 'wb'))
        
    def viewencodedvalues(self): #Display Label encoded values in dataset if present.
        #print('INSIDE VIEWENCODEDVALUES FUNCTION')
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
                    tk.Label(self.frame, text = str(self.ticks[self.val][self.k]) + '=>' + str(np.unique(self.dataset.iloc[:,self.val])[self.k]), font=("Courier", 15) ).grid( row=self.k+1, column = self.val, sticky = 'w', ipadx = 10)
    
    def kmeans(self): #Function to perform K means clustering
        
        self.kmeanswindow = tk.Toplevel()
        self.kmeanswindow.geometry("800x500")
        self.kmeanswindow.title('K means clustering')
        
        self.col1 = {}
        self.col2 = {}
        self.shape = self.dataset.shape
        #self.opts = {}
        self.colnumlist = {}
        self.col = [[0 for self.xt in range(self.width)] for self.yt in range(self.height)]
        self.coltemp = [[0 for self.xt in range(self.width)] for self.yt in range(self.height)]
        self.colnum = [0 for self.xt in range(self.width)]
        self.k = tk.StringVar()
        self.k.set('3')
        
        self.scrollwin = self.kmeanswindow
        self.scrollwindow(750,450)
        
        self.b1 = tk.Label(self.frame, text = 'Select attributes:')
        #self.b2 = tk.Label(self.frame, text = 'Select label') 
        self.b1.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        tk.Checkbutton(self.frame, text = 'Select All',command = partial(self.selectallboxes,self.col1)).grid(row=1,column=0, sticky = 'w')
        #self.b2.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b2 = tk.Button(self.frame, text = 'View Elbow graph', command = self.clickelbowgr)
        self.b2.grid(row=0, column=1, ipadx = 0)
        tk.Label(self.frame, text="Enter K:" ).grid(row = 0, column = 2)
        tk.Entry(self.frame, textvariable = self.k).grid(row = 0, column = 3)
        self.b3 = tk.Button(self.frame, text = 'Enter', command = self.clickkmeans)
        self.b3.grid(row=1, column=2, ipadx = 0)
        #self.w1 = tk.Scale(self.frame, from_=0.05, to=0.95, orient= 'horizontal', length = 200, label = 'Select Test:Training ratio', resolution = 0.05)
        #self.w1.set(0.30)
        #self.w1.grid(row = 0, column = 3, sticky = 'w',ipadx = 5)
        #self.w2 = tk.Scale(self.frame, from_=10, to=1000, orient= 'horizontal', length = 200, label = 'Select Epochs', resolution = 10)
        #self.w2.set(100)
        #self.w2.grid(row = 0, column = 4, sticky = 'w',ipadx = 5)
        
        for self.val in range(self.shape[1]):
            self.col1[self.val] = tk.IntVar()
            self.lft = tk.Checkbutton(self.frame, text = self.DecideLabel(self.colname[self.val],self.val), variable = self.col1[self.val])
            self.lft.grid(row=self.val+2, column=0, sticky = 'w')
        self.v2 = tk.IntVar()     
        self.v2.set(0)
        '''
        for self.i in range(self.shape[1]):
            self.col2[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col2 in enumerate(self.col2):
            self.rgt = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
        '''
    
    def clickelbowgr(self): #Function to get elbow graph from the button pressed in kmeans()
        wcss = []
        elbloc = []
        self.key_save = []
        for self.key, self.value in self.col1.items():
            if self.value.get() > 0:
                #print('selected column Y axis',key+1)
                self.key_save.append(self.key)
    
        self.x = self.dataset.iloc[:,self.key_save]
        #self.y = self.dataset.iloc[:,self.v2.get()]
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
            kmeans.fit(self.x)
            wcss.append(kmeans.inertia_)
            elbloc.append(i)
        #print(wcss,elbloc)
        
        try: #if library needs to install: pip install kneed
            from kneed import KneeLocator
            self.kn = KneeLocator(elbloc, wcss, curve='convex', direction='decreasing')
            print(self.kn.knee)
            tk.Label(self.frame, text = 'Elbow value (K): ' + str(self.kn.knee)).grid(row = 1, column = 1)
            self.k.set(str(self.kn.knee))
            print(self.k.get())
        except Exception as e:
            pass
        
        plot.figure(figsize = (10,10))
        plot.plot(range(1, 11), wcss)
        plot.title('The elbow method')
        plot.xlabel('Number of clusters')
        plot.ylabel('Sum of squared errors') #within cluster sum of squares
        plot.show()

    def clickkmeans(self): #Perform Kmeans after clicking button in Kmeans()
        self.key_save = []
        for self.key, self.value in self.col1.items():
            if self.value.get() > 0:
                #print('selected column Y axis',key+1)
                self.key_save.append(self.key)
                
        self.x = self.dataset.iloc[:,self.key_save]
        print(self.k.get())
        self.kmeansrows = self.x.shape[0]
        self.kmeanscolumns = self.x.shape[1]
        self.kmeansmean = np.mean(self.x, axis = 0)
        self.kmeansstd = np.std(self.x, axis = 0)
        #print('k',type(int(self.k.get())), type(self.k.get()))
        #print('cols',self.kmeanscolumns,type(self.kmeanscolumns))
        #print('std',self.kmeansstd,type(self.kmeansstd),self.kmeansstd.shape )
        #print('mean',self.kmeansmean,type(self.kmeansmean))
        self.kmeanscenters = np.random.randn(int(self.k.get()),self.kmeanscolumns)*self.kmeansstd.values + self.kmeansmean.values
        
        self.kmeanscenters_old = np.zeros(self.kmeanscenters.shape) # to store old centers
        self.kmeanscenters_new = deepcopy(self.kmeanscenters) # Store new centers
        self.kmeanserror = np.linalg.norm(self.kmeanscenters_new - self.kmeanscenters_old)
        self.kmeansclusters = np.zeros(self.kmeansrows)
        self.distances = np.zeros( (self.kmeansrows,int(self.k.get()) ) )
        kmeanspasses = 0
        #xplus = self.x.assign(op = pandas.Series(np.random.randn(self.kmeansrows)).values)
        while self.kmeanserror != 0 and kmeanspasses < 100:
            #print(error)
            # Measure the distance to every center
            for i in range(int(self.k.get())):
                self.distances[:,i] = np.linalg.norm(self.x - self.kmeanscenters[i], axis=1)
            # Assign all training data to closest center
            self.kmeansclusters = np.argmin(self.distances, axis = 1)
            print(self.kmeansclusters)
            self.kmeanscenters_old = deepcopy(self.kmeanscenters_new)
            # Calculate mean for every cluster and update the center
            for i in range(int(self.k.get())):
                self.kmeanscenters_new[i] = np.mean(self.x[self.kmeansclusters == i], axis=0)
            self.kmeanserror = np.linalg.norm(self.kmeanscenters_new - self.kmeanscenters_old)
            print(self.kmeanserror)
            kmeanspasses = kmeanspasses+1
            #print(kmeanspasses)
        #self.dataset = pandas.concat([self.dataset,self.kmeanserror],axis=1)
        #print(xplus)
        #print(pandas.Series(self.kmeansclusters))
        #clus = pandas.Series(self.kmeansclusters)
        #xplus.iloc['op'] = clus.iloc[:,0]
        xplus = self.x.assign(clusters = pandas.Series(self.kmeansclusters).values)
        #print(xplus)
        
        #xplus.iloc['op'] = pandas.Series(self.kmeansclusters)
        sns.pairplot(xplus, hue = 'clusters', size=1.8, aspect=1.8, 
                      palette='rainbow',
                      plot_kws=dict(edgecolor="black", linewidth=0.5))
        
    
    
    def is_number(self,testn): #Useful in function DecideLabel. Called in conjunction with DecideLabel, not by itself. Tells if value passed is a number or a string
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
        #self.name = self.colname[self.i]
        tempname = self.is_number(name)
        if type(tempname) is str:
            return name
        elif type(tempname) is not str :
            return "column " + str(i+1)

class loadmodules():
    
    def __init__(self): #constructor of the class
        self.filename = ' '
        #self.dataset = dataset
        #self.width = 2
        #self.height = 5
        #self.scaler = StandardScaler()
        #self.colname = self.dataset.columns.values
        #self.dup_dataset = copy.copy(self.dataset)
        #self.ticks = [' ' for self.i in range(len(self.dataset.columns.values))]
        #self.tickmark = [0 for self.i in range(len(self.dataset.columns.values))]
        #self.alpha_data_process()
            
    def myfunction(self,event):  #used in scroll function
                self.canvas.configure(scrollregion=self.canvas.bbox("all"),width=self.widh,height=self.heig)
        
    def scrollwindow(self,widh,heig):  #creates a scroll window for desired TK frame
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
        
    def loadmodel(self): #Creates a dialog box and asks for model to be loaded. Determines if model is NN, Regression or SVM
        self.filename = askopenfilename()
        self.loadmodelwindow = tk.Toplevel()
        self.loadmodelwindow.geometry("800x500")
        self.loadmodelwindow.title('Load model')
        self.ipvalues = {}
        self.scrollwin = self.loadmodelwindow
        self.scrollwindow(750,450)
        
        
        #filename = 'D:/ML_data/iris.data.csv' 
        #filename_model = 'pickle_trial2.sav'
        self.loaded_model = pickle.load(open(self.filename, 'rb'))
        self.modeltype = str(type(self.loaded_model))
        if self.modeltype == "<class 'keras.engine.sequential.Sequential'>":
            self.loadmodelNN()
        elif self.modeltype == "<class 'sklearn.linear_model.base.LinearRegression'>":
            self.loadmodelLR()
        elif self.modeltype == "<class 'sklearn.svm.classes.SVC'>":
            self.loadmodelSM()
        #self.loaded_model.summary()
    
    def loadmodelSM(self): #Finds input and output shape of SVM model
        self.ipshape = self.loaded_model.support_vectors_.shape
        print(self.ipshape)
        tk.Label(self.frame, text = 'Enter input values',font=("Courier", 15)).grid(row=0, column=0)
        for i in range(self.ipshape[1]):
            self.ipvalues[i] = tk.StringVar()
            self.ipvalues[i].set('0')
            tk.Entry(self.frame, textvariable = self.ipvalues[i]).grid(row = i+1, column = 0)
        tk.Button(self.frame, text = 'Predict',command = self.loadmodel_predict).grid(row=0, column=1, sticky = 'w')
        tk.Label(self.frame, text = 'Predicted value(s):' ,font=("Courier", 12) ).grid(row=1, column=1,sticky='w')
        
    def loadmodelLR(self): #Finds input and output shape of Regression model
        self.ipshape = self.loaded_model.coef_.shape
        print(self.ipshape)
        tk.Label(self.frame, text = 'Enter input values',font=("Courier", 15)).grid(row=0, column=0)
        for i in range(self.ipshape[1]):
            self.ipvalues[i] = tk.StringVar()
            self.ipvalues[i].set('0')
            tk.Entry(self.frame, textvariable = self.ipvalues[i]).grid(row = i+1, column = 0)
        tk.Button(self.frame, text = 'Predict',command = self.loadmodel_predict).grid(row=0, column=1, sticky = 'w')
        tk.Label(self.frame, text = 'Predicted value(s):' ,font=("Courier", 12) ).grid(row=1, column=1,sticky='w')
        
    def loadmodelNN(self):#Finds input and output shape of NN model
        self.ipshape = self.loaded_model.input_shape
        print(self.ipshape)
        tk.Label(self.frame, text = 'Enter input values',font=("Courier", 15)).grid(row=0, column=0)
        for i in range(self.ipshape[1]):
            self.ipvalues[i] = tk.StringVar()
            self.ipvalues[i].set('0')
            tk.Entry(self.frame, textvariable = self.ipvalues[i]).grid(row = i+1, column = 0)
        tk.Button(self.frame, text = 'Predict',command = self.loadmodel_predict).grid(row=0, column=1, sticky = 'w')
        tk.Label(self.frame, text = 'Predicted value(s):' ,font=("Courier", 12) ).grid(row=1, column=1,sticky='w')
        
    def loadmodel_predict(self): #Predicts output to the input value based on inputs and output shape
        tempvar = []
        for i in range(self.ipshape[1]):
            #print(self.ipvalues[i].get())
            tempvar.append(float(self.ipvalues[i].get()))
        #print(tempvar)
        tempvar = np.array([tempvar])
        print(tempvar)
        #print(type(tempvar))
        predval = self.loaded_model.predict(tempvar)
        print(predval)
        #tk.Label(self.frame, text = 'Predicted value(s) :',font=("Courier", 12) ).grid(row=1, column=1,sticky='w')
        if self.modeltype == "<class 'keras.engine.sequential.Sequential'>":
            print(self.loaded_model.output_shape)
            for i in range((self.loaded_model.output_shape[1])):
                tk.Label(self.frame, text = str(predval[0][i])+'           ',font=("Courier", 12) ).grid(row=i+2, column=1,sticky='w')
        elif self.modeltype == "<class 'sklearn.linear_model.base.LinearRegression'>":
            print(self.ipshape[0])
            for i in range((self.ipshape[0])):
                tk.Label(self.frame, text = str(predval[0][i])+'           ',font=("Courier", 12) ).grid(row=i+2, column=1,sticky='w')
        elif self.modeltype == "<class 'sklearn.svm.classes.SVC'>":
            tk.Label(self.frame, text = 'class '+str(predval[0])+'           ',font=("Courier", 12) ).grid(row=2, column=1,sticky='w')