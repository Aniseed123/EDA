# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:31:14 2019

Update:     trial2:
            >using classes
            trial 3:
            >Using factorize for non-numeric data
            trial 4:
            >Attempting to use duplicate dataset where factorization is not vital
            >13/3/19: Made DecideLabel and is_number independent of val (using local variables now)
@author: 510571
"""
import tkinter as tk
import matplotlib.pyplot as plot
import pandas
import numpy as np
import seaborn as sns
import itertools
from mpl_toolkits.mplot3d import Axes3D
import copy

class visuals():
    
    def __init__(self,dataset): #constructor of class
        self.dataset = dataset
        self.scrollwin = 'NONE'
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
        '''         
        for i in range(len(self.dataset.columns.values)):
            if( (self.dataset.iloc[:,i].dtype != np.number) or (self.dataset.iloc[:,i].dtype != np.floating)):
                    self.ticks[i] = np.unique(self.dataset.iloc[:,i])
                    self.tickmark[i] = 1
                    self.tempcol = pandas.factorize(self.dataset.iloc[:,i])
                    self.tempcol[0]
                    self.dataset.iloc[:,i] 
        '''      

    def myfunction(self,event): #vital function in scrollwindow()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"),width=self.widh,height=self.heig)
        
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
    
    def selectallboxes(self,col1):   #To select all columns at once in OpenFileSelective
        lft = {}
        
        for val in range(self.shape[1]):
            col1[val] = tk.IntVar()
            lft[val] = tk.Checkbutton(self.frame, text = self.DecideLabel(self.colname[val],val), variable = col1[val])
            lft[val].select()
            lft[val].grid(row=val+2, column=0, sticky = 'w')
            
    def Plotgraph(self): #Function to plot 2D graph along X and Y axis
        self.plotwindow = tk.Toplevel()
        self.plotwindow.geometry("500x500")
        self.plotwindow.title('Plot Graph')
        #global colname
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.plotwindow
        self.scrollwindow(450,450)
        
        
        
        self.b1 = tk.Label(self.frame, text = 'Y axis')
        self.b2 = tk.Label(self.frame, text = 'X axis') 
        self.b1.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
    
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
            self.rgt = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
            
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickplotgraph)
        self.clickbutton.grid(row=0, column=60, ipadx = 0, ipady =0)
    def clickplotgraph(self):   #Function executed after button pressed in Plotgraph() function
            #print('using dup_dataset')
            self.x = self.dup_dataset.iloc[:,self.v2.get()]
            self.y = self.dup_dataset.iloc[:,self.v1.get()]
            
            plot.figure(figsize = (10,10))
            '''
            if (self.tickmark[self.v2.get()] == 1):
                plot.xticks(np.unique(self.x), self.ticks[self.v2.get()])
            if (self.tickmark[self.v1.get()] == 1):
                plot.yticks(np.unique(self.y), self.ticks[self.v1.get()])
            '''    
            plot.xlabel(self.colname[self.v2.get()])

            plot.ylabel(self.colname[self.v1.get()])
            plot.scatter(self.x,self.y)
            plot.title('Graph')
            plot.show()
    
    def Histogram(self): #Function to plot histogram for a column
        self.histwindow = tk.Toplevel()
        self.histwindow.geometry("500x500")
        self.histwindow.title('Histogram')
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.histwindow
        self.scrollwindow(450,450)
        
        self.w1 = tk.Scale(self.frame, from_=1,tickinterval=0, to=100, orient= 'horizontal', length = 200, label = 'Select Bins')
        self.w1.set(10)
        self.w1.grid(row = 0, column = 1, sticky = 'w',ipadx = 10)

        self.b1 = tk.Label(self.frame, text = 'Select attribute: ')
        self.b1.grid(row=0, column=0, ipadx = 10)
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
        
        
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickhistogram)
        self.clickbutton.grid(row=0, column=2, ipadx = 10)
        
    def clickhistogram(self): #Function executed after button pressed in Histogram() function
        self.x = self.dup_dataset.iloc[:,self.v.get()]
        plot.figure(figsize = (10,10))
        '''
        if (self.tickmark[self.v.get()] == 1):
                plot.xticks(np.unique(self.x), self.ticks[self.v.get()])
        '''
        plot.xlabel(self.colname[self.v.get()])
        plot.hist(self.x, bins = self.w1.get())
        plot.title('Histogram')
        plot.show()
            
    def CorrDiagram(self): #Function to show Correaltion diagrams between columns
        plot.figure(figsize = (10,10))
        plot.title('Correlation Matrix')
        self.corr = self.dataset.corr()
        self.colname = self.dataset.columns.values
        corrlabel = [' ' for i in range(len(self.colname))]
        #ycorr = [' ' for i in range(len(self.corr.columns.values))]
        for self.val in range(len(self.colname)):
            corrlabel[self.val] = self.DecideLabel(self.colname[self.val],self.val)
            
        sns.heatmap(self.corr, 
                xticklabels=corrlabel,
                yticklabels=corrlabel)
        
    def PieChart(self): #Function to plot a pie chart for a column
        self.piewindow = tk.Toplevel()
        self.piewindow.geometry("500x500")
        self.piewindow.title('Pie Chart')
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.piewindow
        self.scrollwindow(450,450)
        
        self.b1 = tk.Label(self.frame, text = 'Select column to plot')
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
            
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickpiechart)
        self.clickbutton.grid(row=0, column=1)
    
    def clickpiechart(self):   #Function executed after button pressed in PieChart() function
           
            plot.figure(figsize = (10,10))
            '''
            if (self.tickmark[self.v.get()] == 1):
                self.labeldata = self.ticks[self.v.get()]
                #plot.xticks(np.unique(self.x), self.ticks[self.v2.get()])
            else:
            '''    
            self.labeldata = self.dup_dataset.iloc[:,self.v.get()].unique()
            plot.xlabel(self.colname[self.v.get()])
            plot.title('Pie Chart')
            plot.show()
            plot.pie(self.dataset.groupby([self.dataset.iloc[:,self.v.get()]]).size(), labels = self.labeldata, autopct='%1.0f%%')
    
    def BarGraph(self): #Function to plot a bargraph between X an Y axis
        self.barwindow = tk.Toplevel()
        self.barwindow.geometry("500x500")
        self.barwindow.title('Bar Graph')
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.barwindow
        self.scrollwindow(450,450)
        
        self.w1 = tk.Scale(self.frame, from_=0.01, to=1, orient= 'horizontal', length = 200, label = 'Select Width', resolution = 0.01)
        self.w1.set(0.8)
        self.w1.grid(row = 0, column = 3, sticky = 'w',ipadx = 5)
        self.b1 = tk.Label(self.frame, text = 'Y axis')
        self.b2 = tk.Label(self.frame, text = 'X axis') 
        self.b1.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
            self.rgt = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
        self.clickbutton = tk.Button(self.frame, text = 'Plot',command = self.clickbargraph)
        self.clickbutton.grid(row=0, column=2, ipadx = 0)
        
    def clickbargraph(self): #Function executed after button pressed in BarGraph() function
        self.x = self.dup_dataset.iloc[:,self.v2.get()]
        self.y = self.dup_dataset.iloc[:,self.v1.get()]
        plot.figure(figsize = (10,10))
        '''
        if (self.tickmark[self.v2.get()] == 1):
                plot.xticks(np.unique(self.x), self.ticks[self.v2.get()])
        if (self.tickmark[self.v1.get()] == 1):
            plot.yticks(np.unique(self.y), self.ticks[self.v1.get()])
        '''
        plot.xlabel(self.colname[self.v2.get()])
        plot.ylabel(self.colname[self.v1.get()])
        plot.bar(self.x,self.y,width = self.w1.get())
        plot.title('Bar Graph')
        plot.show()
            
    def Graph3D(self): #Function to plot a 3D graph with X, Y and Z axis
        self.G3Dwindow = tk.Toplevel()
        self.G3Dwindow.geometry("500x500")
        self.G3Dwindow.title('3D Scatter Graph')
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.G3Dwindow
        self.scrollwindow(450,450)
        
        self.b1 = tk.Label(self.frame, text = 'Y axis')
        self.b2 = tk.Label(self.frame, text = 'X axis')
        self.b3 = tk.Label(self.frame, text = 'Z axis')
        self.b1.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.b3.grid(row=0, column =2, ipadx = 0, ipady =0, sticky = 'w')
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v3 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        self.v3.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
            self.rgt = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
            self.zax = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v3, 
                  value=self.val)
            self.zax.grid(row = self.val+1 ,column = 2,ipadx = 0, ipady =0, sticky = 'w')
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickgraph3d)
        self.clickbutton.grid(row=0, column=3, ipadx = 0)
    
    def clickgraph3d(self): #Function executed after button pressed in Graph3D() function
            self.x = self.dataset.iloc[:,self.v2.get()]
            self.y = self.dataset.iloc[:,self.v1.get()]
            self.z = self.dataset.iloc[:,self.v3.get()]
            
            self.fig = plot.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            if (self.tickmark[self.v2.get()] == 1):
                self.ax.set_xticks(np.unique(self.x))
                self.ax.set_xticklabels(self.ticks[self.v2.get()])
            if (self.tickmark[self.v1.get()] == 1):
                self.ax.set_yticks(np.unique(self.y))
                self.ax.set_yticklabels(self.ticks[self.v1.get()])
            if (self.tickmark[self.v3.get()] == 1):
                self.ax.set_zticks(np.unique(self.z))
                self.ax.set_zticklabels(self.ticks[self.v3.get()])
            self.ax.scatter(self.x, self.y, self.z, s=50, alpha=0.6, edgecolors='w')
            self.ax.set_xlabel(self.colname[self.v2.get()])
            self.ax.set_ylabel(self.colname[self.v1.get()])
            self.ax.set_zlabel(self.colname[self.v3.get()])
    
    def BubbleChart(self): #Function to plot a Bubble chart
        self.bubblewindow = tk.Toplevel()
        self.bubblewindow.geometry("700x500")
        self.bubblewindow.title('Bubble Chart')
        
        self.colname = self.dataset.columns.values
        
        self.scrollwin = self.bubblewindow
        self.scrollwindow(650,450)
        
        self.col1 = {}
        self.shape = self.dataset.shape
        self.w1 = tk.Scale(self.frame, from_=1, to=500, orient= 'horizontal', length = 200, label = 'Bubble Magnification')
        self.w1.set(25)
        self.w1.grid(row = 0, column = 4, sticky = 'w',ipadx = 10)
        
        
        self.b1 = tk.Label(self.frame, text = 'Y axis')
        self.b2 = tk.Label(self.frame, text = 'X axis')
        self.b3 = tk.Label(self.frame, text = 'Bubble size attribute')
        self.b1.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.b3.grid(row=0, column =2, ipadx = 0, ipady =0, sticky = 'w')
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v3 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        self.v3.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
            self.rgt = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
            self.zax = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v3, 
                  value=self.val)
            self.zax.grid(row = self.val+1 ,column = 2,ipadx = 0, ipady =0, sticky = 'w')
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickbubblechart)
        self.clickbutton.grid(row=0, column=3, ipadx = 0)
    
    def clickbubblechart(self): #Function executed after button pressed in BubbleChart() function
            self.x = self.dataset.iloc[:,self.v2.get()]
            self.y = self.dataset.iloc[:,self.v1.get()]
            self.z = self.dataset.iloc[:,self.v3.get()]
           
            self.fig = plot.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111)
            
            if (self.tickmark[self.v2.get()] == 1):
                self.ax.set_xticks(np.unique(self.x))
                self.ax.set_xticklabels(self.ticks[self.v2.get()])
            if (self.tickmark[self.v1.get()] == 1):
                self.ax.set_yticks(np.unique(self.y))
                self.ax.set_yticklabels(self.ticks[self.v1.get()])
            
            plot.scatter(self.x, self.y, s=self.z*self.w1.get(), 
                alpha=0.4, edgecolors='w')
            self.ax.set_title('Bubble Chart')
            self.ax.set_xlabel(self.colname[self.v2.get()])
            self.ax.set_ylabel(self.colname[self.v1.get()])
            
    def HueGraph(self): #Function to plot a hue graph between X and Y axis and a Hue column
        self.huewindow = tk.Toplevel()
        self.huewindow.geometry("700x500")
        self.huewindow.title('Hue Graph')
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.huewindow
        self.scrollwindow(650,450)
        
        self.b1 = tk.Label(self.frame, text = 'Y axis')
        self.b2 = tk.Label(self.frame, text = 'X axis')
        self.b3 = tk.Label(self.frame, text = 'Hue attribute')
        self.b1.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.b3.grid(row=0, column =2, ipadx = 0, ipady =0, sticky = 'w')
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v3 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        self.v3.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
            self.rgt = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
            self.zax = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v3, 
                  value=self.val)
            self.zax.grid(row = self.val+1 ,column = 2,ipadx = 0, ipady =0, sticky = 'w')
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickhuegraph)
        self.clickbutton.grid(row=0, column=3, ipadx = 0)
    
    def clickhuegraph(self): #Function executed after button pressed in HueGraph() function
            self.x = self.dataset.iloc[:,self.v2.get()]
            self.y = self.dataset.iloc[:,self.v1.get()]
            self.z = self.dataset.iloc[:,self.v3.get()]
           
            self.fig = plot.figure(figsize=(10, 10))
            self.ax = self.fig.add_subplot(111)
            
            if (self.tickmark[self.v2.get()] == 1):
                self.ax.set_xticks(np.unique(self.x))
                self.ax.set_xticklabels(self.ticks[self.v2.get()])
            if (self.tickmark[self.v1.get()] == 1):
                self.ax.set_yticks(np.unique(self.y))
                self.ax.set_yticklabels(self.ticks[self.v1.get()])
           
            print(self.z)
            plot.scatter(self.x, self.y, c=self.z, cmap=plot.cm.coolwarm,edgecolors='black',s =50)
            self.ax.set_title('Hue Graph')
            self.ax.set_xlabel(self.colname[self.v2.get()])
            self.ax.set_ylabel(self.colname[self.v1.get()])
            #self.ax.legend( labels = self.ticks[self.v3.get()]) #!!!INCORRECT-NEED TO FIX
            plot.show()
    
    def KernelPlot(self): #Function to plot a kernel density plot between X and Y axis
        self.kernelwindow = tk.Toplevel()
        self.kernelwindow.geometry("500x500")
        self.kernelwindow.title('Kernel Density Plot')
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
       
        self.scrollwin = self.kernelwindow
        self.scrollwindow(450,450)
        
        self.b1 = tk.Label(self.frame, text = 'Y axis')
        self.b2 = tk.Label(self.frame, text = 'X axis')
        self.b1.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
            self.rgt = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickkernalplot)
        self.clickbutton.grid(row=0, column=2, ipadx = 0)
    
    def clickkernalplot(self): #Function executed after button pressed in KernelPlot() function
            self.x = self.dataset.iloc[:,self.v2.get()]
            self.y = self.dataset.iloc[:,self.v1.get()]
            
            self.fig = plot.figure(figsize=(10, 10))
            
            self.ax = self.fig.add_subplot(111)
            
            if (self.tickmark[self.v2.get()] == 1):
                self.ax.set_xticks(np.unique(self.x))
                self.ax.set_xticklabels(self.ticks[self.v2.get()])
            if (self.tickmark[self.v1.get()] == 1):
                self.ax.set_yticks(np.unique(self.y))
                self.ax.set_yticklabels(self.ticks[self.v1.get()])
            
                #self.ax.set_yticklabels(np.unique(self.y),self.ticks[self.v1.get()])
            self.ax = sns.kdeplot(self.x, self.y,
                      cmap="YlOrBr", shade=True, shade_lowest=False)
            self.ax.set_title('Kernel Density Plot')
            self.ax.set_xlabel(self.colname[self.v2.get()])
            self.ax.set_ylabel(self.colname[self.v1.get()])
            plot.show()
            #for i in len(np.unique(z)):
            #    ax.legend(i)
    
    def KernelPlot3D(self): #Function to plot 3D kernel blot btween X and Y axis and a Label
        self.kernel3dwindow = tk.Toplevel()
        self.kernel3dwindow.geometry("700x500")
        self.kernel3dwindow.title('Kernel Density Plot')
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.kernel3dwindow
        self.scrollwindow(650,450)
        
        self.b1 = tk.Label(self.frame, text = 'Y axis')
        self.b2 = tk.Label(self.frame, text = 'X axis')
        self.b3 = tk.Label(self.frame, text = 'Label')
        self.b1.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b3.grid(row=0, column =2, ipadx = 0, ipady =0, sticky = 'w')
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v3 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        self.v3.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
            self.rgt = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
            self.zax = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v3, 
                  value=self.val)
            self.zax.grid(row = self.val+1 ,column = 2,ipadx = 0, ipady =0, sticky = 'w')
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickkernalplot3d)
        self.clickbutton.grid(row=0, column=3, ipadx = 0)
    
    def clickkernalplot3d(self): #!!! Make more robust.. Function executed after button pressed in KernelPlot3D() function
    
        self.x = self.dataset.iloc[:,self.v2.get()]
        self.y = self.dataset.iloc[:,self.v1.get()]
        self.z = self.dataset.iloc[:,self.v3.get()]
        
        #print(z)
        self.label = np.unique(self.z)
        self.fig = plot.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        #labeltemp = pandas.factorize(label)
        #z = pandas.factorize(z)
        self.dup_x = self.x
        self.dup_y = self.y
        self.shapeleb = self.label.shape
        self.w, self.h = self.shape[0], self.shapeleb[0]
        self.tempx = [['x' for self.dup_x in range(self.w)] for self.dup_y in range(self.h)]
        self.tempy = [['x' for self.dup_x in range(self.w)] for self.dup_y in range(self.h)]
        self.newarrayx = np.column_stack((self.x,self.z))
        self.newarrayy = np.column_stack((self.y,self.z))
        for self.i in range(self.shapeleb[0]):
             for self.k in range(self.shape[0]):
                 if(self.label[self.i] == self.newarrayx[self.k][1]):
                     self.tempx[self.i][self.k] = (self.newarrayx[self.k][0])
        for self.i in range(self.shapeleb[0]):
             for self.k in range(self.shape[0]):
                 if(self.label[self.i] == self.newarrayy[self.k][1]):
                     self.tempy[self.i][self.k] = (self.newarrayy[self.k][0])
        for self.i in range(self.shapeleb[0]):
            while 'x' in self.tempy[self.i]:
                self.tempy[self.i].remove('x')
        for self.i in range(self.shapeleb[0]):
            while 'x' in self.tempx[self.i]:
                self.tempx[self.i].remove('x')
        #palette = itertools.cycle(sns.color_palette())   
        self.palette = itertools.cycle(['Blues','Reds','Greens','Greys','winter','Oranges','rainbow'])   
        print(self.tempx, self.tempy)
        for self.i in range(self.shapeleb[0]):
            try:
                self.ax = sns.kdeplot(self.tempx[self.i], self.tempy[self.i],
                cmap=next(self.palette), shade=True, shade_lowest=False)
            except Exception:
                pass
        self.ax.set_title('Kernel Density Plot')
        self.ax.set_xlabel(self.colname[self.v2.get()])
        self.ax.set_ylabel(self.colname[self.v1.get()])
        plot.show()
            
                    
        
           
            #for i in len(np.unique(z)):
            #    ax.legend(i)
            
    def ViolinPlot(self): #Function to display a Violin plot between X and Y axis
        self.violinwindow = tk.Toplevel()
        self.violinwindow.geometry("500x500")
        self.violinwindow.title('Violin Plot')
        
        self.colname = self.dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.violinwindow
        self.scrollwindow(450,450)
        
        self.b1 = tk.Label(self.frame, text = 'Y axis')
        self.b2 = tk.Label(self.frame, text = 'X axis') 
        self.b1.grid(row=0, column =1, ipadx = 0, ipady =0, sticky = 'w')
        self.b2.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        
        for self.val, self.col1 in enumerate(self.col1):
            self.lft = tk.Radiobutton(self.frame, 
            text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v1, 
                  value=self.val)
            self.lft.grid(row = self.val+1 ,column = 1,ipadx = 0, ipady =0, sticky = 'w')
            self.rgt = tk.Radiobutton(self.frame, 
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v2, 
                  value=self.val)
            self.rgt.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickviolinplot)
        self.clickbutton.grid(row=0, column=2, ipadx = 0)
        
    def clickviolinplot(self): #Function executed after button pressed in ViolinPlot() function
            self.x = self.dup_dataset.iloc[:,self.v2.get()]
            self.y = self.dup_dataset.iloc[:,self.v1.get()]
            plot.figure(figsize = (10,10))
            '''
            if (self.tickmark[self.v2.get()] == 1):
                plot.xticks(np.unique(self.x), self.ticks[self.v2.get()])
                #print('x activated')
            if (self.tickmark[self.v1.get()] == 1):
                plot.yticks(np.unique(self.y), self.ticks[self.v1.get()])
                #print('y activated')
            '''
            plot.xlabel(self.colname[self.v2.get()])
            plot.ylabel(self.colname[self.v1.get()])
            sns.violinplot(x=self.x, y=self.y, data = self.dup_dataset)
            #plot.bar(x,y,width = w1.get())
            plot.title('Violin Plot')
            plot.show()
            
    def PairGraph3D(self): #Function to plot a pairwise graph
        self.pairG3Dwindow = tk.Toplevel()
        self.pairG3Dwindow.geometry("500x500")
        self.pairG3Dwindow.title('Pairwise Graph')
        
        self.colname = self.dup_dataset.columns.values
        self.col1 = {}
        self.shape = self.dataset.shape
        
        self.scrollwin = self.pairG3Dwindow
        self.scrollwindow(450,450)
       
        self.b3 = tk.Label(self.frame, text = 'Hue column')
        self.b3.grid(row=0, column =0, ipadx = 0, ipady =0, sticky = 'w')
        self.v1 = tk.IntVar()
        self.v2 = tk.IntVar()
        self.v3 = tk.IntVar()
        self.v1.set(0)
        self.v2.set(0)
        self.v3.set(0)
        for self.i in range(self.shape[1]):
            self.col1[self.i] = [self.colname[self.i],self.i]
        for self.val, self.col1 in enumerate(self.col1):
            self.zax = tk.Radiobutton(self.frame,
                  text= self.DecideLabel(self.colname[self.val],self.val),
                  variable=self.v3, 
                  value=self.val)
            self.zax.grid(row = self.val+1 ,column = 0,ipadx = 0, ipady =0, sticky = 'w')
        self.clickbutton = tk.Button(self.frame, text = 'Plot', command = self.clickpairgraph)
        self.clickbutton.grid(row=0, column=1, ipadx = 0)
    
    def clickpairgraph(self): #Function executed after button pressed in PairGraph3D() function
        #if (self.tickmark[self.v3.get()] == 1):
                #plot.yticks(np.unique(self.y), self.ticks[self.v3.get()])
        sns.pairplot(self.dataset, hue = self.colname[self.v3.get()] ,size=1.8, aspect=1.8, 
                      palette='rainbow',
                      plot_kws=dict(edgecolor="black", linewidth=0.5))
        
    
    def viewencodedvalues(self): #Display Label encoded values in dataset if present
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
        
    def is_number(self,testn):  #Useful in function DecideLabel. Tells if value passed is a number or a string
        #self.testn = self.name
        try:
            float(testn)   # Type-casting the string to `float`.
               # If string is not a valid `float`, 
               # it'll raise `ValueError` exception
        except ValueError:
            return testn
        return float(testn)
    
    def DecideLabel(self,name,i):  #Used in naming columns for selections in menu. If columns have string heading, displays string. If numeric value/ no proper column heading, returns 'Column n'
        #self.i = self.val 
        #self.name = self.colname[self.i]
        
        tempname = self.is_number(name)
        if type(tempname) is str:
            return name
        elif type(tempname) is not str :
            return "column " + str(i+1)