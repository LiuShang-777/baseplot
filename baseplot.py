# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:55:45 2019

@author: liushang
"""
print('>==Starting the basic plot===<')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import argparse
print('Import the packages successfully')
parser=argparse.ArgumentParser(description='basic plot based on matplotlib and seaborn')
parser.add_argument('-df','--dataframe',type=str,help='the input file of dataframe')
parser.add_argument('-f','--features',type=str,help='the features you select in dataframe',nargs='*')
parser.add_argument('-rf','--result_file',type=str,help='the result directory')
parser.add_argument('-sp','--sns_plot',type=str,help='the seaborn plot',choices=['density','regplot','distplot','heat_map_d',
                                                                                 'heat_map_r','heat_map_c'])
parser.add_argument('-bp','--baseplot',type=str,help='don\'t depend on seaborn',choices=['violin',
                                                        'pie', 'radar','hist','row_barh','row_bar','multi_plot','single_plot'])
parser.add_argument('-s','--size',type=str,help='the size of picture saved',nargs='*')
parser.add_argument('-ri','--radar_index',type=str,help='the index that radar plot display')
args=parser.parse_args()
data=pd.read_csv(args.dataframe,sep='\t')
features=args.features
result=args.result_file
name=data.columns.tolist()[0]
data=pd.read_csv(args.dataframe,sep='\t',index_col=name)
if len(args.size)!=2:
    exit('the picture\'s size shall be in two dimensions')
else:    
    figheight=int(args.size[1])
    figwidth=int(args.size[0])
#matplotlib is adequate
#单线形图
if args.baseplot !=None:
    print('***you choose the basic plot***')
    if args.baseplot=='single_plot':       
        def single_plot(single_feature):
            fig=plt.figure(figsize=(figwidth,figheight))
            ax=fig.add_subplot(1,1,1)
            global data
            ax.plot(data[single_feature])
            ax.set_xticklabels(data.index)
            plt.savefig(result+'single_plot.svg')
            plt.savefig(result+'single_plot.png')
            plt.clf()
        single_plot(features[0])
    elif args.baseplot=='multi_plot':       
        def multi_plot(features):
            global data
            fig=plt.figure(figsize=(figwidth,figheight))
            ax=fig.add_subplot(1,1,1)
            ax.plot(data[features])
            ax.set_xticklabels(data.index)
            plt.legend()
            plt.savefig(result+'multi_plot.svg')
            plt.savefig(result+'multi_plot.png')
            plt.clf()
        multi_plot(features)
    elif args.baseplot=='row_bar':
        #柱状图  
        def row_bar(dataframe,features,result):
            global data
            plt.figure(figsize=(figwidth,figheight))
            dataframe=data[features]
            dataframe.plot.bar()
            plt.savefig(result+'row_bar.svg')
            plt.savefig(result+'row_bar.png')
        row_bar(data,features,result)
    elif args.baseplot=='row_barh':        
        def row_barh(dataframe,features,result):
            global data
            plt.figure(figsize=(figwidth,figheight))
            dataframe=data[features]
            dataframe.plot.barh(stacked=True)
            plt.savefig(result+'row_barh.svg')
            plt.savefig(result+'row_barh.png')
            plt.clf()
        row_barh(data,features,result)
    elif args.baseplot=='hist':       
        def hist(dataframe,features,result):
            global data
            plt.figure(figsize=(figwidth,figheight))
            if len(features)!=1:
                exit('the hist plot only need one feature')
            dataframe=data
            n,bins,patches=plt.hist(dataframe[features[0]],color='r',edgecolor='k',bins=10,rwidth=0.9)
            sigma,mean=dataframe[features[0]].std(),dataframe[features[0]].mean()                       
            plt.xlabel(features[0])
            plt.ylabel('number')
            plt.title('the distibution of %s (mean=%.3f,std=%.3f)'%(features[0],mean,sigma))
            plt.savefig(result+'hist.svg')
            plt.savefig(result+'hist.png')
            plt.clf()
        hist(data,features,result)
    elif args.baseplot=='radar':
            index=args.radar_index
            def radar(index,features,result):
                global data
                dataframe=data[features]
                dataframe=dataframe[dataframe.index==index]
                theta=np.linspace(0,2*np.pi,len(features),endpoint=False)
                values=np.array(dataframe)[0]
                theta=np.concatenate((theta,[theta[0]]))
                values=np.concatenate((values,[values[0]]))
                ax=plt.subplot(111,projection='polar')
                ax.plot(theta,values,'b-',linewidth=0.5)
                ax.fill(theta,values,'b')
                ax.set_thetagrids(theta*180/np.pi,features)  
                plt.savefig(result+'radar.svg')
                plt.savefig(result+'radar.png')
                plt.clf()
            radar(index,features,result)
    elif args.baseplot=='violin':       
        def violin(dataframe,features,result):
            global data
            figure=plt.figure(figsize=(figwidth,figwidth))
            ax=figure.add_subplot(1,1,1)
            dataframe=data[features]
            plotlist=[]
            for i in features:
                plotlist.append(dataframe[i])
            ax.violinplot(plotlist)
            ax.set_xticks(range(1,len(features)+1))
            ax.set_xticklabels(features)
            plt.savefig(result+'violin.svg')
            plt.savefig(result+'violin.png')
            plt.clf()
        violin(data,features,result)
    elif args.baseplot=='pie':
        classify=features[0]
        def pie(dataframe,classify,result):
            global data
            plt.figure(figsize=(figwidth,figwidth))
            dataframe=data.groupby(classify).size()
            dataframe.name=''
            plt.axes(aspect='equal')
            dataframe.plot(kind='pie',counterclock=False,
                           autopct='%.1f%%',
                              radius=1,
                              startangle=180)
            plt.savefig(result+'pie.svg')
            plt.savefig(result+'pie.png')
            plt.clf()
        pie(data,classify,result)
elif args.sns_plot!=None:
    print('>===you have chose plotting on seaborn===<')
    import seaborn as sns
    print('seaborn is imported successfully')
    if args.sns_plot=='density':
        def density(dataframe,features,result):
            global data
            if len(features)!=1:
                exit('density could only accept one feature to output the plot')
            dataframe=data[features[0]]
            plt.figure(figsize=(figwidth,figheight))
            sns.distplot(dataframe,color='k')
            plt.savefig(result+'density.svg')
            plt.savefig(result+'density.png')
            plt.clf()
        density(data,features,result)
    elif args.sns_plot=='regplot':      
        def regplot(dataframe,features,result):
            global data
            if len(features)!=2:
                exit('the regression plot should accept two features')
            dataframe=data[features]
            plt.figure(figsize=(figwidth,figheight))
            sns.regplot(features[0],features[1],data=dataframe)
            plt.savefig(result+'regplot.svg')
            plt.savefig(result+'regplot.png')
            plt.clf()
        regplot(data,features,result)
    elif args.sns_plot=='distplot':        
        def dist_plot(dataframe,features,result):
            dataframe=data[features]
            plt.figure(figsize=(figwidth,figheight))
            sns.pairplot(dataframe,diag_kind='kde')
            plt.savefig(result+'distri.svg')
            plt.savefig(result+'distri.png')
            plt.clf()
        dist_plot(data,features,result)
    elif args.sns_plot=='heat_map_d':       
        def heat_map_d(dataframe,features,result):
            global data
            dataframe=data[features]
            plt.figure(figsize=(figwidth,figheight))
            sns.clustermap(dataframe,method ='ward',metric='euclidean',cmap='RdYlBu',linewidths=0.3)
            plt.savefig(result+'clust_heat_double.svg')
            plt.savefig(result+'clust_heat_double.png')
            plt.clf()
        heat_map_d(data,features,result)
    elif args.sns_plot=='heat_map_r':
        def heat_map_r(dataframe,features,result):
            global data
            dataframe=data[features]
            plt.figure(figsize=(figwidth,figheight))
            sns.clustermap(dataframe,method ='ward',metric='euclidean',cmap='RdYlBu',linewidths=0.3,col_cluster=False)
            plt.savefig(result+'clust_heat_row.svg')
            plt.savefig(result+'clust_heat_row.png')
            plt.clf()
        heat_map_r(data,features,result)
    elif args.sns_plot=='heat_map_c':
        def heat_map_c(dataframe,features,result):
            global data
            dataframe=data[features]
            plt.figure(figsize=(figwidth,figheight))
            sns.clustermap(dataframe,method ='ward',metric='euclidean',cmap='RdYlBu',linewidths=0.3,row_cluster=False)
            plt.savefig(result+'clust_heat_col.svg')
            plt.savefig(result+'clust_heat_col.png')
            plt.clf()
        heat_map_c(data,features,result)
else:
    exit('You could only use one type of plot one time')
print('>===the plotting process was end===<')