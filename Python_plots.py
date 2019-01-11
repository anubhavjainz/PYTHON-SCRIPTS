# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:36:34 2018

@author: 555224
"""


import numpy as np               # For linear algebra
import pandas as pd              # For data manipulation
import matplotlib.pyplot as plt  # For 2D visualization
import seaborn as sns


###########DATA IMPORT

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
headers

DF=pd.read_csv("D://imports-85.data.txt",header=None,names=headers)

DF.describe()

DF.dtypes


###########DATA PREPARATION

DF.replace("?",np.nan,inplace=True)

help(DF.describe)

DF.describe(include='all')

help(DF.dropna)

DF.dropna(axis=0,subset=["price"],inplace=True)

DF.describe(include='all')

DF["price"]=DF["price"].astype("int")

print(DF[DF['normalized-losses'].isnull()])

DF['normalized-losses'] = pd.to_numeric(DF['normalized-losses'], errors='coerce')



DF["normalized-losses"].dtypes


DF2=pd.get_dummies(DF['fuel-type'])

DF2.columns=['fuel-diesel','fuel_gas']

help(pd.concat)

DF=pd.concat([DF,DF2],axis=0)


######## DATA VISUALIZATION

#BOX PLOT
sns.boxplot(y="price",data=DF)

sns.boxplot(x="body-style", y="price", data=DF)
plt.yticks(rotation=45)
plt.xticks(rotation=45)

####### REGRESSION PLOT

tips = sns.load_dataset("tips")
tips.head()

help(sns.regplot)
sns.regplot(y='price',x='length',data=DF,fit_reg=False)


sns.regplot(x="total_bill", y="tip",data=tips)

sns.lmplot(y='price',x='length',data=DF,fit_reg=False,size=10,hue='fuel-type') 

sns.factorplot(x='price',y='length',data=DF,hue='fuel-type',kind='box',size=10)

help(sns.factorplot)

########## DENSITY PLOT

sns.kdeplot(DF['price'], shade=True)

########## HISTOGRAM

plt.hist(x=DF['price'],bins=15,color='red')
help(plt.hist)

sns.distplot(DF['price'],bins=10)
########## PAIRPLOT

help(sns.pairplot)


iris = sns.load_dataset("iris")
g = sns.pairplot(tips,hue='sex')

########## FACET GRID

tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))


grid = sns.FacetGrid(tips, row="sex",col="smoker", margin_titles=True)
grid.map(sns.kdeplot, "tip_pct")


with sns.axes_style(style='ticks'):
    g = sns.factorplot(x="day",y= "total_bill", hue="sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill");
    
  
#MULTIPLE PLOTS

f,ax = plt.subplots(2,2,figsize=(8,4))
vis1 = sns.distplot(DF["price"],bins=10, ax= ax[0][0])
vis3 = sns.distplot(DF["width"],bins=10, ax=ax[1][0])
vis4 = sns.regplot(y='price',x='length',data=DF,fit_reg=False,ax=ax[1][1])


#RESIDUAL PLOTS
sns.residplot(DF['FUELCONSUMPTION_COMB_MPG'],DF['CO2EMISSIONS'])
plt.show()

###### SCATTER PLOT
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS)
plt.xlabel("Engine Size")
plt.ylabel("Emissions")