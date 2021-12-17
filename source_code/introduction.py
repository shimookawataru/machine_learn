import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="IPAexGothic")
import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
df1 = pd.read_table("../data/diabete.txt")
lreg = LinearRegression()
X=df1.drop('Y',axis=1)
Y=df1['Y']
lreg.fit(X,Y)
test1 = np.array([[24,1,25,84,198,131,40,5,5,89]])
print(lreg.predict(test1))
test1 = np.array([[50,2,30,100,180,120,50,4,5,77]])
print(lreg.predict(test1))
test1 = np.array([[35,1,25,80,130,50,60,4,4,60]])
print(lreg.predict(test1))