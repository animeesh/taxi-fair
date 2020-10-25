import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("taxi.csv")
#print(data.head())

#[:,0:-1] all rows and all expect last column (independent  variable) also feature column
data_x = data.iloc[:,:-1].values

#[:,-1]all rows and only last column (dependent variable) also target column
data_y = data.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3, random_state=0)

reg = LinearRegression()
reg.fit(x_train,y_train)



print("train_score=" ,reg.score(x_train,y_train))
print("train_score=" ,reg.score(x_test,y_test))

pickle.dump(reg,open("taxi.pkl",'wb'))
model=pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80,1770000,6000,85]]))

'''
print(reg.predict([[80,1770000,6000,85]]))

%matplotlib inline
plt.xlabel("feature")
plt.ylabel("target")
plt.scatter(data_x.feature,data_y.target,color= red, marker="+")
'''