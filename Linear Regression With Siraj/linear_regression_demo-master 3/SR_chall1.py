
# coding: utf-8

# In[60]:

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
    
rdata = pd.read_fwf('challenge_dataset.txt')
rdata = pd.DataFrame(rdata.FirstSecond.str.split(',',1).tolist(), columns=['First','Second'])


# In[62]:

print(rdata)


# In[88]:

x = rdata[['First']]
y = rdata[['Second']]


# In[89]:

regress = linear_model.LinearRegression()
regress.fit(x , y)


# In[90]:

plt.scatter(x,y)
plt.plot(x,regress.predict(x))


# In[91]:

plt.show()


# In[ ]:



