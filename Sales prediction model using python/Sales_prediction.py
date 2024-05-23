
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
df = pd.read_csv("advertising.csv")
df.head()

# %%
df.shape

# %%
df.describe()

# %%
# Create pair plots
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', kind='scatter')

# Show the plot
plt.show()


# %%
df['TV'].plot.hist(bins=10)

# %%
df['Radio'].plot.hist(bins=10,color="green", xlabel="Radio")

# %%
df['Newspaper'].plot.hist(bins=10,color="purple", xlabel="Newspaper")

# %%
sns.heatmap(df.corr(),annot= True)
plt.show()

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test= train_test_split(df[['TV']],df[['Sales']],test_size=0.3,random_state =0)

# %%
print(x_train)

# %%
print(y_train)

# %%
print(x_test)

# %%
print(y_test)

# %%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# %%
res = model.predict(x_test)
print(res)

# %%
model.coef_

# %%
model.intercept_


# %%
0.05473199*69.2 +7.1432225

# %%
plt.plot(res)

# %%
plt.scatter(x_test, y_test)
plt.plot(x_test,7.1432225+0.05473199 *x_test , 'r')
plt.show()


