# Data-analysis-and-Visualization-for-Marketing
# IBM's Cognitive Class Guided project


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

pd.set_option("precision", 2)
pd.options.display.float_format = '{:.2f}'.format

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0SL1EN/marketing_campaign.csv', sep=";")

df

df.describe()

df.info()

df.Income = df.Income.fillna(0)

df['Education'].value_counts().to_frame()

df['Marital_Status'].value_counts().to_frame()

u = list(df['Marital_Status'].unique())
u.sort()
u

df[['Education','Marital_Status']] = df[['Education','Marital_Status']].astype('category')
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df[['Education','Marital_Status', 'Dt_Customer']].info()

df.describe()

df.describe(include='category')

pd.options.display.float_format = '{:.1f}%'.format
ed = df['Education'].value_counts(normalize=True)*100
ed

pd.crosstab(df['Education'],
            df['Marital_Status'], normalize=True
            )*100

plt.figure(figsize=(12,5))
sns.heatmap((pd.crosstab(df['Education'],
            df['Marital_Status'], normalize=True
            )*100).astype('int'), annot=True, fmt="d")

pd.crosstab(df['Education'],
            df['AcceptedCmp1'], normalize=True, margins=True, margins_name="Total"
            )*100

pd.crosstab(df['Marital_Status'],
            df['AcceptedCmp1'], normalize=True, margins=True, margins_name="Total"
            )*100

pd.options.display.float_format = '{:.2f}'.format
df.pivot_table(index= "Education", columns = 'Marital_Status', values = 'AcceptedCmp1', aggfunc = "sum")

dfa1 = df[df['AcceptedCmp1']>0]
dfa1['AcceptedCmp'] = 1
dfa1

dfa2 = df[df['AcceptedCmp2']>0]
dfa2['AcceptedCmp'] = 2

dfa3 = df[df['AcceptedCmp3']>0]
dfa3['AcceptedCmp'] = 3

dfa4 = df[df['AcceptedCmp4']>0]
dfa4['AcceptedCmp'] = 4

dfa5 = df[df['AcceptedCmp5']>0]
dfa5['AcceptedCmp'] = 5

dfa = dfa1.append(dfa2)
dfa = dfa.append(dfa3)
dfa = dfa.append(dfa4)
dfa = dfa.append(dfa5)
dfa

df_p = pd.pivot_table(dfa, values='ID', index=['AcceptedCmp','Education'],
               columns=['Marital_Status'], aggfunc="count", fill_value=0, margins=True, margins_name="Total")
df_p

dfa['AcceptedCmp'].value_counts(normalize=True).sort_index()


d = df_p.reset_index()
d = d[d.columns[:-1]]
d = d.set_index('Education')
d

df_dif = pd.DataFrame()
for a in range(2,6):
    df_dif = df_dif.append(d[d['AcceptedCmp']==a]-d[d['AcceptedCmp']==1])
df_dif['AcceptedCmp'] = df_dif['AcceptedCmp'] + 1
df_dif.groupby(['AcceptedCmp', 'Education' ],).sum()

now = pd.Timestamp('now').year
age = now-dfa.Year_Birth
age

bins = pd.IntervalIndex.from_tuples([(17, 30), (30, 40), (40, 50), (50, 60), (60, 100)])
dfa['Age'] = pd.cut(age, bins)
dfa['Age']

dfa['Age'].value_counts().to_frame()

#Data Visualization

# analyze the average income received by different age groups of customers

ageIncome = dfa.groupby(['Age'])['Income'].mean().reset_index()
plt.figure(figsize=(12,5))
sns.barplot(data = ageIncome[['Age', 'Income']], x = "Age", y = "Income")
plt.xlabel('Age', size = 15)
plt.ylabel('Average Income', size = 15)
plt.title('Average Incomе for different groups of Age', color = 'red', size = 20)
plt.show()

plt.figure(figsize=(12,5))
sns.boxplot(x = "Age", y = "Income", data=dfa[(dfa.Income>dfa.Income.quantile(0.1))&(dfa.Income<dfa.Income.quantile(0.9))])
plt.xlabel('Age', size = 15)
plt.ylabel('Average Income', size = 15)
plt.title('Average Incomе for different groups of Age', color = 'red', size = 20)
plt.show()

plt.figure(figsize=(12,5))
sns.displot(data=dfa[(dfa.Income>dfa.Income.quantile(0.1))&(dfa.Income<dfa.Income.quantile(0.9))].reset_index(),
            x = "Income", hue='Age', kind='kde')
plt.xlabel('Average Income', size = 15)
plt.ylabel('Density', size = 15)
plt.title('Average Incomе for different groups of Age', color = 'red', size = 20)
plt.show()

plt.figure(figsize=(12,5))
sns.displot(data=dfa[(dfa.Income>dfa.Income.quantile(0.05))&(dfa.Income<dfa.Income.quantile(0.95))].reset_index(), 
            x = "Income", hue='Age', multiple='dodge')
plt.xlabel('Average Income', size = 15)
plt.ylabel('Count', size = 15)
plt.title('Average Incomе for different groups of Age', color = 'red', size = 20)
plt.show()

sns.displot(
    dfa[(dfa.Income>dfa.Income.quantile(0.05))&(dfa.Income<dfa.Income.quantile(0.95))].reset_index(), 
    x="Income", col="Age", row='Education')

size = dfa['Education'].value_counts(normalize=True)
plt.figure(figsize=(12,5))
plt.pie(size, shadow = True, autopct = "%.2f%%", labels=size.index)
plt.title('Educational')
plt.show()

df_pie = df_p.reset_index()
df_pie.Total = df_pie.Total.fillna(0)

def func(pct, allvals):
    absolute = int(np.round(pct/100.*np.sum(allvals)))
    return "{:.1f}%\n{:d}".format(pct, absolute)

for e in range(1, 6):
    plt.figure(figsize=(12,5))
    dt = df_pie[df_pie['AcceptedCmp']==e]
    exp = np.zeros(len(dt))
    exp[np.argmax(dt.Total)] = 0.1

    plt.pie(dt.Total, shadow = True,   explode=exp, 
            autopct=lambda pct: func(pct, dt.Total),
            textprops=dict(color="w"))

    plt.title('AcceptedCmp ' + str(e))
    plt.legend(dt.Education, title="Education",
          loc="center right", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.show()




