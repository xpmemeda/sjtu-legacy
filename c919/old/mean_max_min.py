from pandas import read_csv,DataFrame
dataset=read_csv('dataset.csv',header=0,index_col=0)
a=DataFrame()
a['mean']=dataset.mean()
a['max-min']=dataset.apply(lambda x:x.max()-x.min())
print(a)