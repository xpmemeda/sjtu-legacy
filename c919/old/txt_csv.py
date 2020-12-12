import pandas
dataset = pandas.read_csv('20171017.txt',sep='\t',index_col=0)
dataset.to_csv('20171017.csv',index=False)