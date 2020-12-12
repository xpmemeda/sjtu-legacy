from pandas import read_csv, concat

data1 = read_csv("data1.txt", header=0)#header=0不添加猎头，index_col=0不添加索引
data2 = read_csv("data2.txt", header=0)
data3 = read_csv("data3.txt", header=0)
data4 = read_csv("data4.txt", header=0)
data = concat([data1, data2, data3, data4],axis=1)
data.to_csv("data.txt", index=False)#index=False不要索引，header=False不要列头