import json
fname1='/home/guoqiang/reid_competition/submission.json'
with open(fname1,'r') as f:
    data=json.load(f)
s1=set(data.keys())
fname2='/home/guoqiang/reid_competition/test_set/query_a_list.txt'
f=open(fname2)
s2=[]
for line in f.readlines():
    a=line.split(' ')[0]
    s2.append(a.split('/')[1])
s2=set(s2)
print(len(s2&s1))
