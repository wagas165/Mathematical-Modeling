import torch
import numpy as np
def origin(string):
    a=[]
    for x in string:
        a.append(ord(x)-65)
    return a

def encoding(string):
    a=[]
    for x in string:
        a.append(ord(x)+3)
    return a

def decoding(list):
    string=''
    for x in list:
        if x!=35:
            string+=chr(((x-65-3)%26)+65)
        else:
            string+=' '
    return string

def hill_3(string):
    x=[]
    y=[]
    z=[]
    A=torch.tensor(np.array([[1,2,4],[4,7,6],[6,10,5]]))
    for i in range(0,len(string)//3):
        x.append(origin(string)[3*i:3*i+3])
    for vec in x:
        y.append(list(np.array(torch.mv(A,(torch.tensor(vec)))%26)))
    return y

print(hill_3('CRYPTOGRAPHY'))
print(decoding(encoding('AHSI HAYUIO')))
