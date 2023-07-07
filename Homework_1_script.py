#The order of exercises is the one I followed to sole them, they are not divided into categories
#Since I've never studied XML and Regex, I haven't done them. I'll study them during this term and I'll update the git_repository

#Say Hello World with python_exercise_1
#controllo
print("Hello, World!")

#If-Else_exercise_2

import math
import os
import random
import re
import sys



n=int(input())
if n%2==0 and 2 <= n <= 5: 
    print("Not Weird")
elif n%2==0 and n>20:
     print("Not Weird")
elif n%2!=0:
     print("Weird")
elif n%2==0 and 6 <= n <= 20:
    print("Weird")

#Arithmetic Operators_exerise_3

a = int(input())
b = int(input())

print(a+b)
print(a-b)
print(a*b)

#Division_exercise_4


a = int(input())
b = int(input())
print(a//b)
print(a/b)

#Loops_exercise_5

n = int(input())
for i in range(n):
    print(i*i)

#Write a function_exercise_6

def is_leap(year):
    leap = False
    if(year%4 == 0) and (year%100 != 0) or (year%400 == 0):
     leap = True


#print a function_exercise_7


if __name__ == '__main__':
    n = int(input())
    s=''
    for i in range(1,n+1):
        s+=str(i)
    print(s)

#list Comprehension_exercise_8

x = int(input())
y = int(input())
z = int(input())
n = int(input())
l=[]
p=0
for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if i+j+k!=n:
                l.append([])
                l[p]=[i,j,k]
                p+=1

print(l)

#Find the Runner up Score_exercise_9

if __name__ == '__main__':
    i = int(input())
    lis = list(map(int,raw_input().strip().split()))[:i]
    z = max(lis)
    while max(lis) == z:
        lis.remove(max(lis))

    print max(lis)

#Lists_exrcise_10

if __name__ == '__main__':
    n = input()
    l = []
    for _ in range(n):
        s = raw_input().split()
        cmd = s[0]
        args = s[1:]
        if cmd !="print":
            cmd += "("+ ",".join(args) +")"
            eval("l."+cmd)
        else:
            print l

#Swap cases_exercise_11

def swap_case(s):
    import string
    
    
    return string.swapcase(s)

#String split and join_exercise_12

def split_and_join(line):
    line = line.split(" ")
    a = "-".join(line)
    return a

#What's your name?_exercise_13

def print_full_name(a, b):
    print (("Hello %s %s! You just delved into python.") % (a, b))

#Mutations_exercise_14

def mutate_string(string, position, character):
    return string[:int(position)] + character + string[int(position)+1:]

#Designer_door_mat_exercise_15

N,M = map(int,input().split())
for i in range(1, N, 2): 
    print (( str('.|.')*i ).center(M, '-'))
print(str('WELCOME').center(M, '-'))
for i in range(N-2, -1, -2): 
    print(( str('.|.')*i ).center(M, '-'))

#The Minion Game_exrcise_16

def minion_game(string):
    # your code goes here
    lun=len(string)
    k=0
    s=0
    for i in range(lun):
        if string[i] in "AEIOU":
            k += lun - i
        else:
            s += lun - i        
            
    if k > s:
        print('Kevin',k)
    elif k < s:
        print('Stuart',s)

#Introduction to Set_exercise_17

    def average(array):
    s=set(array)
    av = sum(s)/len(s)
    return av

#No idea!_exercise_18

nm =set(input().split())
arr = map(int,input().split()) #array
A = map(int,input().split()) #setA
B = map(int,input().split()) #setB
al=list(arr)
As=set(A)
Bs=set(B)

h=0

for i in al:
    
    if i in As:
        h=h+1
    if i in Bs:
        h=h-1

print(h)

#Symmetric Difference_exercise_19

n = int(input())
N = map(int, input().split())
m = int(input())
M = map(int, input().split())
Ns=set(N)
Ms=set(M)
a =list(Ns.union(Ms)-Ns.intersection(Ms))
a.sort()
for i in a:
    print(i)

#Set.add()_exercise_20

n = int(input())
s=set([])
for i in range(1,n):
    s.add(input())
print(len(s))

#Set discard, remove & pop_exercise_21

n = int(input())
s = set(map(int, input().split()))
ncomm = int(input())
for _ in range(ncomm):
    l = list(input().split())
    
    if(l[0]=='remove'):
        if int(l[1]) in s:
            s.remove(int(l[1]))
    elif(l[0]=='discard'):
        if int(l[1]) in s:
            s.discard(int(l[1]))
    else:
        s.pop()

print(sum(s))

#Set union operation_exercise_22

n = int(input())
A = set(input().split())
m = int(input())
B = set(input().split())
print(len(A.union(B)))

#Set intersection_exrcise_23

n = int(input())
A = set(input().split())
m = int(input())
B = set(input().split())
print(len(A.intersection(B)))

#Set difference_exrcise_24

n = int(input())
A = set(input().split())
m = int(input())
B = set(input().split())
print(len(A.difference(B)))

#Set symmetric difference operation_excise_25

n = int(input())
A = set(input().split())
m = int(input())
B = set(input().split())
print(len(A.symmetric_difference(B)))

#Set mutations_exercise_26


a = int(raw_input())
A = set(map(int, raw_input().split()))
for i in range(int(raw_input())):
    s, b = raw_input().split()
    if s == 'intersection_update':
        A &= set(map(int, raw_input().split()))
    elif s == 'update':
        A |= set(map(int, raw_input().split()))
    elif s == 'symmetric_difference_update':
        A ^= set(map(int, raw_input().split()))
    else:
        A -= set(map(int, raw_input().split()))
print sum(A)

#the captain room_exercise_27

n = int(raw_input())
arr = map(int, raw_input().split())
s=set(arr)
print(((sum(s)*n)-(sum(arr)))//(n-1))

#check subset_exercise_28

n1 = int(input())
for i in range(n1):
    n2 = int(input())
    arr = set(input().split())
    
    n3 = int(input())
    arr2 = set(input().split())
    print(not bool(arr.difference(arr2)))

#check strict superset_exercise_29

#not all the test have been good

arr = set(input().split())
n2 = int(input())
a = []
for i in range(n2):
    
    arr1 = set(input().split())
    
    
    a.append(arr>arr1)
i=0
for i in a:
    if i==True:
        i+=1
if i == n2:
    print(True)
else:
    print(False)

#collection counter_exercise_30

from collections import Counter
n = int(input())
size = list(map(int,input().split()))
cust = int(input())
items=(Counter(size).items())
earn=0
for i in range(cust):
    l = list(map(int,input().split()))
    if l[0] in size:
        earn+=l[1]
        size.remove(l[0])
print(earn)

#Collections order dict_exercise_31

from collections import OrderedDict
d = OrderedDict()
for _ in range(int(input())):
    item, space, quantity = input().rpartition(' ')
    d[item] = d.get(item, 0) + int(quantity)
for item, quantity in d.items():
    print(item, quantity)

#word order_exercise_32

from collections import Counter, OrderedDict
class OrderedCounter(Counter, OrderedDict):
    pass
d = OrderedCounter(input() for _ in range(int(input())))
print(len(d))
print(*d.values())


#Collection dequeue_exercise_33

from collections import deque
d = deque()
for _ in range(int(input())):
    inp = input().split()
    getattr(d, inp[0])(*[inp[1]] if len(inp) > 1 else [])
print(*[item for item in d])

#Pilling Up!_exercise_34

for t in range(input()):
    input()
    lst = map(int, raw_input().split())
    l = len(lst)
    i = 0
    while i < l - 1 and lst[i] >= lst[i+1]:
        i += 1
    while i < l - 1 and lst[i] <= lst[i+1]:
        i += 1
    print "Yes" if i == l - 1 else "No"

#Calendar Module_exercise_35

day = {0:'MONDAY', 1:'TUESDAY', 2:'WEDNESDAY', 3:'THURSDAY', 4:'FRIDAY', 5:'SATURDAY', 6:'SUNDAY'}

month,da,year = map(int,raw_input().split())
print day[calendar.weekday(year,month,da)]

#Exceptions_exercise_36

for i in range(int(input())):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

#Incorrect Regex_exercise_37

import re
for _ in range(int(input())):
    a = True
    try:
        reg = re.compile(input())
    except re.error:
        a = False
    print(a)

#Zipped_exercise_38

n, x = map(int, input().split()) 

l = []
for _ in range(x):
    l.append( map(float, input().split()) ) 

for i in zip(*l): 
    print( sum(i)/len(i) )

#GinortS_exercise_39

print(*sorted(input(), key=lambda c: (-ord(c) >> 5, c in '02468', c)), sep='')

#Arrays_exercise_40

def arrays(arr):
   return(numpy.array(arr[::-1], float))

#Shape and reshape_exercise_41

import numpy

print(numpy.reshape(numpy.array(input().split(),int),(3,3)))

#Transpose and flatten_exercise_42

import numpy as np

n, m = map(int, raw_input().split())
array = np.array([raw_input().strip().split() for _ in range(n)], int)
print (array.transpose())
print (array.flatten())

#Concatenate_exercise_43

import numpy as np
a, b, c = map(int,raw_input().split())
arrA = np.array([raw_input().split() for _ in range(a)],int)
arrB = np.array([raw_input().split() for _ in range(b)],int)
print(np.concatenate((arrA, arrB), axis = 0))

#Zero and ones_exercise_44

import numpy as np
nums = tuple(map(int, raw_input().split()))
print np.zeros(nums, dtype = np.int)
print np.ones(nums, dtype = np.int)

#Eye and identity_exercise_45

import numpy
print str(numpy.eye(*map(int,raw_input().split()))).replace('1',' 1').replace('0',' 0')

#Array Mathematics_exercise_46

import numpy as np
n, m = map(int, input().split())
a, b = (np.array([input().split() for _ in range(n)], dtype=int) for _ in range(2))
print(a+b, a-b, a*b, a//b, a%b, a**b, sep='\n')

#Floor ceil and rint_exercise_47

import numpy

numpy.set_printoptions(sign=' ')

a = numpy.array(input().split(),float)

print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))

#Sum and Prod_exercise_48

import numpy
N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)],int)
print(numpy.prod(numpy.sum(A, axis=0), axis=0))

#mean, var and std_exercise_49

import numpy as np 

n,m = map(int, input().split())
b = []
for i in range(n):
    a = list(map(int, input().split()))
    b.append(a)

b = np.array(b)

np.set_printoptions(legacy='1.13')
print(np.mean(b, axis = 1))
print(np.var(b, axis = 0))
print(np.std(b))

#min and max_exercise_50

import numpy as np
N, M = map(int, input().split())
print(np.array([input().split() for _ in range(int(N))], int).min(1).max())

#Dot and cross_exercise_51

from __future__ import print_function
import numpy
n=int(raw_input())
a = numpy.array([raw_input().split() for _ in range(n)],int)
b = numpy.array([raw_input().split() for _ in range(n)],int)
m = numpy.dot(a,b)
print (m)

#Inner and outer_exercise_52

import numpy
a = numpy.array(raw_input().split(), int)
b = numpy.array(raw_input().split(), int)
print numpy.inner(a, b), "\n", numpy.outer(a, b)

#Polynomials_exercise_53

import numpy as np

print(np.polyval(list(map(float,input().split())),float(input())))

#Linear Algebra_exercise_53

import numpy as np
n=int(input())
a=np.array([input().split() for _ in range(n)],float)
np.set_printoptions(legacy='1.13') #important
print(np.linalg.det(a))

#Birthday Cake Candles_exercise_54

def birthdayCakeCandles(ar):
    max = ar[0]
    count = 0
    n = len(ar)
    for i in range(n):
        if ar[i] > max:
            max = ar[i]
    for i in range(n):
        if ar[i] == max:
            count+=1
    return count

#Viral Advertising_execise_55

def viralAdvertising(n):
    m = 2
    tot = 2
    for _ in range(1, n):
        m += m>>1
        tot += m
    return tot

#Ricursive digit sum_exercise_56

def superDigit(n, k):
    f=list(map(int,n))
    x=sum(f)
    x=x*k

    
    if(x>10):
        return superDigit(str(x),1)
    else:
        return x

#Insertion Sort1_exercise_57

def insertionSort1(n, arr):
    e=arr[len(arr)-1]
    ar=arr[0:len(arr)-1]
    l=len(ar)
    last=l-1
    parr=[]
    while last>-1: # 2 4 6 8
        tmp=ar[last]
        if tmp<e:
            break
        parr=ar[0:last]+[tmp]+ar[last:l]     
        
        print (" ".join(str(s) for s in parr))
        last=last-1
    parr[last+1]=e
    print (" ".join(str(s) for s in parr))

#Insertion Sort2_exercise_58

for i in range(1,len(a)):
    key=a[i]
    j=i-1
    while j>=0 and a[j]>key:
        a[j+1]=a[j]
        j=j-1
    a[j+1]=key
    print(*a)

#Kangaroo_exercise_59

def kangaroo(x1, v1, x2, v2):
    k1 = x1;
    k2 = x2;
    if x2>x1 and v2>v1:
        fptr.write("NO")
    
    else:
        for i in range(10000):
            k1+=v1
            k2+=v2
            if k1==k2:
                fptr.write("YES")
                exit(0)
                

            
    
        fptr.write("NO")

#Athlete Sort_exercise_60

if __name__ == '__main__':
    n, m = map(int, input().split())
    nums = [list(map(int, input().split())) for i in range(n)]
    k = int(input())

    nums.sort(key=lambda x: x[k])

    for line in nums:
        print(*line, sep=' ')

#Time Delta_exercise_61

if __name__ == '__main__':
    

    fmt = '%a %d %b %Y %H:%M:%S %z'
    for _ in range(int(input())):
        time1 = dt.strptime(input(), fmt)
        time2 = dt.strptime(input(), fmt)
        print(int(abs((time1 - time2).total_seconds())))

#Nested List_exercise_62

if __name__ == '__main__':
        marksheet=[]
        scorelist=[]
        for _ in range(int(input())):
                name = input()
                score = float(input())
                marksheet+=[[name,score]]
                scorelist+=[score]
        b=sorted(list(set(scorelist)))[1] 

        for a,c in sorted(marksheet):
             if c==b:
                    print(a)

#Finding Percentage_exercise_63

if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
    query_scores = student_marks[query_name]
    print("{0:.2f}".format(sum(query_scores)/(len(query_scores))))
    

#Tuples_exercise_64

if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    t=tuple(integer_list)
    print hash(t)


#Find a string_exercise_65

def count_substring(string, sub_string):
    count=0
    
    for i in range(0, len(string)-len(sub_string)+1):
        if string[i] == sub_string[0]:
            flag=1
            for j in range (0, len(sub_string)):
                if string[i+j] != sub_string[j]:
                    flag=0
                    break
            if flag==1:
                count += 1
    return count

#String Validators_exercise_67

if __name__ == '__main__':
    s = input()
    for test in ('isalnum', 'isalpha', 'isdigit', 'islower', 'isupper'):
        print(any(eval("c." + test + "()") for c in s))

#Text Alignment_exercise_68

thickness = int(raw_input())  
c = 'H'

# Top Cone
for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))

# Top Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) +
          (c * thickness).center(thickness * 6))

# Middle Belt
for i in range((thickness + 1) / 2):
    print((c * thickness * 5).center(thickness * 6))

# Bottom Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) +
          (c * thickness).center(thickness * 6))

# Bottom Cone
for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c +
          (c * (thickness - i - 1)).ljust(thickness)).rjust(thickness * 6))


#Alphabet Rangoli_exercise_69

import string
def print_rangoli(n):
    
    alpha = string.ascii_lowercase

    
    L = []
    for i in range(n):
        s = "-".join(alpha[i:n])
        L.append((s[::-1]+s[1:]).center(4*n-3, "-"))
    print('\n'.join(L[:0:-1]+L))

#Capitalize_exercise_70

def solve(s):
    
    for x in s[:].split():
        s = s.replace(x, x.capitalize())
    return s

#Merge the tools_exercise_71

def merge_the_tools(S, N):
     
    for part in zip(*[iter(S)] * N):
        d = dict()
        print(''.join([ d.setdefault(c, c) for c in part if c not in d ]))

#Default Dict_exercise_72

from collections import defaultdict

d = defaultdict(list)
list1=[]

n, m = map(int,raw_input().split())

for i in range(0,n):
    d[raw_input()].append(i+1) 

for i in range(0,m):
    list1=list1+[raw_input()]  

for i in list1: 
    if i in d:
        print " ".join( map(str,d[i]) )
    else:
        print -1

#Collections named tuple()_exercise_73

from collections import namedtuple

(n, categories) = (int(input()), input().split())
Grade = namedtuple('Grade', categories)
marks = [int(Grade._make(input().split()).MARKS) for _ in range(n)]
print((sum(marks) / len(marks)))


#Map and Lambda function_exercise_74

cube = lambda x: x ** 3

def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        yield a
        a, b = b, a + b

#Standardize Mobile Number Using Decorators_exercise_75

def wrapper(f):
    def fun(l):
        f('+91 {} {}'.format(n[-10:-5], n[-5:]) for n in l)
    return fun


#Decorators 2 - Name Directory_exercise_76

import operator

def age(x):
    return int(x[2])

def person_lister(f):
    def inner(people):
        return map(f,sorted(people, key=age))
    return inner
            

  




