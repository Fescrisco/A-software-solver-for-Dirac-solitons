import random
import numpy as np
import scipy as sc
import sympy as sympy
from sympy.matrices import Matrix, eye, zeros, ones, diag
import math as math
import Parser
from Parser import Interpreter
from sympy import Function, Symbol, Derivative
from sympy import Array, simplify
from sympy import I, sqrt
from sympy import cos, sin, symbols
from sympy import expand, conjugate
from sympy.physics.matrices import msigma
from sympy.diffgeom import metric_to_Ricci_components, TensorProduct
from sympy.diffgeom import Manifold, Patch, CoordSystem
from sympy.solvers import solve
from sympy import latex
from sympy import exp, log
import functools

from sympy.interactive import printing
printing.init_printing(use_latex=True)


def save(expr, path = 'exprCart2.txt'):
    f = open(path, "w")
    print(sympy.srepr(expr), file = f)
    f.close

def load(path = 'expr.txt'):
    f = open(path, "r")
    l = f.read()
    return sympy.sympify(l)

@functools.lru_cache(maxsize = 5)
def createAlpha(index, dim):
    base = zeros(dim)
    if (index == 0):
        for i in range(dim):
            base[i, i] = 1
    if (index == 1):
        for i in range(dim):
            base[dim-1-i, i] = 1
    elif (index == 2):
        f = 1
        for i in range(dim):
            f = f*-1
            base[i,dim-1-i] = I * f
    elif (index == 3):
        val = -1
        for i in range(dim):
            val = val*-1
            if (i < dim/2):
                base[i, 2+i] = 1*val
            else:  
                base[i, 1-(dim-1-i)] = 1*val
    return base

def spinorRot(psi, dim):
    spinorRot = eye(dim)
    for i in range(round(dim/2),dim):
        spinorRot[i,i] = -1
    return (psi.H * spinorRot).subs(conjugate(1/r),1/r)

#------------------------------------------------------------
cDim = 4
fDim = 4
#--------------------------------------------------------------
@functools.lru_cache(maxsize = 5)
def gamma(index: int):
    return beta*createAlpha(index, fDim)

x = []
for i in range(cDim):
    x.append(Symbol('x'+str(i), real = True))

def nab(i,j):
    if (i == j and i == 0):
        return -1
    elif (i==j):
        return 1
    else:
        return 0


array = []
arrayE = []
arrV = []
for i in range(cDim):
    innerArrE = []
    innerArr = []
    innerV = []
    for j in range(fDim):
        gsymb = 0
        esymb = 0
        vsymb = Function('v'+str(i)+str(j), real = True)(x[1], x[2], x[3])
        esymb = Function('e'+str(i)+str(j), real = True)(x[1], x[2], x[3])
        if (i <= j):
            gsymb = Symbol('g'+str(i)+str(j))
        else:
            gsymb = Symbol('g'+str(j)+str(i))
            #esymb = Function('e'+str(j)+str(i), real = True)(x[1], x[2])
        if (i != j):
            0
            #gsymb = 0
        innerArr.append(gsymb)
        innerArrE.append(esymb)
        innerV.append(vsymb)
    array.append(innerArr)
    arrayE.append(innerArrE)
    arrV.append(innerV)
            
vieb = Matrix(arrayE)
t = vieb
#t[0,1] = 0
#t[0,2] = 0
#t[0,3] = 0
#t[2,1] = Function('e21', real = True)(x[1], x[2])
#t[3,1] = Function('e31', real = True)(x[1], x[2])
#t[3,2] = Function('e32', real = True)(x[1], x[2])
vieb = t
print("vieb")
print(vieb)
print()
nab = Matrix(fDim,fDim, nab)
print("Nab")
print(nab)
print()

metric = vieb*nab*vieb.transpose()
lowV = vieb*nab

print("Metric")
print(metric)
print('inv vieb')
invV = Matrix(arrV)
print(invV)
print()
print("|g|")
metDet = metric.det()
print(metDet)
print()
#print(simplify(sqrt(-metDet)))

beta = zeros(fDim)
for i in range(fDim):
    if(i >= fDim/2):
        beta[i, i] =  -1
    else:
        beta[i, i] =  1
        


@functools.lru_cache(maxsize = 5)
def gammaComm(a: int, b: int):
    return gamma(a)*gamma(b) - gamma(b)*gamma(a)

@functools.lru_cache(maxsize = 1024)
def viebTri(a: int, b: int, c: int, mu: int, beta:int):
    inner = lowV[mu,a].diff(x[beta])-lowV[beta,a].diff(x[mu])
    return invV[mu,c]*invV[beta,b]*inner
    #return simplify(invV[mu,c]*invV[beta,b]*inner)

@functools.lru_cache(maxsize = 64)
def vT3(a: int, b: int, c: int):
    runsum = 0
    for mu in range(cDim):
        for beta in range(cDim):
            runsum += viebTri(a,b,c,mu,beta)
            #runsum += simplify(viebTri(a,b,c,mu,beta))
    return runsum
#------------------------------------------------------------------
r = sqrt(x[1]**2 + x[2]**2 + x[3]**2)
sigmaR = msigma(3)*(x[3]/r) + msigma(1)*(x[1]/r)
sigmaR = sigmaR + msigma(2)*(x[2]/r)
e1 = Matrix([[1],[0]])
e2 = Matrix([[0],[1]])
        
        
a,b = symbols('a,b', cls = Function, real = True)
a = a(x[1], x[2], x[3])#/r
b = b(x[1], x[2], x[3])#/r
u = a  #*(x[1])*sqrt(sqrt(metric[0,0]))
v = -I*b  #*(x[1])*sqrt(abs(sqrt(metric[0,0])))
#v = b
        
topvar  = u*e1
botvar  = sigmaR*v*e1
topvar2  = u*e2
botvar2  = sigmaR*v*e2
w = Symbol('w', real = True)
    
psi = (cos(w*x[0]) - I*sin(w*x[0]))*Matrix([[topvar[0]],[topvar[1]],[botvar[0]],[botvar[1]]])
psi2 = (cos(w*x[0]) - I*sin(w*x[0]))*Matrix([[topvar2[0]],[topvar2[1]],[botvar2[0]],[botvar2[1]]])

# Convert fsy solution states to cartesian then use to test that it is stable
#
#
#
#-------------------------------------------------------------------
print()
print("Spinor")
print(str(psi))
print()

print("Conjugate spinor")
print(str(spinorRot(psi, fDim)))
print()
add1 = 0
add2 = 0
add3 = 0
adds = [add1, add2, add3]

def DirAction(psi, adds):
    left = spinorRot(psi, fDim)
    final = left @ (- Symbol('m')*psi)
    print('m term')
    print(final)
    adds[0] += final[0]
    print()
    der1 = zeros(4, 1)
    print("Calculating Der terms")
    for i in range(cDim):
        print(i)
        for j in range(fDim):
            print(j)
            der1 += (I*invV[i,j]*gamma(j)) @ psi.diff(x[i])
    temp = left @ der1
    adds[1] += temp[0]
    #print('der term')
    #print(temp)
    #print()
    final += temp
    print("Calculating Correction terms")
    der2 = zeros(4, 1)
    for a in range(fDim):
        print(a)
        for b in range(fDim):
            print(b)
            for c in range(fDim):
                print(c)
                inner = vT3(a,b,c) + vT3(c,b,a) + vT3(b,c,a)
                der2 += ((I/2)*gamma(c)*inner*gammaComm(a,b)) @ psi
    #print('corr term')
    temp = left @ der2
    adds[2] += temp[0]
    #print(temp)
    #print()
    final += temp
    
    return expand(final), adds

def DirOP(psi, adds):
    final = (- Symbol('m')*psi)
    print('m term')
    print(final)
    adds[0] += final[0]
    print()
    der1 = zeros(4, 1)
    for i in range(cDim):
        for j in range(fDim):
            der1 += (I*invV[i,j]*gamma(j)) @ psi.diff(x[i])
    temp = der1
    adds[1] += temp[0]
    #print('der term')
    #print(temp)
    #print()
    final += temp
    
    der2 = zeros(4, 1)
    for a in range(fDim):
        for b in range(fDim):
            for c in range(fDim):
                inner = vT3(a,b,c) + vT3(c,b,a) + vT3(b,c,a)
                der2 += ((I/2)*gamma(c)*inner*gammaComm(a,b)) @ psi
    #print('corr term')
    temp = der2
    adds[2] += temp[0]
    #print(temp)
    #print()
    final += temp
    
    return expand(final), adds

diracAction, adds = DirAction(psi, adds)
#-------------
dA2, adds = DirAction(psi2, adds)
diracAction = dA2 + diracAction
#-------------------


print("Total m terms")
t1 = expand(adds[0])
print(t1)
print()
t1 = t1.subs(I,0)
print(t1)
print()
#t1 = simplify(t1)
#print(t1)
print()
print("Total der terms")
t2 = expand(adds[1])
print(t2)
print()
t2 = t2.subs(I,0)
print(t2)
print()
#t2 = simplify(t2)
#print(t2)
print()
print("Total correction terms")
t3 = expand(adds[2])
print(t3)
print()
t3 = t3.subs(I,0)
print(t3)
print()
#t3 = simplify(t3)
#print(t3)
print()

#print("Add and simplify all")
#print(simplify(t1+t2+t3))
print('saving langrangian')
print()
diracAction = diracAction.subs(I, 0)
diracAction = diracAction*sqrt(-metDet)
#diracAction = simplify(diracAction)
save(diracAction)
print(diracAction)



# Manually symmeterize matrix