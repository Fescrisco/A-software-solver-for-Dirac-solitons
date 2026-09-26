# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:52:27 2025

@author: fescr
"""

import numpy as np
import scipy as sc
import sympy as sp
from sympy.matrices import Matrix, eye, zeros, ones, diag
import math as math
import Parser
from Parser import Interpreter
from sympy import Function, Symbol, Derivative
from sympy import Array, simplify
from sympy import sin, I, sqrt
from sympy import cos, sin, symbols
from sympy import expand
from sympy.physics.matrices import msigma
from sympy.diffgeom import metric_to_Ricci_components, TensorProduct
from sympy.diffgeom import Manifold, Patch, CoordSystem
from sympy.solvers import solve
from sympy import latex
from sympy import exp, log

import time
import functools

x = sp.symbols('x0 x1 x2 x3', real = True)

r = x[1]**2 + x[2]**2 + x[3]**2
rs = sp.symbols('r', real = True, positive = True)
print(x[0])
def load(path = 'expr.txt'):
    f = open(path, "r")
    l = f.read()
    return sp.sympify(l)


cT = load('T.txt')
eT = load('EinT.txt')


# dr/dx = x/r
# At x = y = z = 1
# x/r = 1/Sqrt[3]

# df/dx = df/dr * dr/dx /3 = df/dr / 3*Sqrt(3)
# alpha, beta, A, T
fields = np.array([0.0231946, 0.000505483, 0.994013, 1.14349])
fieldDerR = np.array([0.0241099, 0.00106582, -0.012627, -0.00396237])
fSecDR = np.array([-0.0030903, 0.00103731, -0.0123104, -0.00409386])
# if second d^2r/dx^2 = (y^2 + z^2) / r^3
# if mixed d^2r/dx dy = -xy/r^3 
# d^2f/dxi dxj = df/dr dr^2/dxi dxj + d^2f/dr^2 dr/dxi dr/xj
# d^2f/dxi dxj = df/dr dr^2/dxi dxj + d^2f/dr^2 / 3
# i == j -> d2fd/dxi^2 = df/dr (2/3*Sqrt[3]) + d^2f/dr^2 / 3
# i != j -> d2fd/dxi dxj = df/dr (-1/3*Sqrt[3]) + d^2f/dr^2 / 3
fieldD = fieldDerR/(3*np.sqrt(3))
fieldDD = fieldDerR*(2/(3*np.sqrt(3))) + fSecDR/3
fmixed = -fieldDerR*(1/(3*np.sqrt(3))) + fSecDR/3
print(fieldD)
print(fieldDD)
print(fmixed)

A = Function('A', real = True, positive = True)(x[1], x[2], x[3])
T = Function('T', real = True, positive = True)(x[1], x[2], x[3])
a = Function('a', real = True)(x[1], x[2], x[3])
b = Function('b', real = True)(x[1], x[2], x[3])
As, Ts, aS, bS = symbols('A T a b')
DA, DT, Da, Db = symbols('DA DT Da Db')
DDA, DDT, DDa, DDb = symbols('DDA DDT DDa DDb')
MA, MT, Ma, Mb = symbols('MA MT Ma Mb')
m = Symbol('m')
w = Symbol('w')

readableS1 = []
readableS2 = []

for i in range(1,4):
    readableS1.append((Derivative(As,x[i]),Symbol('DA')))
    readableS1.append((Derivative(Ts,x[i]),Symbol('DT')))
    readableS1.append((Derivative(aS,x[i]),Symbol('Da')))
    readableS1.append((Derivative(bS,x[i]),Symbol('Db')))
    for j in range(1,4):
        if (i==j):
            readableS2.append((Derivative(As,(x[i],2)),Symbol('DDA')))
            readableS2.append((Derivative(Ts,(x[i],2)),Symbol('DDT')))
            readableS2.append((Derivative(aS,(x[i],2)),Symbol('DDa')))
            readableS2.append((Derivative(bS,(x[i],2)),Symbol('DDb')))
        if (i!=j):
            readableS2.append((Derivative(As,x[i],x[j]),Symbol('MA')))
            readableS2.append((Derivative(Ts,x[i],x[j]),Symbol('MT')))
            readableS2.append((Derivative(aS,x[i],x[j]),Symbol('Ma')))
            readableS2.append((Derivative(bS,x[i],x[j]),Symbol('Mb')))
            
secondset = []
secondset.append((A,As))
secondset.append((T,Ts))
secondset.append((a,aS))
secondset.append((b,bS))

def subs(tensor):
    tensor = tensor.subs(secondset)
    tensor = tensor.subs(readableS2)
    tensor = tensor.subs(readableS1)
    return tensor

Es = subs(eT)
SE = subs(cT)
fit = Es-8*sp.pi*SE


varArgs = (aS, bS, As, Ts, Da, Db, DA, DT, DDa, DDb, DDA, DDT, Ma, Mb, MA, MT, m, w, x[1], x[2], x[3])

entries = [[0,0],[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]]
totaltime = 0
for i in entries:
    test = fit[i[0],i[1]]
    Ein = Es[i[0],i[1]]
    StressE = SE[i[0],i[1]]
    fun = sp.lambdify(varArgs, test)
    left = sp.lambdify(varArgs, Ein)
    right = sp.lambdify(varArgs, -8*sp.pi*StressE)
    start = time.time()
    output = fun(*fields, *fieldD, *fieldDD, *fmixed, 0.533874, 0.499274, 1, 1, 1)
    end = time.time()
    c1 = left(*fields, *fieldD, *fieldDD, *fmixed, 0.533874, 0.499274, 1, 1, 1)
    c2 = right(*fields, *fieldD, *fieldDD, *fmixed, 0.533874, 0.499274, 1, 1, 1)
    runtime = end-start
    totaltime += runtime
    print("Tensor entry: " +str(i[0])+','+str(i[1]))
    print(runtime)
    print("Equation value")
    print(output)
    print("Actual Values (E vs 8 Pi T)")
    print(c1)
    print(-c2)
    print("Percentage difference")
    print(100*abs(c1+c2)/((c1-c2)/2))
    print()

print("Total function runtimes")
print(totaltime)



print()
print()
print(fit[0,0])
print()
print()
expand(fit[0,0])
simplify(fit[0,0])
print(fit[0,0])
print()
print()
simplify(fit[1,1])
print(fit[1,1])
print()
print()
simplify(fit[1,2])
print(fit[1,2])
print()
print()
simplify(fit[1,3])
print(fit[1,3])
print()
print()
simplify(fit[2,3])
print(fit[2,3])
print()
print()
simplify(fit[2,2])
print(fit[2,2])
print()
print()
simplify(fit[3,3])
print(fit[3,3])

# Numerical solver, save hessian function
# Call r its own variable then substitute later

# Suppress argument
# Derive(A) -> DA(1,2)
# Check order of derivation pushed to numerical order

# Double check numerical solver can do calculations of mixed derivatives

# Check derivative calculations in numerical solver are approximately accurate at chosen grid size