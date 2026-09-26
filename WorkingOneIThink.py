import scipy as sc
import numpy as np
np.bool = np.bool_
import math
from scipy.optimize import minimize
import numdifftools as nd
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp
from functools import lru_cache
import datetime


"""
Take a look at images to use
Look to the conformal case for how asymptotic behaviour is expected

Possible later implementations
Mixed derivative handling
# Look to tensor notation and sympy tensor application
# Take a look at tensor
# py bvp recheck for boundary value problem solvers
# Jax, autodiff - Check packages and see
# Check that errors can be defined as easily as with symp
# Check that conversion from field to grid of field points is viable?

# Take a look at contacting Herbert Fruchtl regarding question for computer
# Option for running clusters at

Two fermion solutions in seperation/brought together
Potentially of heidrero metric go to applying a uniform magnetic field


Cost function modification:
    - 1/(m * r) of node, Scaling on furtherst node

Potential things to optimize,
Remove if statement calls to derterms for rolling

"""

a = 1

initialSize = 5
currSize = 5
maxIter = 1

zSym = sp.Symbol('z')
conversion = -sp.log((1/zSym)-1)

#dataPath = "FSY2rnpy"
dataPath = "FSYTESTRUNS"
load = True
MaxIter = 10000

def symbuild(function, derivative_terms, variables):
    expression = sp.diff(function, *derivative_terms)
    f = sp.lambdify(variables, expression, "numpy")
    return f


def genDerivative(fields, conditions):
    if (len(conditions) != 3):
        raise ValueError("2nd Argument should be a tuple of index, order, dz")
    field = fields[conditions[0]]
    name = field.name
    # sF is before f[i], lf is after f[i]
    axis = conditions[2]
    sF = sp.Symbol(name + ",{" + str(axis) + "-1}")
    lF = sp.Symbol(name + ",{" + str(axis) + "+1}")
    if (conditions[1] == 1):
        return ([field - sF, lF - field])/(dz[0]), [sF, lF] ,[-1, +1]
    elif (conditions[1] == 2):
        return (lF + sF - 2*field)/(dz[0]**2), [sF, lF] ,[-1, +1]
    elif (conditions[1] == -1):
        raise ValueError("Integral not yet implemented")
    else:
        raise ValueError("Not implemented for derivatives higher than 2nd")

#--------------------------------------------------------------------
# Compare derivative values
# Make weight around minimizing deviations of w based of each equation
# Staggerd grid: Derivatives from normal grid, field poitns as average
#--------------------------------------------------------------------

def makeFields(names):
    if (len(names) == 1):
        return [sp.Symbol(names[0])]
    else:
        fieldList = []
        for name in names:
            fieldList.append(sp.Symbol(name))
        return fieldList

def makeIndexes(derConds, fields):
    arr = []
    for i in range(len(fields)):
        arr.append(None)
    for i in range(len(derConds)):
        arr[derConds[i][0]] = i
    return arr


"""
- Robin boundary conditions on FSY, 1d problem along R
- Robin boundary conditions 'delete' the matter terms and look at large R behaviour
- 2d particle in the box -> 2d  harmonic oscillator
- 3d fsy

- pdb library
- 'movies' of the process being visible
- Send meeting invitation Wensday noon
- Change T as an integration variable to the log of the T-1

- Axial solution (Heidrero metric, up in fsy), 'Req Her metric into FSY metric form'
- Metric  - 
- Potential analogue to sigma matrix choice for FSY spinor to simplify problem
- H 

- Conformal solution (second derivative stabalizes)

[b,____,b]
[b, b,_____,b,b]
Compact differencing


------------------------------------------------------------------------
"""
gridDim = 1
# Define axis sizes
sizes = np.array([20])
jumps = np.ones(len(sizes))
for i in range(len(jumps) - 2, -1, -1):
    jumps[i] = sizes[i+1]*jumps[i+1]

# Grids are uniform in z_n but can take a coordinate transform to x
zMin = np.array([0])
zMax = np.array([1])
# Coordinate transformation for each coordinate


# None if no coordinate transformation is needed
vals = np.linspace(zMin[0], zMax[0], sizes[0]+2)
valT = np.linspace(zMin[0], zMax[0], 2*sizes[0] + 3)[1::2]

coordsNames = ["x0"]
coords = makeFields(coordsNames)
cvals = [vals[1:-1]]
scaleF = 0.1

# r =  sc * z / sqrt(1 - z**2)
# r =  sc * z / sqrt(1 - z**2)
# z = r /(1/m + r)
# z/(m - zm)  = r

t1 = scaleF*coords[0]/sp.sqrt(1- coords[0]*coords[0])
t1 = coords[0]/(scaleF - scaleF*coords[0])

c1 = sp.simplify(1/sp.diff(t1, coords[0]))
radiusF = sp.lambdify(coords[0], t1, "numpy")
changeGrid = sp.lambdify(coords[0], c1, "numpy")
tC = sp.Symbol("z,r")
dzdr1 = sp.Symbol("z,r-")
dzdr2 = sp.Symbol("z,r+")
print(c1)

rvals = radiusF(vals)
print(rvals)
rS = radiusF(valT)
r1 = rS[0:-1]
r2 = rS[1::]

tMap = changeGrid(vals)
tMap2 = changeGrid(valT)
print(tMap2)
tMap = tMap[1:-1]
transforms = [None]
cChange = sp.Symbol('c')
# Specify if the coordinates loop on themselves (IE azimuthal axis in polar)
Cyclic = [False]
# 
sizeAdj = [1]

dz = (zMax - zMin)/(sizes + 1)

fieldNames = ["A","T","a","b"]
fields = makeFields(fieldNames)
# Boundry values for fields (either constant or actual boundary values)
#T0 = 1.17778
T0 = 1.14534
#T0 = 1.16306
#T0 = 1.17778
"""
rA' = - A + 1 , Set this as boundary condition
2ArT'/T - A + 1 = 0, Set this 
"""


# Change from dz to a list of axis for mixed derivatives
# In order - Field, Order, axis
derivativeFields = [(0, 1, 0),(1, 1, 0), (2, 1, 0), (3, 1, 0)]
# change sF to dfm and lF to dfp
dTerms = []
dSymb = []
dPos = []
for field in derivativeFields:
    dF, ds, dp = genDerivative(fields, field)
    dSymb.append(ds)
    dTerms.append(dF)
    dPos.append(dp)
dIndexes = makeIndexes(derivativeFields, fields)

scale = 1

constnames = ["w"]
const = makeFields(constnames)

# sum of the product of the initial grid sizes
sys = np.zeros(len(fields)*np.prod(sizes) + len(constnames))

def genAvg(index: int, fw):
    return (fields[index] + dSymb[index][fw])/2



#m = 0.961562
m = 0.533874
#m = 0.777865
#m = 1.11467

w = m - const[0]**2
kappa = 2
G = 1
subVal = 8*np.pi*G*abs(kappa)
#---------------
r1s = sp.Symbol('r-')
r = r1s
A = genAvg(0, 0)
alp = genAvg(2, 0)
beta = genAvg(3, 0)
sqA = sp.sqrt(A)
T = genAvg(1, 0)

ders = []
for i in range(len(dTerms)):
    derEx = dTerms[i]
    exp = derEx[0]*dzdr1
    ders.append(exp)

dA, dT, dalp, dbeta = ders


ab2 = alp**2 + beta**2
cr1 = sp.Symbol('l-')
scaleUn = 1*cr1

eq1 = ((sqA*dalp) - (kappa*alp/(2*r)) + beta*(m + T*w))*scaleUn
eq2 = ((sqA*dbeta) - alp*(T*w - m) + (kappa*beta/(2*r)))*scaleUn
eq3 = ((r*dA + A - 1) + (subVal*(T**2)*w*(ab2)))*scaleUn
eq4 = (((2*A*r*dT/T) - A + 1) + subVal*T*(T*w*(ab2) - 2*alp*beta/r - m*(alp**2 - beta**2)))*scaleUn

er1 = [eq1, eq2, eq3, eq4]
#----------------

r2s = sp.Symbol('r+')
r = r2s
A = genAvg(0, 1)
alp = genAvg(2, 1)
beta = genAvg(3, 1)
sqA = sp.sqrt(A)
T = genAvg(1, 1)

ders = []
for i in range(len(dTerms)):
    derEx = dTerms[i]
    exp = derEx[1]*dzdr2
    ders.append(exp)

dA, dT, dalp, dbeta = ders


ab2 = alp**2 + beta**2
cr2 = sp.Symbol('l+')
scaleUn = 1*cr2

eq1 = ((sqA*dalp) - (kappa*alp/(2*r)) + beta*(m + T*w))*scaleUn
eq2 = ((sqA*dbeta) - alp*(T*w - m) + (kappa*beta/(2*r)))*scaleUn
eq3 = ((r*dA + A - 1) + (subVal*(T**2)*w*(ab2)))*scaleUn
eq4 = (((2*A*r*dT/T) - A + 1) + subVal*T*(T*w*(ab2) - 2*alp*beta/r - m*(alp**2 - beta**2)))*scaleUn

er2 = [eq1, eq2, eq3, eq4]

#===============

r = coords[0]
A = fields[0]
alp = fields[2]
beta = fields[3]
sqA = sp.sqrt(A)
T = fields[1]
ab2 = alp**2 + beta**2

gridIntErrors = [(((dz[0]/tC*4*np.pi*T*(ab2)/sqA) - (1/sizes[0]))*(scale))*scaleUn]
errors = [er1[0],er2[0],er1[1],er2[1],er1[2],er2[2],er1[3],er2[3]]

splices = []
indStarts = []
for i in range(len(fields)):
    indStarts.append(i*np.prod(sizes))
    splices.append(slice(i*np.prod(sizes), np.prod(sizes)*(i+1)))
for i in range(len(const)):
    indStarts.append(i*np.prod(-len(const) + i))
    splices.append(-len(const) + i)
    
size = np.prod(sizes)

"""
-------------------------------------------------------------------------
Initial Values go in this section
"""
varArgs = (*fields, *np.concatenate(dSymb), *const, *coords, dzdr1, dzdr2, tC, cr1, cr2, r1s, r2s)

print(vals)
print(np.cos((np.pi/2)*vals[1:-1]))

# A
sys[splices[0]] = np.ones(np.prod(sizes))
# T
sys[splices[1]] = np.linspace(T0, 1, sizes[0]+2)[1:-1]
# alpha
#sys[splices[2]] = (cvals[0]/10)*np.exp(-0.1*cvals[0])*np.sin(2*np.pi*vals[1:-1])
sys[splices[2]] = (cvals[0]/10)*np.exp(-0.1*cvals[0])*np.cos((np.pi/2)*vals[1:-1])
# beta
sys[splices[3]] = (cvals[0]/100)*sys[splices[2]]#*np.sin(2*np.pi*vals[1:-1])
# root(w)
sys[-1] = 0.01


'''
sys[splices[0]] = [0.99949, 0.997754, 0.994447, 0.989181, 0.981565, 0.971281, 0.958216, \
0.942686, 0.925704, 0.909228, 0.896131, 0.889572, 0.891657, 0.902016, \
0.917654, 0.934722, 0.950748, 0.96511, 0.977964, 0.989533]
sys[splices[1]] = [1.14519, 1.14465, 1.14362, 1.14195, 1.13946, 1.13593, 1.13111, \
1.12476, 1.11666, 1.10673, 1.09513, 1.08231, 1.06905, 1.05625, \
1.0446, 1.0344, 1.02558, 1.01792, 1.0112, 1.00527]
sys[splices[2]] = [0.00681604, 0.01427, 0.0223511, 0.0309969, 0.0400606, 0.0492669, \
0.058152, 0.0659984, 0.0717869, 0.0742288, 0.0719798, 0.0641395, \
0.0509901, 0.0346002, 0.01865, 0.00700987, 0.00140848, 0.0000804297, \
0, 0]
sys[splices[3]] = [0.0000431079, 0.000189686, 0.000468913, 0.000913234, 0.00155497, \
0.00241908, 0.00350967, 0.0047881, 0.00614292, 0.00736109, 0.0081271, \
0.00809438, 0.00705616, 0.00515264, 0.0029387, 0.00115275, \
0.000239268, 0.0000140227, 0, 0]
sys[-1] = np.sqrt(0.0164738)
'''
'''
sys[splices[0]] = [0.997715, 0.990372, 0.978107, 0.962787, 0.948295, 0.939611, 0.94014, \
0.948215, 0.955753, 0.953255, 0.942057, 0.939249, 0.948031, 0.939032, \
0.933441, 0.932523, 0.950744, 0.96511, 0.977964, 0.989533]
sys[splices[1]] = [1.17703, 1.17455, 1.17003, 1.16341, 1.15504, 1.14573, 1.13664, \
1.12881, 1.12248, 1.11678, 1.11031, 1.10285, 1.09579, 1.08872, \
1.07959, 1.07024, 1.02558, 1.01792, 1.0112, 1.00527]
sys[splices[2]] = [0.00961294, 0.0194221, 0.0283493, 0.0348507, 0.0370187, 0.0329604, \
0.0215136, 0.00319776, -0.0187471, -0.0372802, -0.0413249, \
-0.0204644, 0.0209652, 0.0485539, 0.01252, -0.0567508, 0, 0, 0, 0]
sys[splices[3]] = [0.000203751, 0.000871458, 0.00203938, 0.00363316, 0.00541562, \
0.00697557, 0.00779141, 0.0073494, 0.00526535, 0.00149545, \
-0.00311704, -0.00625941, -0.00506101, 0.000876967, 0.00539773, \
0.00126533, 0, 0, 0, 0]
sys[-1] = np.sqrt(0.0604386)
'''
'''
sys[splices[0]] = [0.998896, 0.995207, 0.988452, 0.978433, 0.965526, 0.951015, 0.937314, \
0.927746, 0.925472, 0.93145, 0.942077, 0.948336, 0.93975, 0.916409, \
0.899247, 0.90697, 0.928105, 0.96511, 0.977964, 0.989533]
sys[splices[1]] = [1.16271, 1.16152, 1.15928, 1.15574, 1.15073, 1.14414, 1.13608, \
1.12696, 1.11749, 1.10856, 1.10083, 1.09422, 1.08746, 1.07839, \
1.06575, 1.05137, 1.03804, 1.01792, 1.0112, 1.00527]
sys[splices[2]] = [0.008139, 0.0168567, 0.0258501, 0.0346043, 0.0423217, 0.0478679, \
0.0497819, 0.0464261, 0.0363412, 0.0188042, -0.00547485, -0.0334364, \
-0.058246, -0.0684289, -0.0541545, -0.0245294, -0.00453294, 0, 0, 0]
sys[splices[3]] = [0.000100577, 0.000438637, 0.00106577, 0.00201787, 0.00329302, \
0.00482131, 0.00643463, 0.00785566, 0.00873092, 0.00871056, \
0.00752983, 0.00505356, 0.00142153, -0.00237547, -0.00407388, \
-0.00258048, -0.000575182, 0, 0, 0]
sys[-1] = np.sqrt(0.0354872)    
'''
initT = sys[splices[1]]
initA = sys[splices[0]]
inital = sys[splices[2]]
initbet = sys[splices[3]]

if (load):
    sys = np.load(dataPath+".npy")


#print(sys[splices[0]][-3::])
#print(sys[splices[1]][-3::])
#print(sys[splices[2]][-3::])
#print(sys[splices[3]][-3::])


bounds = [[1,1],[T0,1],[0,0],[0,0]]


"""
-------------------------------------------------------------------------
"""

errfuncs = []
interrfuncs = []
for i in errors:
    errfuncs.append(sp.lambdify(varArgs, i, "numpy"))
for i in gridIntErrors:
    interrfuncs.append(sp.lambdify(varArgs, i, "numpy"))

def vecTval(v, vec: bool):
    if (isinstance(v, np.ndarray)):
        if (not vec):
            return np.sum(v)
    else:
        if (vec):
            return np.ones(size)*v
    return v

#@lru_cache(maxsize = 4*len(errfuncs))
def getErr(args, index: int):
    return vecTval(errfuncs[index](*args), True)

def ErrRMS(err):
    return np.sqrt(err/(size + 1))

    
#@lru_cache(maxsize = 4*len(interrfuncs))
def getIntErr(args, index: int):
    return np.sum(vecTval(interrfuncs[index](*args), True))

fullslice = []
for i in sizes:
    fullslice.append(slice(i))

boundaryTemplate = []
for i in range(len(sizes)):
    bsize = np.delete(sizes, i)
    boundaryTemplate.append(np.ones(bsize))

testfield = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]])
testsizes = [2,2,3]


def boundV(index: int, axis: int, forward = True):
    if (forward):
        return bounds[index][1] * boundaryTemplate[axis]
    return bounds[index][0] * boundaryTemplate[axis]

def fillComp(size, index1S, index1E, index2S, index2E ,vals):
    temp = np.zeros((size, size))
    for i in range(len(index1S)):
        s1 = slice(index1S[i],index1E[i])
        s2 = slice(index2S[i],index2E[i])
        np.fill_diagonal(temp[s1,s2], vals[i])
        
    return temp


    

def cutField(field, fieldInf, fieldInf2, fill = True):
    # field inf has (derpos, deraxis)
    size: int = len(field)
    rS: int = 0
    cS: int = 0
    rE: int = rS + size
    cE: int = cS + size
    rChange: int = int(jumps[fieldInf[1]]*fieldInf[0])
    cChange: int = int(jumps[fieldInf2[1]]*fieldInf2[0])
    rSign = np.sign(rChange)
    cSign = np.sign(cChange)

    sliceOff = 0
    if(rSign*cSign == -1):
        if (rSign == 1):
            sliceOff = slice(-cChange, size -rChange)
        else:
            sliceOff = slice(-rChange, size -cChange)
        loss = abs(rChange - cChange)
        rE -= loss
        cS += loss
        field = field[sliceOff]
    else:
        tot: int = size - abs(rChange + cChange)
        if (rChange == cChange):
            tot:int = size - abs(rChange)
        if (rSign == 1 or cSign == 1):
            sliceOff = slice(0, tot)
            rS += rChange
            cS += cChange
            rE = rS + tot
            cE = cS + tot
        else:
            sliceOff = slice(size - tot, size)
            rE += rChange
            rS = rE - tot
            cE += cChange
            cS = cE - tot
        field = field[sliceOff]
    
    newJumps = [size, *jumps]
    newSize = len(field)
    inds = newJumps[fieldInf[1]]*(1+np.arange(newSize//newJumps[fieldInf[1]]))
    inds = -jumps[fieldInf[1]] + inds
    for i in inds:
        field[i:i+jumps[fieldInf[1]]] = np.zeros(jumps[fieldInf[1]])
    inds = newJumps[fieldInf2[1]]*(1+np.arange(newSize//newJumps[fieldInf2[1]]))
    inds = -jumps[fieldInf2[1]] + inds
    for i in inds:
        field[i:i+jumps[fieldInf2[1]]] = np.zeros(jumps[fieldInf2[1]])
    out = np.zeros((size, size))
    np.fill_diagonal(out[rS:rE,cS:cE], field)
    if (fill):
        return out
    else:
        return np.diag(out)

def genRolls(field, i: int, axis: int):
    index = fullslice
    before = np.roll(field, 1,  axis)
    after = np.roll(field, -1, axis)
    index[axis] = 0
    ind = tuple(index)
    before[ind] = boundV(i, axis, False)
    index[axis] = -1
    ind = tuple(index)
    after[ind] = boundV(i, axis, True)
    return before, after

def addBounds(field, i: int):
    for ax in range(len(sizes)):
        valEnd = boundV(i, ax, True)
        valStart = boundV(i, ax, False)
        field = np.append(field, valEnd, axis = ax)
        field = np.append(valStart, field, axis = ax)
    return field


def getArgs(sys):
    f = []
    flen = len(fields)
    for i in range(flen):
        field = sys[splices[i]]
        f.append(field)
    for i in range(len(dSymb)):
        index = derivativeFields[i][0]
        axis = derivativeFields[i][2]
        fm, fp = genRolls(f[index], index, axis)
        f.append(fm)
        f.append(fp)
    for i in range(len(const)):
        f.append(sys[splices[flen+i]])
    for i in range(len(sizes)):
        f.append(rvals[1:-1])
    f.append(tMap2[0:-1])
    f.append(tMap2[1::])
    f.append(tMap)
    correction = np.ones(sizes[0])/np.sqrt(2)
    correction[-1] = 1
    f.append(correction)
    correction[0] = 1/np.sqrt(2)
    correction[-1] = 1
    f.append(correction)
    f.append(r1)
    f.append(r2)
    return f

def argAt(args, index: int):
    f = []
    for i in range(len(args)-len(const)):
        f.append(args[i][index])
    for i in range(len(args)-len(const),len(args)):
        f.append(args[i])
    return f
    
# How to make field into grid?
# Fields have associated positional value
# Turn error into a function that works out error at every position

# How to get derivatives of numeric diff components?
# If on staggered grid then formular for nth derivative looks like
# [xb, x0, x1, x2, x3 ,x4, xB]
# [x1-x0/d1, x2-x1/d2, x3-x2/d3, x4+x2/d4]
# [-x_b + (x1-x0/d1) / d0,-(x1-x0/d1)+ x2-x1/d2 /(d.15), ...,xB- x4+x2/d4]/df]
# For now assume const d
# [-x_b + x1 - x0,..., x(i+1) - 2*xi + x(i-1), ...,xB - x4 + x3]/d^2
# Generate numeric form of errors for all involved components


testarg = getArgs(sys)

print("Check Normalization")
run = 0
print(getIntErr(testarg, 0)**2)
print(np.sum(interrfuncs[0](*testarg))**2)
print("Error in alpha")
vec = getErr(testarg, 0)**2
print(np.sum(vec))
print(np.sum(errfuncs[0](*testarg)**2))
print("Error in beta")
vec = getErr(testarg, 1)**2
print(np.sum(vec))
print(np.sum(errfuncs[1](*testarg)**2))
print("Error in A")
vec = getErr(testarg, 2)**2
print(np.sum(vec))
print(np.sum(errfuncs[2](*testarg)**2))
print("Error in T")
vec = getErr(testarg, 3)**2
print(np.sum(vec))
print(np.sum(errfuncs[3](*testarg)**2))

'''
print("Error in w")
print(np.sum(errfuncs[4](*testarg)**2))
'''
print("Function value to start")


testf = sys[0:np.prod(sizes)]
behind = np.roll(testf, 1)
behind[0] = 0
infront = np.roll(testf, -1)
infront[-1] = 0

left = (sys[-1]**2)*testf
right = (- 2*testf + (behind + infront))/dz**2


def f_dir(sys):
    args = getArgs(sys)
    runsum = 0
    for i in range(0, len(errors), 1):
        vec = getErr(args, i)**2
        runsum += np.sum(vec)
    funcval = runsum + getIntErr(args, 0)**2
    #print("func val")
    #print(funcval)
    return funcval
def f_dir2(sys):
    args = getArgs(sys)
    runsum = 0
    for i in range(len(errors)):
        vec = getErr(args, i)**2
        runsum += np.sum(vec)
    funcval = runsum + getIntErr(args, 0)**2
    return funcval

print(f_dir(sys))
'''
sys[splices[0]] = 2 - sys[splices[0]]
print(f_dir(sys))
'''
# --------------------------------------------------------------------------------------------------
#he = nd.Hessian(f_dir2)
#testhess = he(sys)
#print("Test hes col")
#print(testhess[-1])
# --------------------------------------------------------------------------------------------------

def buildDiag(errors, fields, varArgs):
    diag = []
    der = []
    for e in errors:
        temp = []
        tempD = []
        for f in fields:
            d = symbuild(e**2, (f, f), varArgs)
            temp.append(d)
        for f in dSymb:
            for s in f:
                d = symbuild(e**2, (s, s), varArgs)
                tempD.append(d)
        diag.append(temp)
        der.append(tempD)
    return [diag, der]

def buildIntDiag(errors, fields, varArgs):
    diags = []
    s = sp.Symbol("Int")
    for e in errors:
        comp = []
        for f in fields:
            d = sp.diff(e, f)
            dd = sp.diff(d, f)
            func = sp.lambdify((s, *varArgs), 2*(s*dd + d**2), "numpy")
            comp.append(func)
        diags.append(comp)
    return diags


diagsB = buildDiag(errors, fields+const, varArgs)
diagsBI = buildIntDiag(gridIntErrors, fields+const, varArgs)
err = getIntErr(testarg, 0)


def diagVals(funcs, funcsI, args, length):
    runsum = np.zeros(length)
    for i in range(len(funcs[0])):
        e = funcs[0][i]
        arr = np.array([])
        for j in range(len(e)):
            val = e[j](*args)
            val = vecTval(val, j < len(fields))
            arr = np.append(arr, val)
        runsum += arr
        runIndex = 0
        for k in range(len(dSymb)):
            index = splices[derivativeFields[k][0]]
            for z in range(len(dSymb[k])):
                f = funcs[1][i][runIndex]
                val = vecTval(f(*args), True)
                if (dPos[k][z] > 0):
                    ran = slice(-dPos[k][z], len(val))
                else:
                    ran = slice(0,-dPos[k][z])
                val[ran] = np.zeros(abs(dPos[k][z]))
                val = np.roll(val, 1*dPos[k][z])
                runsum[index] += val
                runIndex += 1
            
    for i in range(len(funcsI)):
        arr = np.array([])
        err = getIntErr(args, i)
        
        for j in range(len(funcsI[i])):
            val = funcsI[i][j](err, *args)
            val = vecTval(val, j < len(fields))
            arr = np.append(arr, val)
        runsum += arr
        
    return runsum

'''
print("Test diag gen")
print("Calculated values")
outd = diagVals(diagsB, diagsBI, testarg, len(sys))
print(outd)
print("Expected values")
print(np.diagonal(testhess))
'''


def buildCross(errors, symbols, varArgs):
    cross = []
    cross2 = []
    for e in errors:
        temp = []
        temp2 = []
        for i in range(len(symbols)):
            derBox = []
            for j in range(i+1, len(symbols)):
                block1 = []
                block2 = []
                block3 = []
                d = symbuild((e**2), (symbols[i], symbols[j]), varArgs)
                temp.append(d)
                bol1 : bool = i < len(dIndexes) and dIndexes[i] is not None
                bol2 : bool = j < len(dIndexes) and dIndexes[j] is not None
                if(bol1):
                    for s in dSymb[dIndexes[i]]:
                        d = symbuild(e**2, (s, symbols[j]), varArgs)
                        block1.append(d)
                    
                if(bol2):
                    for s2 in dSymb[dIndexes[j]]:
                        d = symbuild(e**2, (symbols[i], s2), varArgs)
                        block2.append(d)
                        
                if(bol1 and bol2):
                    for s in dSymb[dIndexes[i]]:
                        for s2 in dSymb[dIndexes[j]]:
                            d = symbuild(e**2, (s, s2), varArgs)
                            block3.append(d)
                derBox.append([block1, block2, block3])
            temp2.append(derBox)
        cross.append(temp)
        cross2.append(temp2)
    return [cross, cross2]


def buildIntCross(errors, fields, varArgs):
    cross = []
    s = sp.Symbol("Int")
    for e in errors:
        errc = []
        for i in range(len(fields)):
            comp1 = []
            comp2 = []
            d = sp.diff(e, fields[i])
            comp1 = sp.lambdify(varArgs, d, "numpy")
            for j in range(i+1, len(fields)):
                dd = sp.diff(d, fields[j])
                func = sp.lambdify((s, *varArgs), 2*s*dd, "numpy")
                comp2.append(func)
            errc.append([comp1, comp2])
        cross.append(errc)
    return cross

crossB = buildCross(errors, fields+const, varArgs)
crossI = buildIntCross(gridIntErrors, fields+const, varArgs)

def crossVals(funcs, funcsI, args, length):
    runsum = np.zeros((length, length))
    lenf : int = len(fields)
    lenT : int = lenf + len(const)
    
    fieldD = []
    constD = []
    
    for e in range(len(funcsI)):
        zf = np.zeros((lenf, sizes[0]))
        zc = np.zeros(lenT-lenf)
        for i in range(lenf):
            zf[i] = funcsI[e][i][0](*args)
        for i in range(lenT - lenf):
            zc[i] = np.sum(funcsI[e][i+lenf][0](*args))
        fieldD.append(zf)
        constD.append(zc)
    
    
    funcIndex = 0
    for i in range(lenf):
        splice1 = splices[i]
        ind1 = dIndexes[i]
        
        for j in range(i+1, lenf):
            ind2 = dIndexes[j]
            splice2 = splices[j]
            for e in range(len(funcs[0])):
                # Diagonal of non-diagonal matrix block
                diags = funcs[0][e][funcIndex](*args)
                # use IND 2 for der on der
                
                funcs[1][e][i][j-i-1]
                f1 =  funcs[1][e][i][j-i-1][0]
                for derp1 in range(len(f1)):
                    dvals = f1[derp1](*args)
                    dvals = vecTval(dvals, True)
                    dp = dPos[ind1][derp1]
                    runsum[splice1, splice2] += cutField(dvals, [dp, 0], [0, 0])
                
                f2 =  funcs[1][e][i][j-i-1][1]
                for derp1 in range(len(f2)):
                    dvals = f2[derp1](*args)
                    dvals = vecTval(dvals, True)
                    dp = dPos[ind1][derp1]
                    runsum[splice1, splice2] += cutField(dvals, [0, 0], [dp, 0]) 
                    
                runsum[splice1, splice2] += np.diag(diags)
            for e in range(len(funcsI)):
                #If non diagonal 2nd der is 0
                err = getIntErr(args, e)
                # This is partial by 2*pi*pj (a block of matrix)
                runsum[splice1, splice2] += 2*np.outer(fieldD[e][i], fieldD[e][j])
                # This is partial 2*s*dd (Only on diagonal)
                diags = funcsI[e][i][1][j-i-1](err, *args)
                diags = vecTval(diags, True)
                runsum[splice1, splice2] += np.diag(diags)
            funcIndex += 1
        for j in range(lenf, lenf + len(const)):
            splice2 = splices[j]
            # Line vectors corresponding to field v const
            for e in range(len(funcs[0])):
                dfuncs = funcs[1][e][i][j-i-1][0]
                line = funcs[0][e][funcIndex](*args)
                for s in range(len(dfuncs)):
                    valad = dfuncs[s](*args)
                    valad = vecTval(valad, True)
                    ran = None
                    if (dPos[ind1][s] > 0):
                        ran = slice(-dPos[ind1][s], len(valad))
                    else:
                        ran = slice(0,-dPos[ind1][s])
                    valad[ran] = np.zeros(abs(dPos[ind1][s]))
                    valad = np.roll(valad, 1*dPos[ind1][s])
                    line += valad
                runsum[splice1, splice2] += line
            for e in range(len(funcsI)):
                err = getIntErr(args, e)
                runsum[splice1, splice2] += 2*fieldD[e][i]*constD[e][j-lenf]
                runsum[splice1, splice2] += funcsI[e][i][1][j-i-1](err, *args)
            funcIndex += 1
            
    for i in range(lenf, lenf + len(const) - 1):
        splice1 = splices[i]
        for j in range(i+1, lenf + len(const)):
            splice2 = splices[j]
            for e in range(len(funcs)):
                runsum[splice1,splice2] += np.sum(funcs[0][e][funcIndex](*args))
            for e in range(len(funcsI)):
                err = getIntErr(args, e)
                val1 = 2*constD[e][i-lenf]*constD[e][j-lenf]
                val2 = funcsI[e][i][1][j-(i+1)](err, *args)
                runsum[splice1, splice2] += np.sum(val2) + val1
            funcIndex += 1
            
    return runsum
'''
print("Test off diagonal sections")
out1 = crossVals(crossB, crossI, testarg, len(sys))
print("Calculated const cross terms")
print(out1[:,-1])
print("Expected cont cross terms")
t = testhess[:,-1]
print(t)
print(abs(t - out1[:,-1])/(t+out1[:,-1]))
'''

def buildInterCross(errors, fields, const, varArgs):
    cross = []
    for e in errors:
        temp = []
        for i in range(len(dSymb)):
            dcont = []
            temp2 = []
            fIndex = derivativeFields[i][0]
            for s in range(len(dSymb[i])):
                d = symbuild(e**2, (fields[fIndex], dSymb[i][s]), varArgs)
                dcont.append(d)
                for j in  range(s+1, len(dSymb[i])):
                    d = symbuild(e**2, (dSymb[i][s], dSymb[i][j]), varArgs)
                    shift = dPos[i][s]-dPos[i][j]
                    rem = [d, shift, i]
                    if (dPos[i][s] < 0):
                        rem.append([abs(dPos[i][j]), 0])
                    else:
                        rem.append([0, abs(dPos[i][j])])
                    if (dPos[i][j] > 0):
                        rem.append([abs(dPos[i][j]), 0])
                    else:
                        rem.append([0, abs(dPos[i][j])])
                    temp2.append(rem)
            dcont.append(temp2)
            temp.append(dcont)
        cross.append(temp)
    return cross



def buildIntInter(errors, fields, varArgs):
    temp = sp.Symbol('q¬+')
    funcs = []
    for e in errors:
        tempfuncs = []
        for f in fields:
            part = symbuild(e, [f], varArgs)
            tempfuncs.append(part)
        funcs.append(tempfuncs)
    return funcs

interB = buildInterCross(errors, fields, const, varArgs)
interI = buildIntInter(gridIntErrors, fields + const, varArgs)

def interVals(funcs, funcsI, args, length):
    runsum = np.zeros((length, length))
    lenf : int = len(fields)
    lenT : int = lenf + len(const)
    for e in range(len(errors)):
        for f in range(len(dSymb)):
            fIndex = derivativeFields[f][0]
            index = splices[fIndex]
            for s in range(len(dSymb[f])):
                diag = funcs[e][f][s](*args)
                diag= vecTval(diag, True)
                ran = None
                if (dPos[f][s] > 0):
                    ran = slice(0, -dPos[f][s])
                else:
                    ran = slice(-dPos[f][s], len(diag))
                diag = diag[ran]
                runsum[index, index] += np.diag(diag, dPos[f][s])
            for k in range(len(funcs[e][f][len(dSymb[f])])):
                arr = funcs[e][f][len(dSymb[f])][k]
                vals = vecTval(arr[0](*args), True)
                shift = arr[1]
                index = splices[arr[2]]
                vals = vals[arr[3][0]::]
                vals[0:arr[3][1]] = np.zeros(arr[3][1])
                vals = vals[0:-arr[4][0]]
                vals[len(vals)-arr[4][1]::] = np.zeros(arr[4][1])
                runsum[index, index] += np.diag(vals, shift)
            
    for e in range(len(funcsI)):
        for f in range(len(fields)):
            index = splices[f]
            line = vecTval(funcsI[e][f](*args), True)
            runsum[index, index] += np.outer(line, line)
    return runsum

out = interVals(interB, interI, testarg, len(sys))
out = out + np.transpose(out)
'''
print("Calculated off index")
print(out[splices[1],splices[1]])
print("Expected")
print(testhess[splices[1],splices[1]])
'''
'''
print("Bulk comp Calc")
print(out[0])
outd1 = np.append(outd, 0)
testbulk = testhess - out1 - np.diag(outd)
print("Expected")
print(testbulk[0])
print()
'''
#testmat = np.fromfunction(lambda i, j: cf3(field[i], field[j], sys[-1]), (20,20), dtype=int)
# 
# Convert sys into input matrices to pass to hess func
call_count = 0
def hes(sys):
    hessi = np.zeros((len(sys), len(sys)))
    args = getArgs(sys)
    diag = diagVals(diagsB, diagsBI, args, len(sys))
    hessi += crossVals(crossB, crossI, args, len(sys))
    hessi += interVals(interB, interI, args, len(sys))
    hessi += np.transpose(hessi)
    np.fill_diagonal(hessi, diag)
    return hessi


def makeJac(errors, intErr, vArgs):
    funcs = []
    for e in errors:
        temp = []
        for f in fields:
            temp.append(symbuild(e**2, [f], vArgs))
        for c in const:
            temp.append(symbuild(e**2, [c], vArgs))
        for d in dSymb:
            temp2 = []
            for s in d:
                temp2.append(symbuild(e**2, [s], vArgs))
            temp.append(temp2)
        funcs.append(temp)
            
    funcs2 = []
    for e in intErr:
        temp = []
        for f in fields:
            temp.append(symbuild(e, [f], vArgs))
        for c in const:
            temp.append(symbuild(e, [c], vArgs))
        funcs2.append(temp)
    return funcs, funcs2

jfuncs = makeJac(errors, gridIntErrors, varArgs)
#ja = nd.Jacobian(f_dir2)

def jacob(sys):
    lenf = len(fields)
    lent = len(const) + lenf
    
    args = getArgs(sys)
    jacobi = np.zeros(len(sys))
    
    funcs1, funcs2 =  jfuncs
    for i in funcs1:
        for j in range(lenf):
            temp = vecTval(i[j](*args), True)
            jacobi[splices[j]] += temp
        for j in range(lenf, lent):
            jacobi[splices[j]] += np.sum(vecTval(i[j](*args), True))
        for j in range(lent, lent + len(dSymb)):
            ind1 = j-lent
            for k in range(len(dSymb[ind1])):
                valad = vecTval(i[j][k](*args), True)
                ran = None
                if (dPos[ind1][k] > 0):
                    ran = slice(-dPos[ind1][k], len(valad))
                else:
                    ran = slice(0,-dPos[ind1][k])
                valad[ran] = np.zeros(abs(dPos[ind1][k]))
                valad = np.roll(valad, 1*dPos[ind1][k])
                jacobi[splices[derivativeFields[ind1][0]]] += valad
                
    for i in range(len(funcs2)):
        interr = getIntErr(args, i)
        for j in range(lenf):
            temp = vecTval(2*interr*funcs2[i][j](*args), True)
            jacobi[splices[j]] += temp
        for j in range(lenf, lent):
            jacobi[splices[j]] += np.sum(vecTval(2*interr*funcs2[i][j](*args), True))
    '''
    print("Comparing differences in estimated Jacobian vs Calculated")
    t1 = ja(sys)[0]
    print(t1)
    print(jacobi)
    print(abs(t1-jacobi)/((t1+jacobi)/2))
    print(t1-jacobi)
    print(sys)
    '''
    return jacobi


#print()
#print("PRE-RUN")
#print("Calculated Jacobian")
#print(jacob(sys))



#def jj(sys):
    #t2 = ja(sys)[0]
    #t1 = jacob(sys)
#    return t2
#he = nd.Hessian(f_dir)

#def hh(sys):
    #args = getArgs(sys)
    #h1 = hes(sys)
    #h2 = he(sys)
    #print(sys)
    #print("Percentage difference in hessian")
    #print((abs(h1 - h2)/((h1+h2))/2)[2])
    #return h2
    
"""
print("Expected jac")
print(ja(sys)[0])
t1 = jacob(sys)
t2 = ja(sys)[0]
print("Percentage difference")
print(abs(t1 - t2)/((t1+t2))/2)
print()
"""

"""
print("Calc Hess matrix")
h1 = hes(sys)
h2 = he(sys)
print(h1)
print("Approx Hess Matrix")
print(h2)
print("Percentage difference in hessian")
print((abs(h1 - h2)/((h1+h2))/2))
print()
"""
'''
start = datetime.datetime.now()

out = minimize(f_dir, sys, method = 'BFGS', options={'disp': True})
totalBFGS = datetime.datetime.now() - start
'''
'''
mat1 = hes(sys)
print(mat1[-1])
f5 = plt.figure()
a5 = f5.add_subplot(111)
cax = a5.matshow(np.log10(np.abs(mat1)))
f5.colorbar(cax)

mat2 = testhess
print(mat2[-1])
f6 = plt.figure()
a6 = f6.add_subplot(111)
cax = a6.matshow(np.log10(np.abs(mat2)))
f6.colorbar(cax)

mat3 = mat1-mat2
print(mat3[-1])
f7 = plt.figure()
a7 = f7.add_subplot(111)
cax = a7.matshow(np.log10(np.abs(mat3)))
f7.colorbar(cax)
'''
print("Start Iteration")
start = datetime.datetime.now()
MaxIter = 1000
out = minimize(f_dir, sys, method = 'Newton-CG', jac=jacob, hess = hes, options={'maxiter': MaxIter,'disp': True, 'xtol' : 1e-20})
totalNCG =  datetime.datetime.now() - start

print("Iteration time: " + str(totalNCG))
res = out.x
#np.save(dataPath, res)

#print("Time comparison for system of: " + str(len(sys)))
#print("BFGS: " + str(totalBFGS))
#print("BFGS: " + str(totalBFGS))
#fig, ax = plt.subplots(1, subplot_kw={"projection": "3d"}, constrained_layout = True)
fig, ax = plt.subplots(2,2, constrained_layout = True)
args = getArgs(res)
A = args[0]**2
T = args[1]

   
exactA = [0.998293, 0.992715, 0.983019, 0.969959, 0.955734, 0.943984, 0.938716, \
0.941896, 0.950566, 0.956296, 0.951071, 0.939941, 0.941279, 0.948214, \
0.933222, 0.939002, 0.911977, 0.96511, 0.977964, 0.989533]
exactT = [1.17722, 1.17536, 1.1719, 1.16668, 1.15975, 1.15148, 1.14266, \
1.13427, 1.12711, 1.12117, 1.11543, 1.10862, 1.10095, 1.09384, \
1.08575, 1.07576, 1.06396, 1.01792, 1.0112, 1.00527]
exactAlp = [0.00895563, 0.0182867, 0.0272637, 0.0347749, 0.0393042, 0.0390564, \
0.0323393, 0.0182605, -0.00230105, -0.025474, -0.0429583, -0.041882, \
-0.0118667, 0.0354395, 0.048148, -0.020702, -0.0654146, 0, 0, 0]
exactBeta = [0.000163682, 0.000706472, 0.00168241, 0.00308286, 0.00479021, \
0.00653915, 0.00791553, 0.00841331, 0.00752971, 0.00488407, \
0.000510976, -0.0044089, -0.00695022, -0.00396983, 0.00337789, \
0.00505624, -0.00190375, 0, 0, 0]

exacts = [exactA, exactT, exactAlp, exactBeta]
inits = [initA, initT, inital, initbet]

plotdata = []
for i in range(4):
    plotdata.append(np.append(bounds[i][0], np.append(args[i], bounds[i][1])))
    exacts[i] = np.append(bounds[i][0], np.append(exacts[i], bounds[i][1]))
    inits[i] = np.append(bounds[i][0], np.append(inits[i], bounds[i][1]))
    
ax[0,0].plot(vals, plotdata[0])
ax[0,0].plot(vals, inits[0])
#ax[0,0].plot(vals, exacts[0])
ax[0,0].set_title("A")
ax[1,0].plot(vals, plotdata[1])
ax[1,0].plot(vals, inits[1])
#ax[1,0].plot(vals, exacts[1])
ax[1,0].set_title("T")
ax[0,1].plot(vals, plotdata[2])
ax[0,1].plot(vals, inits[2])
#ax[0,1].plot(vals, exacts[2])
ax[0,1].set_title("alpha")
ax[1,1].plot(vals, plotdata[3])
ax[1,1].plot(vals, inits[3])
#ax[1,1].plot(vals, exacts[3])
ax[1,1].set_title("beta")


#print(res)
print("Input m")
print(m)
print("Final w")
print(m - res[-1]**2)
print("Final norm")
print(np.sum(interrfuncs[0](*args))**2)
print(plotdata[1])