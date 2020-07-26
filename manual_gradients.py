#!/usr/bin/env python

import numpy as np
from numpy import linalg
from pyscf import gto, ao2mo, scf, fci, dft, grad
from pyscf.dft import numint
import sys
import os
import matplotlib.pyplot as plt

# Some input parameters, where obviously looping over dists is possible
basisname = "cc-pvdz" # reference basis set
dist = 0.7608986485 # equilibrium distance at this level of theory, verified!
coord = 'H 0 0 0; H 0 0 {0}'.format(str(dist))
verb = 0 # integer verbosity flag
step = 0.001 # should be adapted and so on but whatever


# Do a mean-field KS-DFT calculation at dist
mol = gto.Mole(atom=coord, basis=basisname,charge=0,spin=0) # build a molecule, neutral species, spin is N\alpha-N\beta here 
mol.build()
mf = scf.RKS(mol) # run DFT calculation as reference
mf.verbose = verb # reduce output
mf.xc = 'lda,vwn' # we select an approximate xc functional, many options available e.g. 'pbe,pbe' or 'b3lyp'
edft = mf.kernel() # get DFT energy
nelec = mol.nelec # number of electrons
print('The number of electrons is now {0}'.format(nelec))
print(' At a distance R = {0} angstrom:'.format(dist))
print(' DFT energy: {0} a.u.\n'.format(edft)) # total energies
gradients = grad.RKS(mf).kernel() # We calculate analytical nuclear gradients, we choose the level of theory of the gradients
grad_atom_1 = gradients[0][2]
grad_atom_2 = gradients[1][2]
print(' In the cartesian frame, gradient for atom 1 is {0}, gradient for atom 2 is {1}'.format(grad_atom_1,grad_atom_2))
totgrad = -1*grad_atom_1 + grad_atom_2
print(' With respect to dist, the gradient is {0}'.format(totgrad))


# We could build a newton-rhapson or some other scheme for minimization here

if totgrad < 1e-6 : 
   dist = dist - step*totgrad
   coord = 'H 0 0 0; H 0 0 {0}'.format(str(dist))
   mol = gto.Mole(atom=coord, basis=basisname,charge=0,spin=0) # build a molecule, neutral species, spin is N\alpha-N\beta here 
   mol.build()
   mf = scf.RKS(mol).run(verbose=0,xc='LDA,VWN') # run DFT calculation
   print(' At a distance R = {0} angstrom:'.format(dist))
   print(' New DFT energy: {0} a.u.\n'.format(mf.e_tot)) # total energies
elif totgrad > 1e-6 :
   dist = dist + step*totgrad
   coord = 'H 0 0 0; H 0 0 {0}'.format(str(dist))
   mol = gto.Mole(atom=coord, basis=basisname,charge=0,spin=0) # build a molecule, neutral species, spin is N\alpha-N\beta here 
   mol.build()
   mf = scf.RKS(mol).run(verbose=0,xc='LDA,VWN') # run DFT calculation
   print(' At a distance R = {0} angstrom:'.format(dist))
   print(' New DFT energy: {0} a.u.\n'.format(mf.e_tot)) # total energies


