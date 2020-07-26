#!/usr/bin/env python

import numpy as np
from numpy import linalg
from pyscf import gto, ao2mo, scf, fci, dft, grad
from pyscf.dft import numint
from pyscf.geomopt.berny_solver import optimize
import sys
import os
import matplotlib.pyplot as plt

# Some input parameters, where obviously looping over dists is possible
basisname = "cc-pvdz" # reference basis set
dist = 0.7608986485 # equilibrium distance at this level of theory, verified!
coord = 'H 0 0 0; H 0 0 {0}'.format(str(dist))
verb = 0 # integer verbosity flag

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
mol_eq = optimize(mf)
#print(mol_eq.atom_coords())
dist = ( mol_eq.atom_coords()[0][2] + mol_eq.atom_coords()[1][2] ) * 0.529177 # bohr to angstrom conversion
mf = scf.RKS(mol_eq).run(xc = 'lda,vwn', verbose = 0 )
print(' At a distance R = {0} angstrom:'.format(dist))
print('DFT energy is now: {0} a.u.\n'.format(mf.e_tot)) 

