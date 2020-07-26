#!/usr/bin/env python

import numpy as np
from numpy import linalg
from pyscf import gto, ao2mo, scf, fci, dft
from pyscf.dft import numint
import sys
import os
import matplotlib.pyplot as plt

# Some input parameters, where obviously looping over dists is possible
basisname = "cc-pvdz" # reference basis set
dist = 0.7608986485 # equilibrium distance at this level of theory, verified!
coord = 'H 0 0 0; H 0 0 {0}'.format(str(dist))
verb = 0 # integer verbosity flag

# Do a mean-field KS-DFT calculation
mol = gto.Mole(atom=coord, basis=basisname,charge=0,spin=0) # build a molecule, neutral species, spin is N\alpha-N\beta here 
mol.build()
mf = scf.RKS(mol) # run DFT calculation as reference
mf.verbose = verb # reduce output
mf.xc = 'HF*0.1 + 0.08*LDA + 0.72*B88, 0.81*LYP + 0.19*VWN' # we build an approximate XC functional as we please

edft = mf.kernel() # get DFT energy
nelec = mol.nelec # number of electrons
print('The number of electrons is now {0}'.format(nelec))
norb = mf.mo_coeff.shape[1] # number of MOs, basisname determines this, can get it from many places
enuc = mol.energy_nuc() # nuclear repulsion energy
s = mf.get_ovlp(mol) # the overlap matrix of the basis set, again determined by basisname


print(' At a distance R = {0} angstrom:'.format(dist))
print(' DFT energy: {0} a.u.\n'.format(edft)) # total energies

# Test for DFT density matrix, which gives a correct description
h1 = mf.get_hcore()
rdm1dft = mf.make_rdm1()
eedft_1 = np.trace(h1 @ rdm1dft) #np.einsum('ij,ji->', h1, rdm1dft)
vdft = mf.get_veff(mol, rdm1dft)
eedft_2 = np.einsum('ij,ji->', vdft, rdm1dft) * .5
edft = eedft_1 + eedft_2 + enuc
n = np.trace(s @ rdm1dft)
print(' Number of electrons is {0} before diagonalization.'.format(n))
w = linalg.eig(s @ rdm1dft)[0]
n = np.sum(w).real
print(' Number of electrons is {0} after diagonalization.'.format(n))

# Now lets calculate densities in a grid
#coords = np.random.random((100,3)) # 100 random points
# Or we can generate an array
#print(mol.atom_coords())
zz = np.linspace(-2, 4, 500)
coords = [(0, 0, z) for z in zz]

ao = numint.eval_ao(mol, coords, deriv=0) # we get the AO value for every point in space of coords, returns 2d array (N,nao)
rho_dft = numint.eval_rho(mol,ao,rdm1dft,xctype='LDA',hermi=0) # we can evaluate rho at any point of space
plt.plot(zz,rho_dft,label='DFT')
plt.legend()
plt.savefig('density_dft.png')


# However if we need the derivatives of the density we need the derivatives of the aos
# so we increase deriv
# this returns (:,N,nao) where we get rho,rhox,rhoy,rhoz,drhodx... etc.

ao = numint.eval_ao(mol, coords, deriv=1) # 1 for first derivatives
# now we can evaluate \rho and first derivatives
rho, dx_rho, dy_rho, dz_rho= numint.eval_rho(mol,ao,rdm1dft,xctype='GGA',hermi=0)
plt.plot(zz,dz_rho,label='DFT')
plt.legend()
plt.savefig('1stderivative_dft.png')

ao = numint.eval_ao(mol, coords, deriv=2) # 2 because we need second derivatives
# now we can evaluate \rho, any of its spatial derivatives, \nabla^2 rho (ked) and 1/2(\nabla rho)^2 (tau)
rho, dx_rho, dy_rho, dz_rho, ked, tau = numint.eval_rho(mol,ao,rdm1dft,xctype='mGGA',hermi=0)
plt.plot(zz,ked,label='DFT')
plt.legend()
plt.savefig('ked_dft.png')






