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

# Do a mean-field HF calculation
mol = gto.Mole(atom=coord, basis=basisname,charge=0,spin=0) # build a molecule, neutral species, spin is N\alpha-N\beta here 
mol.build()
mf = scf.RHF(mol) # run HF calculation as reference
mf.verbose = verb # reduce output 
ehf = mf.kernel() # get HF energy
nelec = mol.nelec # number of electrons
print('The number of electrons is now {0}'.format(nelec))
norb = mf.mo_coeff.shape[1] # number of MOs, basisname determines this, can get it from many places
enuc = mol.energy_nuc() # nuclear repulsion energy
s = mf.get_ovlp(mol) # the overlap matrix of the basis set, again determined by basisname

# Do a FCI calculation
h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff # one electron hamiltonian
eri = ao2mo.kernel(mol, mf.mo_coeff) # express ERI in MO basis, note that this is not the 2e Hamiltonian tensor (see pyscf documentation)
fcisolver = fci.direct_spin1.FCI(mol) # we choose and build a FCI solver, for example this simple one
efci, ci = fcisolver.kernel(h1, eri, norb, nelec, ecore=enuc) # do FCI calculation
# At this point , ci contains the FCI vectors. We can use them to build anything
try: 
   assert(np.isclose(efci,-1.163672981178)) # value obtained with PSI4 for this basis set
except:
   print('ERROR. FCI energy {0} does not match reference data.'.format(efci))
   exit()

# More specific calls
# For instance, to get the ELECTRONIC energy, which is the "interesting" part
# The recipe for the total energy is actually simple having the FCI vectors
eefci = fcisolver.energy(h1, eri, ci, norb, nelec)
efci = eefci + enuc # to which we add the nuclear repulsion to get the total energy efci
try: 
   assert(np.isclose(efci,-1.163672981178)) # value obtained with PSI4
except:
   print('ERROR. FCI energy {0} does not match reference data.'.format(efci))
   exit()

print(' At a distance R = {0} angstrom:'.format(dist))
print(' FCI energy: {0} a.u.\n HF energy: {1} a.u.\n Corr. energy: {2} a.u.\n'.format(efci,ehf,abs(efci-ehf))) # total energies

# The spin-traced rdm1 is available in the MO basis as:
one_dm = fcisolver.make_rdm1(ci, norb, nelec)
n = np.trace(one_dm)
print(' Number of electrons is {0} before diagonalization.'.format(n))
w = linalg.eig(one_dm)[0]
n = np.sum(w)
print(' Number of electrons is {0} after diagonalization.'.format(n))
# To get to the AO basis remember that we have the overlap matrix s
# so simply s^-1 @ one_dm

# The spin separated matrices can be obtained as a tuple alpha_dm,beta_dm
one_dm_a, one_dm_b = fcisolver.make_rdm1s(ci, norb, nelec) 
# then obviously one_dm = one_dm_a + one_dm_b

# We can also get the rdm2
rdm1 = fcisolver.make_rdm1(ci, norb, nelec)
rdm2 = fcisolver.make_rdm2(ci,norb,nelec)
# with which we can also calculate the energy

eefci_1 = np.trace(h1 @ rdm1.T) # np.einsum('ij,ji', h1, rdm1) # one-electron part
eri = ao2mo.restore(1, eri, norb) # pyscf can be a little obscure at times, but this is needed to decontract
eefci_2 = np.einsum('pqrs,pqrs',eri,rdm2) * .5 # two-electron part
efci = eefci_1 + eefci_2 + enuc # to which we add the nuclear repulsion to get the total energy efci
try: 
   assert(np.isclose(efci,-1.163672981178)) # value obtained with PSI4
except:
   print('ERROR. FCI energy {0} does not match reference data.'.format(efci))
   exit()

# Test for HF density matrix, which gives a correct description
h1 = mf.get_hcore()
rdm1hf = mf.make_rdm1()
eehf_1 = np.trace(h1 @ rdm1hf) #np.einsum('ij,ji->', h1, rdm1hf)
vhf = mf.get_veff(mol, rdm1hf)
eehf_2 = np.einsum('ij,ji->', vhf, rdm1hf) * .5
ehf = eehf_1 + eehf_2 + enuc
n = np.trace(s @ rdm1hf)
print(' Number of electrons is {0} before diagonalization.'.format(n))
w = linalg.eig(s @ rdm1hf)[0]
n = np.sum(w)
print(' Number of electrons is {0} after diagonalization.'.format(n))

print(' At a distance R = {0} angstrom, using reduced density matrices:'.format(dist))
print(' FCI energy: {0} a.u.\n HF energy: {1} a.u.\n Corr. energy: {2} a.u.\n'.format(efci,ehf,abs(efci-ehf))) # total energies

# The FCI rdm1 needs to be decontracted from the mo coeffs. 
rdm1_corrected = mf.mo_coeff @ rdm1 @ mf.mo_coeff.T

# Now lets calculate densities in a grid
#coords = np.random.random((100,3)) # 100 random points
# Or we can generate an array
#print(mol.atom_coords())
zz = np.linspace(-2, 4, 500)
coords = [(0, 0, z) for z in zz]

ao = numint.eval_ao(mol, coords, deriv=0) # we get the AO value for every point in space of coords, returns 2d array (N,nao)
rho_fci = numint.eval_rho(mol,ao,rdm1_corrected,xctype='LDA',hermi=0) # we can evaluate rho at any point of space
rho_hf = numint.eval_rho(mol,ao,rdm1hf,xctype='LDA',hermi=0) # we can evaluate rho at any point of space
plt.plot(zz,rho_fci,label='FCI')
plt.plot(zz,rho_hf,label='HF')
plt.legend()
plt.savefig('densities.png')


# However if we need the derivatives of the density we need the derivatives of the aos
# so we increase deriv
# this returns (:,N,nao) where we get rho,rhox,rhoy,rhoz,drhodx... etc.

ao = numint.eval_ao(mol, coords, deriv=1) # 1 for first derivatives
# now we can evaluate \rho and first derivatives
rho, dx_rho, dy_rho, dz_rho= numint.eval_rho(mol,ao,rdm1,xctype='GGA',hermi=0)


ao = numint.eval_ao(mol, coords, deriv=2) # 2 because we need second derivatives
# now we can evaluate \rho, any of its spatial derivatives, \nabla^2 rho (ked) and 1/2(\nabla rho)^2 (tau)
rho, dx_rho, dy_rho, dz_rho, ked, tau = numint.eval_rho(mol,ao,rdm1,xctype='mGGA',hermi=0)







