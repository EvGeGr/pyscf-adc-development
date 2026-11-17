#!/usr/bin/env python

'''
ADC(3) IPs calculation for FAP
'''

from pyscf import gto, scf, adc

mol = gto.Mole()
r = 1.098
mol.atom = [
     ['N', ( 0., 0.    , -r/2   )],
     ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'cc-pvdz'}
mol.verbose = 4
mol.max_memory = 1000
mol.build()

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
mf.analyze()

# Frozen core
ncore = 2
mf.mo_coeff = (mf.mo_coeff[:, ncore:])
mf.mo_energy = (mf.mo_energy[ncore:])
mf.mol.nelectron -= ncore*2

myadc = adc.RADC(mf).density_fit('cc-pvdz-ri')
myadc.verbose = 6
#myadc.compute_properties = False
myadc.method = "adc(3)"
myadc.method_type = "ip"
myadc.conv_tol = 1e-9
myadc.max_cycle = 100
myadc.max_memory = 10000
myadc.incore_complete = True
#myadc.spin_cont = True
myadc.kernel(nroots = 5)
myadc.analyze()

