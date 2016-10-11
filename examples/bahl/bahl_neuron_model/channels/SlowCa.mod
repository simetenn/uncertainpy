
COMMENT

changed from (AS Oct0899)
ca.mod
Uses fixed eca instead of GHK eqn

HVA Ca current
Based on Reuveni, Friedman, Amitai and Gutnick (1993) J. Neurosci. 13:
4609-4621.

Author: Zach Mainen, Salk Institute, 1994, zach@salk.edu
modified by Armin Bahl to allow variable time step January 2012
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX sca
	USEION ca READ eca WRITE ica
	RANGE m, h, gca, gbar
	RANGE minf, hinf, mtau, htau, inactF, actF
	GLOBAL q10, temp, tadj, vmin, vmax, vshift
}

PARAMETER {
    inactF = 3
	actF   = 1
	gbar = 0   	(pS/um2)	: 0.12 mho/cm2
	vshift = 0	(mV)		: voltage shift (affects all)

	cao  = 2.5	(mM)	        : external ca concentration
	cai		(mM)
						
	temp = 23	(degC)		: original temp 
	q10  = 2.3			: temperature sensitivity

	v 		(mV)
	dt		(ms)
	celsius		(degC)
	vmin = -120	(mV)
	vmax = 100	(mV)
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ica		(mA/cm2)
	gca		(pS/um2)
	eca		(mV)
	minf 
	hinf
	mtau (ms)
	htau (ms)
	tadj
}
 

STATE { m h }

INITIAL { 
	rates(v-vshift)
	m = minf
	h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gca = tadj*gbar*m*m*h
	ica = (1e-4) * gca * (v - eca)
} 

DERIVATIVE states { 
    rates(v-vshift)      
    m' =  (minf-m)/mtau
    h' =  (hinf-h)/htau
}

PROCEDURE rates(vm) {  
    LOCAL  a, b
	tadj = q10^((celsius - temp)/10)
	
	a = 0.055*(-27 - vm)/(exp((-27-vm)/3.8) - 1)/actF
	b = 0.94*exp((-75-vm)/17)/actF
	
	mtau = 1/tadj/(a+b)
	minf = a/(a+b)

		:"h" inactivation 

	a = 0.000457*exp((-13-vm)/50)/inactF
	b = 0.0065/(exp((-vm-15)/28) + 1)/inactF

	htau = 1/tadj/(a+b)
	hinf = a/(a+b)
}

