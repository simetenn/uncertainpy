COMMENT

Deterministic model of kinetics and voltage-dependence of Ih-currents
in layer 5 pyramidal neuron, see Kole et al., 2006. Implemented by
Stefan Hallermann.

Added possibility to shift voltage activiation (vshift) and allowed access to gating variables, Armin Bahl 2009

Predominantly HCN1 / HCN2 

ENDCOMMENT

TITLE Ih-current

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
     (mM) = (milli/liter)

}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

PARAMETER {
	dt 	   		(ms)
	v 	   		(mV)
        ehd=-47 		(mV) 				       
	gbar=0 (pS/um2)	
	gamma_ih	:not used
	seed		:not used
	vshift = 0
}


NEURON {
	SUFFIX ih
	NONSPECIFIC_CURRENT Iqq
	RANGE Iqq,gbar,vshift,ehd, qtau, qinf, gq
}

STATE {
	qq
}

ASSIGNED {
	Iqq (mA/cm2)
	qtau (ms)
	qinf
	gq	(pS/um2)
	
}

INITIAL {
	qq=alpha(v-vshift)/(beta(v-vshift)+alpha(v-vshift))

	qtau = 1./(alpha(v-vshift) + beta(v-vshift))
	qinf = alpha(v)/(alpha(v-vshift) + beta(v-vshift))
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	
	qtau = 1./(alpha(v-vshift) + beta(v-vshift))
	qinf = alpha(v-vshift)/(alpha(v-vshift) + beta(v-vshift))
	
	gq = gbar*qq
	Iqq = (1e-4)*gq*(v-ehd)
	
}

FUNCTION alpha(v(mV)) {

	alpha = 0.001*6.43*(v+154.9)/(exp((v+154.9)/11.9)-1)
	: parameters are estimated by direct fitting of HH model to
        : activation time constants and voltage activation curve
        : recorded at 34C

}

FUNCTION beta(v(mV)) {
	beta = 0.001*193*exp(v/33.1)			
}

DERIVATIVE state {     : exact when v held constant; integrates over dt step
	qq' = (1-qq)*alpha(v-vshift) - qq*beta(v-vshift)
}
