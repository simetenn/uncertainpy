COMMENT

k_slow.mod

voltage gated potassium channel, Hodgkin-Huxley style kinetics.  

Kinetics were fit to data from recordings of nucleated patches derived 
from pyramidal neurons. Data recordings and fits from Alon Korngreen 

Author: Alon Korngreen,  MPImF Cell Physiology, 1998,
alon@mpimf-heidelberg.mpg.de

last updated 31/7/2002 by AK

ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX kslow
	USEION k READ ek WRITE ik
	RANGE  a, b, b1,gk, gbar, vshift, vshift2, ik
	RANGE  ainf, taua, binf, taub,taub1
	GLOBAL a0, a1, a2, a3, a4, a5, a6
	GLOBAL b0, b11, b2, b3, b4, b5
	GLOBAL bb0,bb1,bb2,bb3,bb4
	GLOBAL v05a, za, v05b, zb
	GLOBAL q10, temp, tadj, vmin, vmax
}

PARAMETER {
	gbar = 0   	(pS/um2)	: 
	vshift = 0	(mV)		: voltage shift
	vshift2 = 0	(mV)		: a second voltage shift
							
	v05a = -14.3	(mV)		: v 1/2 for act (a) 
	za   =  14.6	(mV)		: act slope		
	v05b = -58	(mV)		: v 1/2 for inact (b) 
	zb   = -11  (mV)		: inact slope
		
	a0   =  0.0052  (1/ms 1/mV)		: parameters for alpha and beta for activation
	a1   = 11.1 	(mV)			:      see below
	a2   = 13.1	(mV)				:      see below 
	a3   = 0.01938    (1/ms)		:      see below 
	a4   = -1.27	(mV)			:	see below
	a5   = 71    (mV)
	a6   = -0.0053 (1/ms)	
	
	b0   = 360	(ms)			: fast inact tau (taub) (ms) 
	b11   = 1010	(ms)		:      see below
	b2   = -75	(mV)			:      see below
	b3   = 48	(mV)			:      see below
	b4   = 23.7     (ms/mV)
	b5   = -54      (mV)

	bb0 = 2350	(ms)			: Slow inactivation tau (taub1)
	bb1 = 1380	(ms)
	bb2 = 0.01118 (mV)
	bb3 = -210  (ms)
	bb4 = 0.0306 (mV)

	temp = 21	(degC)		: original temp 
	q10  = 2.3			: temperature sensitivity

	v 		(mV)
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
	ik 		(mA/cm2)
	gk		(pS/um2)
	ek		(mV)
	ainf 		
	binf
	taua (ms)	
	taub (ms)
	taub1 (ms)	
	tadj
}
 

STATE {a b b1}

INITIAL { 
	rates(v-vshift-vshift2)
	a = ainf
	b = binf 
	b1= binf
}

BREAKPOINT {
        SOLVE states METHOD cnexp
        gk = tadj*gbar*a*a*(0.5*b+0.5*b1)
	  ik = (1e-4) * gk * (v - ek)
} 

LOCAL aexp, bexp,b1exp, z 

DERIVATIVE states {   		
        rates(v-vshift-vshift2) 	
        a'  = (ainf-a)/taua
        b'  = (binf-b)/taub
	  b1' = (binf-b1)/taub1
}


PROCEDURE rates(vm) {  

	LOCAL alpha, beta
	TABLE  taua, ainf, binf, taub, taub1  DEPEND celsius FROM vmin TO vmax WITH 199
	tadj = q10^((celsius - temp)/10)
	
	alpha=a0*(vm-a1)/(1-exp(-(vm-a1)/a2))
	beta=a3*exp(-(vm-a4)/a5)+a6

	taua=1/(alpha+beta)
	ainf = alpha/(alpha+beta)
	
	taub = b0 + (b11+b4*(vm-b5))*exp(-(vm-b2)*(vm-b2)/(b3*b3))
    	taub1=bb0+bb1*exp(-bb2*vm)+bb3*exp(-bb4*vm)
	binf = 1/(1+exp(-(vm-v05b)/zb))
}


