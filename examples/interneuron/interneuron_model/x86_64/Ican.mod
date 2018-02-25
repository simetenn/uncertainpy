TITLE Slow Ca-dependent cation current
:
:   Ca++ dependent nonspecific cation current ICAN
:   Differential equations
:
:   This file was taken the study of Zhu et al.: Neuroscience 91, 1445-1460, 1999,
:   where kinetics were based on Partridge & Swandulla, TINS 11: 69-72, 1988

:   Modified by Geir Halnes, Norwegian University of Life Sciences, June 2011
:   (using only 1 of the two calcium pools applied by Zhu et al. 99)


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX ican
	USEION other WRITE iother VALENCE 1
	USEION Ca READ Cai VALENCE 2
      RANGE gbar, i, g
	GLOBAL m_inf, tau_m, beta, cac, taumin, erev, x
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
}


PARAMETER {
	v		(mV)
	celsius	= 36	(degC)
	erev = 10	(mV)
	Cai 	= .00005	(mM)	: initial [Ca]i = 50 nM
	gbar	= 1e-5	(mho/cm2)
	beta = 0.003 
	cac	= 1.1e-4	(mM)		: middle point of activation fct
	taumin = 0.1	(ms)		: minimal value of time constant
	x = 8
}


STATE {
	m
}

INITIAL {
:  activation kinetics are assumed to be at 22 deg. C
:  Q10 is assumed to be 3
:
	VERBATIM
	Cai = _ion_Cai;
	ENDVERBATIM

	tadj = 3.0 ^ ((celsius-22.0)/10)
	evaluate_fct(v,Cai)
	m = m_inf
}

ASSIGNED {
	i	(mA/cm2)
	iother	(mA/cm2)
	g       (mho/cm2)
	m_inf
	tau_m	(ms)
	tadj
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	g = gbar * m*m
	i = g * (v - erev)
	iother = i
}

DERIVATIVE states { 
	evaluate_fct(v,Cai)
	m' = (m_inf - m) / tau_m
}

UNITSOFF

PROCEDURE evaluate_fct(v(mV),Cai(mM)) {  LOCAL alpha
	alpha = beta * (Cai/cac)^x
	tau_m = 1 / (alpha + beta) / tadj
	m_inf = alpha / (alpha + beta)
      if(tau_m < taumin) { tau_m = taumin } 	: min value of time cst
}
UNITSON
