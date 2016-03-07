TITLE Low threshold calcium current
:
:   Ca++ current responsible for low threshold spikes (LTS)
:
:   Written by Alain Destexhe, Salk Institute, Sept 18, 1992
:   Modified by Geir Halnes, Norwegian University of Life Sciences, June 2011:
:
:     - Kinetics adapted to LGN interneuron data from Broicher et al.: Mol Cell Neurosci 36: 132-145, 2007.
:         using Q10 values of 3 and 1.5 for activation/inactivation.
:     - Activation variable shifted 8mV to account for dLGN interneuron data in Halnes et al. 2011


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX it2
	USEION Ca READ Cai, Cao WRITE iCa VALENCE 2
	RANGE gcabar, g
	GLOBAL m_inf, tau_m, h_inf, tau_h, shift2, sm, sh, phi_m, phi_h, hx, mx, shift1
}

UNITS {
	(molar) = (1/liter)
	(mV) =	(millivolt)
	(mA) =	(milliamp)
	(mM) =	(millimolar)
	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}


PARAMETER {
	v		(mV)
	celsius	= 36	(degC)
	gcabar	= 8.5e-6	(mho/cm2)
      hx      = 1.5
      mx      = 3.0
	Cai	= 5e-5 (mM) : Initial Ca concentration
	Cao	= 2	(mM) : External Ca concentration

: GH, parameters fitted to Broicher et al. 07 - data
	minf1 = 46.2
	hinf1 = 69.7
	taum1 = 5.4
	taum2 = 125.7
	taum3 = -19.7
	taum4 = -0.54
	taum5 = 13
	tauh1 = 21
	tauh2 = 22.2
	tauh3 = 9.1
	tauh4 = 362.9
	tauh5 = 46.9
      sm = 8.7
      sh = 6.4
	shift1 = -8 	(mV) : Halnes et al. 2011
      shift2  = 0    	(mV) : Halnes et al. 2011
}


STATE {
	m h
}

ASSIGNED {
	iCa	(mA/cm2)
	g       (mho/cm2)
	carev	(mV)
	m_inf
	tau_m	(ms)
	h_inf
	tau_h	(ms)
	phi_m
	phi_h
}

BREAKPOINT {
	SOLVE castate METHOD cnexp
	g = gcabar * m*m*h
	iCa = g * ghk(v, Cai, Cao)
}

DERIVATIVE castate {
	evaluate_fct(v)
	m' = (m_inf - m) / tau_m
	h' = (h_inf - h) / tau_h
}

UNITSOFF
INITIAL {
	VERBATIM
	Cai = _ion_Cai;
	Cao = _ion_Cao;
	ENDVERBATIM
:
	phi_m = mx ^ ((celsius-23.5)/10)
	phi_h = hx ^ ((celsius-23.5)/10)

	evaluate_fct(v)
	m = m_inf
	h = h_inf
}

PROCEDURE evaluate_fct(v(mV)) { 
	m_inf = 1.0 / ( 1 + exp(-(v+shift1+minf1)/sm) )
	h_inf = 1.0 / ( 1 + exp((v+shift2+hinf1)/sh) )
	tau_m = (taum1+1.0/(exp((v+shift1+taum2)/(taum3))+exp((v+shift1+taum4)/taum5)))/ phi_m
	tau_h = (tauh1+1/(exp((v+shift2+tauh2)/tauh3)+exp(-(v+shift2+tauh4)/tauh5)))/phi_h
}

FUNCTION ghk(v(mV), Ci(mM), Co(mM)) (.001 coul/cm3) {
	LOCAL z, eci, eco
	z = (1e-3)*2*FARADAY*v/(R*(celsius+273.15))
	eco = Co*efun(z)
	eci = Ci*efun(-z)
	:high Cao charge moves inward
	:negative potential charge moves inward
	ghk = (.001)*2*FARADAY*(eci - eco)
}

FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}
UNITSON
