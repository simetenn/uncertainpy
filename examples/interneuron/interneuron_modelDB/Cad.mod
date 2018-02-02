TITLE decay of internal calcium concentration
:
: Simple extrusion mechanism for internal calium dynamics
:
: Written by Alain Destexhe, Salk Institute, Nov 12, 1992
: Modified by Geir Halnes, Norwegian Life Science University of Life Sciences, June 2011


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX Cad
	USEION Ca READ iCa, Cai WRITE Cai VALENCE 2
	RANGE Cainf,taur,k
}

UNITS {
	(molar) = (1/liter)			: moles do not appear in units
	(mM)	= (millimolar)
	(um)	= (micron)
	(mA)	= (milliamp)
	(msM)	= (ms mM)
}


PARAMETER {
	depth	= .1(um)		: depth of shell
	taur	= 50	(ms)		: Zhu et al. used 2 decay terms w/ taus 80ms and 150ms. 1 term 50 ms gives similar decay. 
	Cainf	= 5e-5	(mM)  : Basal Ca-level
	Cainit  = 5e-5 (mM)	: Initial Ca-level
      k       = 0.0155458135   (mmol/C cm)  : Phenomenological constant, estimated to give reasonable intracellular calcium concentration
}


STATE {
	Cai		(mM) <1e-8> : to have tolerance of .01nM
}


INITIAL {
	Cai = Cainit
}


ASSIGNED {
	iCa		(mA/cm2)
	drive_channel	(mM/ms)
	drive_pump	(mM/ms)
}

	
BREAKPOINT {
	SOLVE state METHOD cnexp
}

DERIVATIVE state { 
	drive_channel =  - k * iCa
	if (drive_channel<=0.) { drive_channel = 0. }: cannot pump inward
	Cai' = drive_channel +(Cainf-Cai)/taur
}
