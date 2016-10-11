
COMMENT
26 Ago 2002 Modification of original channel to allow variable time step and to correct an initialization error.
    Done by Michael Hines(michael.hines@yale.e) and Ruggero Scorcioni(rscorcio@gmu.edu) at EU Advance Course in Computational Neuroscience. Obidos, Portugal

kv.mod

Potassium channel, Hodgkin-Huxley style kinetics
Kinetic rates based roughly on Sah et al. and Hamill et al. (1991)

Author: Zach Mainen, Salk Institute, 1995, zach@salk.edu
	
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX kfast : neamed from kv (Armin, Jul 09)
	USEION k READ ek WRITE ik
	RANGE n, gk, gbar, vshift, timefactor_n, ik
	RANGE ninf, ntau
	GLOBAL Ra, Rb
	GLOBAL q10, temp, tadj, vmin, vmax
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

PARAMETER {
	gbar = 0   	(pS/um2)	: 0.03 mho/cm2
	v 		(mV)
	vshift = 0	(mV)
								
	tha  = 25	(mV)		: v 1/2 for inf
	qa   = 9	(mV)		: inf slope		
	
	Ra   = 0.02	(/ms)		: max act rate
	Rb   = 0.002	(/ms)		: max deact rate	

	dt		(ms)
	celsius		(degC)
	temp = 23	(degC)		: original temp 	
	q10  = 2.3			: temperature sensitivity

	vmin = -120	(mV)
	vmax = 100	(mV)
	
	timefactor_n = 1
} 


ASSIGNED {
	a		(/ms)
	b		(/ms)
	ik 		(mA/cm2)
	gk		(pS/um2)
	ek		(mV)
	ninf
	ntau (ms)	
	tadj
}
 

STATE { n }

INITIAL { 
	trates(v-vshift)
	n = ninf
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	gk = tadj*gbar*n
	ik = (1e-4) * gk * (v - ek)
} 



DERIVATIVE  states {   :Computes state variable n 
        trates(v-vshift)      :             at the current v and dt.
        n' =  (ninf-n)/(timefactor_n*ntau)
}

PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        
        TABLE ninf, ntau
	DEPEND  celsius, temp, Ra, Rb, tha, qa
	
	FROM vmin TO vmax WITH 199

	rates(v): not consistently executed from here if usetable_hh == 1


:        tinc = -dt * tadj
:        nexp = 1 - exp(tinc/ntau)

}


PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.

        a = Ra * (v - tha) / (1 - exp(-(v - tha)/qa))
        b = -Rb * (v - tha) / (1 - exp((v - tha)/qa))

        tadj = q10^((celsius - temp)/10)
        ntau = 1/tadj/(a+b)
	ninf = a/(a+b)
}

