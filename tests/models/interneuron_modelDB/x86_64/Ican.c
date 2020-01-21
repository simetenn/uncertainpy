/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ican
#define _nrn_initial _nrn_initial__ican
#define nrn_cur _nrn_cur__ican
#define _nrn_current _nrn_current__ican
#define nrn_jacob _nrn_jacob__ican
#define nrn_state _nrn_state__ican
#define _net_receive _net_receive__ican 
#define evaluate_fct evaluate_fct__ican 
#define states states__ican 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gbar _p[0]
#define i _p[1]
#define g _p[2]
#define m _p[3]
#define Cai _p[4]
#define Dm _p[5]
#define iother _p[6]
#define tadj _p[7]
#define _g _p[8]
#define _ion_iother	*_ppvar[0]._pval
#define _ion_diotherdv	*_ppvar[1]._pval
#define _ion_Cai	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_evaluate_fct(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_ican", _hoc_setdata,
 "evaluate_fct_ican", _hoc_evaluate_fct,
 0, 0
};
 /* declare global and static user variables */
#define beta beta_ican
 double beta = 0.003;
#define cac cac_ican
 double cac = 0.00011;
#define erev erev_ican
 double erev = 10;
#define m_inf m_inf_ican
 double m_inf = 0;
#define taumin taumin_ican
 double taumin = 0.1;
#define tau_m tau_m_ican
 double tau_m = 0;
#define x x_ican
 double x = 8;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "erev_ican", "mV",
 "cac_ican", "mM",
 "taumin_ican", "ms",
 "tau_m_ican", "ms",
 "gbar_ican", "mho/cm2",
 "i_ican", "mA/cm2",
 "g_ican", "mho/cm2",
 0,0
};
 static double delta_t = 1;
 static double m0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "erev_ican", &erev_ican,
 "beta_ican", &beta_ican,
 "cac_ican", &cac_ican,
 "taumin_ican", &taumin_ican,
 "x_ican", &x_ican,
 "m_inf_ican", &m_inf_ican,
 "tau_m_ican", &tau_m_ican,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ican",
 "gbar_ican",
 0,
 "i_ican",
 "g_ican",
 0,
 "m_ican",
 0,
 0};
 static Symbol* _other_sym;
 static Symbol* _Ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 9, _prop);
 	/*initialize range parameters*/
 	gbar = 1e-05;
 	_prop->param = _p;
 	_prop->param_size = 9;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_other_sym);
 	_ppvar[0]._pval = &prop_ion->param[3]; /* iother */
 	_ppvar[1]._pval = &prop_ion->param[4]; /* _ion_diotherdv */
 prop_ion = need_memb(_Ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[2]._pval = &prop_ion->param[1]; /* Cai */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Ican_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("other", 1.0);
 	ion_reg("Ca", 2.0);
 	_other_sym = hoc_lookup("other_ion");
 	_Ca_sym = hoc_lookup("Ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 9, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "other_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "other_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "Ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ican /home/docker/uncertainpy/tests/models/interneuron_modelDB/x86_64/Ican.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Slow Ca-dependent cation current";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int evaluate_fct(double, double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   evaluate_fct ( _threadargscomma_ v , Cai ) ;
   Dm = ( m_inf - m ) / tau_m ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 evaluate_fct ( _threadargscomma_ v , Cai ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tau_m )) ;
  return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   evaluate_fct ( _threadargscomma_ v , Cai ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tau_m)))*(- ( ( ( m_inf ) ) / tau_m ) / ( ( ( ( - 1.0 ) ) ) / tau_m ) - m) ;
   }
  return 0;
}
 
static int  evaluate_fct (  double _lv , double _lCai ) {
   double _lalpha ;
 _lalpha = beta * pow( ( _lCai / cac ) , x ) ;
   tau_m = 1.0 / ( _lalpha + beta ) / tadj ;
   m_inf = _lalpha / ( _lalpha + beta ) ;
   if ( tau_m < taumin ) {
     tau_m = taumin ;
     }
    return 0; }
 
static void _hoc_evaluate_fct(void) {
  double _r;
   _r = 1.;
 evaluate_fct (  *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  Cai = _ion_Cai;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 ();
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  Cai = _ion_Cai;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_other_sym, _ppvar, 0, 3);
   nrn_update_ion_pointer(_other_sym, _ppvar, 1, 4);
   nrn_update_ion_pointer(_Ca_sym, _ppvar, 2, 1);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  m = m0;
 {
   
/*VERBATIM*/
	Cai = _ion_Cai;
 tadj = pow( 3.0 , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   evaluate_fct ( _threadargscomma_ v , Cai ) ;
   m = m_inf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  Cai = _ion_Cai;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   g = gbar * m * m ;
   i = g * ( v - erev ) ;
   iother = i ;
   }
 _current += iother;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  Cai = _ion_Cai;
 _g = _nrn_current(_v + .001);
 	{ double _diother;
  _diother = iother;
 _rhs = _nrn_current(_v);
  _ion_diotherdv += (_diother - iother)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_iother += iother ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  Cai = _ion_Cai;
 { error =  states();
 if(error){fprintf(stderr,"at line 72 in file Ican.mod:\n	SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(m) - _p;  _dlist1[0] = &(Dm) - _p;
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "/home/docker/uncertainpy/tests/models/interneuron_modelDB/Ican.mod";
static const char* nmodl_file_text = 
  "TITLE Slow Ca-dependent cation current\n"
  ":\n"
  ":   Ca++ dependent nonspecific cation current ICAN\n"
  ":   Differential equations\n"
  ":\n"
  ":   This file was taken the study of Zhu et al.: Neuroscience 91, 1445-1460, 1999,\n"
  ":   where kinetics were based on Partridge & Swandulla, TINS 11: 69-72, 1988\n"
  "\n"
  ":   Modified by Geir Halnes, Norwegian University of Life Sciences, June 2011\n"
  ":   (using only 1 of the two calcium pools applied by Zhu et al. 99)\n"
  "\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX ican\n"
  "	USEION other WRITE iother VALENCE 1\n"
  "	USEION Ca READ Cai VALENCE 2\n"
  "      RANGE gbar, i, g\n"
  "	GLOBAL m_inf, tau_m, beta, cac, taumin, erev, x\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(molar) = (1/liter)\n"
  "	(mM) = (millimolar)\n"
  "}\n"
  "\n"
  "\n"
  "PARAMETER {\n"
  "	v		(mV)\n"
  "	celsius	= 36	(degC)\n"
  "	erev = 10	(mV)\n"
  "	Cai 	= .00005	(mM)	: initial [Ca]i = 50 nM\n"
  "	gbar	= 1e-5	(mho/cm2)\n"
  "	beta = 0.003 \n"
  "	cac	= 1.1e-4	(mM)		: middle point of activation fct\n"
  "	taumin = 0.1	(ms)		: minimal value of time constant\n"
  "	x = 8\n"
  "}\n"
  "\n"
  "\n"
  "STATE {\n"
  "	m\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  ":  activation kinetics are assumed to be at 22 deg. C\n"
  ":  Q10 is assumed to be 3\n"
  ":\n"
  "	VERBATIM\n"
  "	Cai = _ion_Cai;\n"
  "	ENDVERBATIM\n"
  "\n"
  "	tadj = 3.0 ^ ((celsius-22.0)/10)\n"
  "	evaluate_fct(v,Cai)\n"
  "	m = m_inf\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	i	(mA/cm2)\n"
  "	iother	(mA/cm2)\n"
  "	g       (mho/cm2)\n"
  "	m_inf\n"
  "	tau_m	(ms)\n"
  "	tadj\n"
  "}\n"
  "\n"
  "BREAKPOINT { \n"
  "	SOLVE states METHOD cnexp\n"
  "	g = gbar * m*m\n"
  "	i = g * (v - erev)\n"
  "	iother = i\n"
  "}\n"
  "\n"
  "DERIVATIVE states { \n"
  "	evaluate_fct(v,Cai)\n"
  "	m' = (m_inf - m) / tau_m\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "PROCEDURE evaluate_fct(v(mV),Cai(mM)) {  LOCAL alpha\n"
  "	alpha = beta * (Cai/cac)^x\n"
  "	tau_m = 1 / (alpha + beta) / tadj\n"
  "	m_inf = alpha / (alpha + beta)\n"
  "      if(tau_m < taumin) { tau_m = taumin } 	: min value of time cst\n"
  "}\n"
  "UNITSON\n"
  ;
#endif
