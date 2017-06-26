/* Created by Language version: 6.2.0 */
/* NOT VECTORIZED */
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
 
#define _threadargscomma_ /**/
#define _threadargs_ /**/
 
#define _threadargsprotocomma_ /**/
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
#define vshift _p[1]
#define vshift2 _p[2]
#define ik _p[3]
#define gk _p[4]
#define ainf _p[5]
#define binf _p[6]
#define taua _p[7]
#define taub _p[8]
#define taub1 _p[9]
#define a _p[10]
#define b _p[11]
#define b1 _p[12]
#define ek _p[13]
#define Da _p[14]
#define Db _p[15]
#define Db1 _p[16]
#define _g _p[17]
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
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
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
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
 "setdata_kslow", _hoc_setdata,
 "rates_kslow", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define a6 a6_kslow
 double a6 = -0.0053;
#define a5 a5_kslow
 double a5 = 71;
#define a4 a4_kslow
 double a4 = -1.27;
#define a3 a3_kslow
 double a3 = 0.01938;
#define a2 a2_kslow
 double a2 = 13.1;
#define a1 a1_kslow
 double a1 = 11.1;
#define a0 a0_kslow
 double a0 = 0.0052;
#define bb4 bb4_kslow
 double bb4 = 0.0306;
#define bb3 bb3_kslow
 double bb3 = -210;
#define bb2 bb2_kslow
 double bb2 = 0.01118;
#define bb1 bb1_kslow
 double bb1 = 1380;
#define bb0 bb0_kslow
 double bb0 = 2350;
#define b5 b5_kslow
 double b5 = -54;
#define b4 b4_kslow
 double b4 = 23.7;
#define b3 b3_kslow
 double b3 = 48;
#define b2 b2_kslow
 double b2 = -75;
#define b11 b11_kslow
 double b11 = 1010;
#define b0 b0_kslow
 double b0 = 360;
#define q10 q10_kslow
 double q10 = 2.3;
#define tadj tadj_kslow
 double tadj = 0;
#define temp temp_kslow
 double temp = 21;
#define usetable usetable_kslow
 double usetable = 1;
#define vmax vmax_kslow
 double vmax = 100;
#define vmin vmin_kslow
 double vmin = -120;
#define v05b v05b_kslow
 double v05b = -58;
#define v05a v05a_kslow
 double v05a = -14.3;
#define zb zb_kslow
 double zb = -11;
#define za za_kslow
 double za = 14.6;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_kslow", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "v05a_kslow", "mV",
 "za_kslow", "mV",
 "v05b_kslow", "mV",
 "zb_kslow", "mV",
 "a0_kslow", "1/ms",
 "a1_kslow", "mV",
 "a2_kslow", "mV",
 "a3_kslow", "1/ms",
 "a4_kslow", "mV",
 "a5_kslow", "mV",
 "a6_kslow", "1/ms",
 "b0_kslow", "ms",
 "b11_kslow", "ms",
 "b2_kslow", "mV",
 "b3_kslow", "mV",
 "b4_kslow", "ms/mV",
 "b5_kslow", "mV",
 "bb0_kslow", "ms",
 "bb1_kslow", "ms",
 "bb2_kslow", "mV",
 "bb3_kslow", "ms",
 "bb4_kslow", "mV",
 "temp_kslow", "degC",
 "vmin_kslow", "mV",
 "vmax_kslow", "mV",
 "gbar_kslow", "pS/um2",
 "vshift_kslow", "mV",
 "vshift2_kslow", "mV",
 "ik_kslow", "mA/cm2",
 "gk_kslow", "pS/um2",
 "taua_kslow", "ms",
 "taub_kslow", "ms",
 "taub1_kslow", "ms",
 0,0
};
 static double b10 = 0;
 static double delta_t = 1;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "v05a_kslow", &v05a_kslow,
 "za_kslow", &za_kslow,
 "v05b_kslow", &v05b_kslow,
 "zb_kslow", &zb_kslow,
 "a0_kslow", &a0_kslow,
 "a1_kslow", &a1_kslow,
 "a2_kslow", &a2_kslow,
 "a3_kslow", &a3_kslow,
 "a4_kslow", &a4_kslow,
 "a5_kslow", &a5_kslow,
 "a6_kslow", &a6_kslow,
 "b0_kslow", &b0_kslow,
 "b11_kslow", &b11_kslow,
 "b2_kslow", &b2_kslow,
 "b3_kslow", &b3_kslow,
 "b4_kslow", &b4_kslow,
 "b5_kslow", &b5_kslow,
 "bb0_kslow", &bb0_kslow,
 "bb1_kslow", &bb1_kslow,
 "bb2_kslow", &bb2_kslow,
 "bb3_kslow", &bb3_kslow,
 "bb4_kslow", &bb4_kslow,
 "temp_kslow", &temp_kslow,
 "q10_kslow", &q10_kslow,
 "vmin_kslow", &vmin_kslow,
 "vmax_kslow", &vmax_kslow,
 "tadj_kslow", &tadj_kslow,
 "usetable_kslow", &usetable_kslow,
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
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "6.2.0",
"kslow",
 "gbar_kslow",
 "vshift_kslow",
 "vshift2_kslow",
 0,
 "ik_kslow",
 "gk_kslow",
 "ainf_kslow",
 "binf_kslow",
 "taua_kslow",
 "taub_kslow",
 "taub1_kslow",
 0,
 "a_kslow",
 "b_kslow",
 "b1_kslow",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 18, _prop);
 	/*initialize range parameters*/
 	gbar = 0;
 	vshift = 0;
 	vshift2 = 0;
 	_prop->param = _p;
 	_prop->param_size = 18;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*f)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _kslow_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
  hoc_register_prop_size(_mechtype, 18, 4);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 kslow /home/simen/Dropbox/phd/uncertainpy/examples/bahl/bahl_neuron_model/channels/x86_64/kslow.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double _zaexp , _zbexp , _zb1exp , _zz ;
 static double *_t_taua;
 static double *_t_ainf;
 static double *_t_binf;
 static double *_t_taub;
 static double *_t_taub1;
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rates(double);
static int rates(double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_rates(double);
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 () {_reset=0;
 {
   rates ( _threadargscomma_ v - vshift - vshift2 ) ;
   Da = ( ainf - a ) / taua ;
   Db = ( binf - b ) / taub ;
   Db1 = ( binf - b1 ) / taub1 ;
   }
 return _reset;
}
 static int _ode_matsol1 () {
 rates ( _threadargscomma_ v - vshift - vshift2 ) ;
 Da = Da  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taua )) ;
 Db = Db  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taub )) ;
 Db1 = Db1  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taub1 )) ;
 return 0;
}
 /*END CVODE*/
 static int states () {_reset=0;
 {
   rates ( _threadargscomma_ v - vshift - vshift2 ) ;
    a = a + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taua)))*(- ( ( ( ainf ) ) / taua ) / ( ( ( ( - 1.0) ) ) / taua ) - a) ;
    b = b + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taub)))*(- ( ( ( binf ) ) / taub ) / ( ( ( ( - 1.0) ) ) / taub ) - b) ;
    b1 = b1 + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taub1)))*(- ( ( ( binf ) ) / taub1 ) / ( ( ( ( - 1.0) ) ) / taub1 ) - b1) ;
   }
  return 0;
}
 static double _mfac_rates, _tmin_rates;
 static void _check_rates();
 static void _check_rates() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rates =  vmin ;
   _tmax =  vmax ;
   _dx = (_tmax - _tmin_rates)/199.; _mfac_rates = 1./_dx;
   for (_i=0, _x=_tmin_rates; _i < 200; _x += _dx, _i++) {
    _f_rates(_x);
    _t_taua[_i] = taua;
    _t_ainf[_i] = ainf;
    _t_binf[_i] = binf;
    _t_taub[_i] = taub;
    _t_taub1[_i] = taub1;
   }
   _sav_celsius = celsius;
  }
 }

 static int rates(double _lvm){ _check_rates();
 _n_rates(_lvm);
 return 0;
 }

 static void _n_rates(double _lvm){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rates(_lvm); return; 
}
 _xi = _mfac_rates * (_lvm - _tmin_rates);
 _i = (int) _xi;
 if (_xi <= 0.) {
 taua = _t_taua[0];
 ainf = _t_ainf[0];
 binf = _t_binf[0];
 taub = _t_taub[0];
 taub1 = _t_taub1[0];
 return; }
 if (_i >= 199) {
 taua = _t_taua[199];
 ainf = _t_ainf[199];
 binf = _t_binf[199];
 taub = _t_taub[199];
 taub1 = _t_taub1[199];
 return; }
 _theta = _xi - (double)_i;
 taua = _t_taua[_i] + _theta*(_t_taua[_i+1] - _t_taua[_i]);
 ainf = _t_ainf[_i] + _theta*(_t_ainf[_i+1] - _t_ainf[_i]);
 binf = _t_binf[_i] + _theta*(_t_binf[_i+1] - _t_binf[_i]);
 taub = _t_taub[_i] + _theta*(_t_taub[_i+1] - _t_taub[_i]);
 taub1 = _t_taub1[_i] + _theta*(_t_taub1[_i+1] - _t_taub1[_i]);
 }

 
static int  _f_rates (  double _lvm ) {
   double _lalpha , _lbeta ;
 tadj = pow( q10 , ( ( celsius - temp ) / 10.0 ) ) ;
   _lalpha = a0 * ( _lvm - a1 ) / ( 1.0 - exp ( - ( _lvm - a1 ) / a2 ) ) ;
   _lbeta = a3 * exp ( - ( _lvm - a4 ) / a5 ) + a6 ;
   taua = 1.0 / ( _lalpha + _lbeta ) ;
   ainf = _lalpha / ( _lalpha + _lbeta ) ;
   taub = b0 + ( b11 + b4 * ( _lvm - b5 ) ) * exp ( - ( _lvm - b2 ) * ( _lvm - b2 ) / ( b3 * b3 ) ) ;
   taub1 = bb0 + bb1 * exp ( - bb2 * _lvm ) + bb3 * exp ( - bb4 * _lvm ) ;
   binf = 1.0 / ( 1.0 + exp ( - ( _lvm - v05b ) / zb ) ) ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
    _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ek = _ion_ek;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
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
  ek = _ion_ek;
 _ode_matsol1 ();
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  a = a0;
  b1 = b10;
  b = b0;
 {
   rates ( _threadargscomma_ v - vshift - vshift2 ) ;
   a = ainf ;
   b = binf ;
   b1 = binf ;
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
  ek = _ion_ek;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   gk = tadj * gbar * a * a * ( 0.5 * b + 0.5 * b1 ) ;
   ik = ( 1e-4 ) * gk * ( v - ek ) ;
   }
 _current += ik;

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
  ek = _ion_ek;
 _g = _nrn_current(_v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
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
 double _break, _save;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
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
 _break = t + .5*dt; _save = t;
 v=_v;
{
  ek = _ion_ek;
 { {
 for (; t < _break; t += dt) {
 error =  states();
 if(error){fprintf(stderr,"at line 101 in file kslow.mod:\n        SOLVE states METHOD cnexp\n"); nrn_complain(_p); abort_run(error);}
 
}}
 t = _save;
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(a) - _p;  _dlist1[0] = &(Da) - _p;
 _slist1[1] = &(b) - _p;  _dlist1[1] = &(Db) - _p;
 _slist1[2] = &(b1) - _p;  _dlist1[2] = &(Db1) - _p;
   _t_taua = makevector(200*sizeof(double));
   _t_ainf = makevector(200*sizeof(double));
   _t_binf = makevector(200*sizeof(double));
   _t_taub = makevector(200*sizeof(double));
   _t_taub1 = makevector(200*sizeof(double));
_first = 0;
}
