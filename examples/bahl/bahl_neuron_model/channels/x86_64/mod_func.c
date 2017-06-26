#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _IKM_reg(void);
extern void _SlowCa_reg(void);
extern void _cad_reg(void);
extern void _h_reg(void);
extern void _kca_reg(void);
extern void _kfast_reg(void);
extern void _kslow_reg(void);
extern void _nap_reg(void);
extern void _nat_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," IKM.mod");
    fprintf(stderr," SlowCa.mod");
    fprintf(stderr," cad.mod");
    fprintf(stderr," h.mod");
    fprintf(stderr," kca.mod");
    fprintf(stderr," kfast.mod");
    fprintf(stderr," kslow.mod");
    fprintf(stderr," nap.mod");
    fprintf(stderr," nat.mod");
    fprintf(stderr, "\n");
  }
  _IKM_reg();
  _SlowCa_reg();
  _cad_reg();
  _h_reg();
  _kca_reg();
  _kfast_reg();
  _kslow_reg();
  _nap_reg();
  _nat_reg();
}
