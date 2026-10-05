/* ---------------------------------------------------------
   Melting of n-octadecane PCM with natural convection
   [0,1]x[0,1]
   Case #1: Okada(1984)
   System of eq: Incompressible Navier-Stokes-Boussinesq +
                  Enthalpy method in Energy eq
   Boyancy force: fB = Ra/(Rey^2 Pr) T
   Continuation on temperature for transient regime

               *** PHYSICAL PARAMETERS***
   --------------------------------------------------------*/

// FES and choice of the scaling -----------------
macro typeT P2// P1 or P2
macro scalAdim()1// 1,2 or 3

// Dimension of the cavity --------------------------------
real xl=0,xr=1.,yl=0,yr=1.; // Square dimension [xl,xr]x[yl,yr]
int nbseg = 60;
mesh ThBackup = square(nbseg,nbseg,[xl+x^1.93*(xr-xl),yl+y*(yr-yl)]);

// Dimensionless numbers ----------------------------------// activ nonlinear convection
real conv=0.;
real Pr=56.2,Ra=3.27e5,Ste=0.045;
real dt=1.e-1,dtmax=.5,tmax=20.1;  // 79.1;

real Rey, IRe, IRa, IPr, Uscal, tscal;
real t,dtorig=dt;
int dtadapt=1,dtstep=5;
int iterCT,nbcontT1=0,nbcontT2=5; // nb iteration for temperature continuation

// time save for Tecplot layout of the paper --------------
int nts;
real epsts=1e-10;
real[int] timesave=[40.1,79.1];

IFMACRO(scalAdim,1)
   Rey=1;
   Uscal = Pr/sqrt(Ra);
   tscal = 1;
ENDIFMACRO

IFMACRO(scalAdim,2)
   Rey=1./Pr;
   Uscal = 1.;
   tscal=1./Pr;
ENDIFMACRO

IFMACRO(scalAdim,3)
   Rey=sqrt(Ra/Pr);
   Uscal = sqrt(Pr);
   tscal = sqrt(Ra/Pr);
ENDIFMACRO

timesave *= tscal;
dt *= tscal;
dtmax *= tscal;
tmax *= tscal;
dtorig = dt;
real c1=1./dt,c2=-1./dt,c3=0.;

// Dimensionless Temp ----------------------------------
real Tf=0;// temperature of fusion
real epsi=1.e-2;// width of the mushy region
real Thot = beta, Tcold = -.01, Tin = beta;
real Thotf = Thot,Tcoldf = Tcold, Tinf = Tin;// final Thot, Tcold and Tin
real Tinit=Tcold;

// Boundary conditions ---------------------------------
int cold=2,hot=4;// label of the boundary
int[int] wall = [1,2,3,4];

macro bcu1()on(wall,uw1=0)//
macro bcu2()on(wall,uw2=0)//
macro bcT()on(hot,Tw=Thot)//

IRe = 1./Rey;
IRa = Ra/(Rey*Rey*Pr); IPr = 1./(Rey*Pr);

// Source term (latent heat) ---------------------------
real Enths=0,Enthl=1./Ste;
real Enthas = 1. ;
real EnthRs = epsi, EnthTs = Tf+epsi;

// Carman-Kozeny ---------------------------------------
real CCK = 1e6 , bmushy = 1e-7;
real CKs = 0. , CKl = 1.;
real CKas = 1 ;
real CKRs = epsi , CKTs = Tf+epsi ;

// Boussinesq forces ----------------------------------------
/*
   For PCM: fB = Ra/(Re^2 Pr)*T = IRa * T
*/
macro fB(T)  (IRa*T)//
macro dfB(T) (IRa)//
