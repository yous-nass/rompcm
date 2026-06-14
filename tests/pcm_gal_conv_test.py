from pyfreefem import FreeFemRunner
from rompcm.paths import DATADIR, TESTSDIR, SRCDIR, podconvTbasis, podconvUbasis, podconvVbasis
import pytest


@pytest.fixture
def ff_exports(request):

    params = request.param

    beta = params.get("beta", 1.0)
    tmax = params.get("tmax", 3.6)
    Nr   = params.get("Nr", 40)
    Nt   = params.get("Nt", 50)
    

    podgalerkin = f"""
                    real beta = {beta};
                    real tmax = {tmax};
                    int Nt = {Nt};

                    string DATADIR = "{DATADIR}/";
                    string TESTDIR = "{TESTSDIR}/";

                    IMPORT "io.edp"
                    load "medit"
                    load "bfstream"
                    load "lapack"
                    
                    real tmin = 2.1, dt=0.5, t = 2.6;
                    real c1 = 3./(2.*dt);
                    real c2 = 4./(2.*dt);
                    real c3 = 1./(2.*dt);
                    
                    // ROM dimensions
                    int Ns = 1;
                    int Nsnapshots = Ns*Nt;   // snapshots to load from FOM run 

                    bool savebasis = false;
                    real epsenergy = 1.e-9;
                    int Nr = savebasis ? Nsnapshots: {Nr} ;
                    int NrT = Nr+56;   // temperature modes
                    int NrU = Nr+238;   // velocity modes (per component)
                    int NrV = Nr+209;    // pressure modes

                    string snapDir = DATADIR + "/PCMconv/";

                    mesh Mh = readmesh(TESTDIR + "mesh_online.msh");
                    fespace Vh(Mh, P2);
                    fespace Wh(Mh, P1);

                    // ----------------------------------
                    // Load reference solutions
                    // ----------------------------------
                    string tbasis  = "{podconvTbasis}/";
                    string ubasis  = "{podconvUbasis}/";
                    string vbasis  = "{podconvVbasis}/";

                    int Ndof = Vh.ndof;
                    int NdofP = Wh.ndof;

                    include "{SRCDIR}/rompcm/macros_.edp"
                    ifstream fg(TESTDIR + "g1.txt");
                    include "{SRCDIR}/rompcm/pcmconv/pod_gal_conv.edp"
                    exportVar(errl2);
                    exportVar(resl2);
                    exportVar(errinterp);
                   """

    runner = FreeFemRunner(podgalerkin)
    
    return runner.execute(verbosity=0)


@pytest.mark.parametrize(
    "ff_exports",
    [
        {"beta": 1.035, "tmax":3.6, "Nr": 100}
    ],
    indirect=True
)

def test_rom(ff_exports):
    errors = ff_exports["resl2"]
    assert errors < 1.e-1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            