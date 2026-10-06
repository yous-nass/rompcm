from pyfreefem import FreeFemRunner
from rompcm.paths import DATADIR, TESTSDIR, SRCDIR, podcondTbasis
import pytest


@pytest.fixture
def ff_exports(request):

    params = request.param

    beta = params.get("beta", 1.01)
    tmax = params.get("tmax", 3.6)
    Nr   = params.get("Nr", 50)
    Nt   = params.get("Nt", 100)

    podgalerkin = f"""
                    real beta = {beta};
                    real tmax = {tmax};
                    int Nr = {Nr};
                    int Nt = {Nt};

                    // string BASEDIR = "{podcondTbasis}/";
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
                    int Ns = 1;
                    int Nsnapshots = Nt * Ns;

                    bool savebasis = false;
                    real epsenergy = 1.e-9;
                    string snapPath = DATADIR + "/PCMnoconv/";
                    string pathbasis = "{podcondTbasis}/";

                    mesh Mh = readmesh(TESTDIR + "mesh_online.msh");
                    fespace Vh(Mh, P2);
                    int Ndof = Vh.ndof;

                    include "{SRCDIR}/rompcm/macros_.edp"
                    ifstream fg(TESTDIR + "g1.txt");
                    include "{SRCDIR}/rompcm/pcmcond/pod_gal_cond.edp"
                    exportVar(errl2);
                    exportVar(resl2);
                    exportVar(errinterp);
                   """

    runner = FreeFemRunner(podgalerkin)
    
    return runner.execute(verbosity=0)


@pytest.mark.parametrize(
    "ff_exports",
    [
        {"beta": 1.035, "tmax":3.1, "Nr": 52, "Nt": 100}
    ],
    indirect=True
)

def test_rom(ff_exports):
    errors = ff_exports["resl2"]
    assert errors < 1.e-2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            