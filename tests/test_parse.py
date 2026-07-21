import pytest

from avlnn.parse import AvlParseError, parse_eigenvalues, parse_stability_derivatives

# Field layout verified against real AVL 3.52 ST output (live run, 2026-07-15); values here
# are synthetic but the names/format mirror the real transcript.
SAMPLE_ST_OUTPUT = """
 Standard axis orientation,  X fwd, Z down

 Run case:  Cruise

  Alpha =   3.45123     pb/2V =  -0.00000     p'b/2V =  -0.00000
  Beta  =   0.00000     qc/2V =   0.00000
  Mach  =     0.500     rb/2V =  -0.00000     r'b/2V =  -0.00000

  CXtot =   0.00456     Cltot =   0.00000     Cl'tot =   0.00000
  CYtot =   0.00000     Cmtot =  -0.02341
  CZtot =  -0.45231     Cntot =   0.00000     Cn'tot =   0.00000

  CLtot =   0.45231
  CDtot =   0.03521

                             alpha                beta
                  ----------------    ----------------
 z' force CL |    CLa =   5.234567    CLb =   0.000000
 y  force CY |    CYa =   0.000000    CYb =  -0.456789
 x' mom.  Cl'|    Cla =   0.000000    Clb =  -0.098765
 y  mom.  Cm |    Cma =  -1.123456    Cmb =   0.000000
 z' mom.  Cn'|    Cna =   0.000000    Cnb =   0.123456

 Neutral point  Xnp =   0.612345
"""

# Real .eig file contents from AVL 3.52's own runs/plane.eig sample (format: run-case
# index, real part, imag part -- see EIGOUT in AVL's amode.f).
SAMPLE_EIG_FILE = """\
# Plane Vanilla
#
#   Run case     Eigenvalue
       1    -12.907391         0.0000000
       1   -0.24627495         2.0653901
       1   -0.24627495        -2.0653901
       1    0.92339096E-02     0.0000000
       1    -7.0458219         6.6507334
       1    -7.0458219        -6.6507334
       1   -0.40459605E-02    0.56021121
       1   -0.40459605E-02   -0.56021121
"""


def test_parse_stability_derivatives():
    d = parse_stability_derivatives(SAMPLE_ST_OUTPUT)
    assert d.cl == pytest.approx(0.45231)
    assert d.cd == pytest.approx(0.03521)
    assert d.cm == pytest.approx(-0.02341)
    assert d.alpha_deg == pytest.approx(3.45123)
    assert d.cl_alpha == pytest.approx(5.234567)
    assert d.cm_alpha == pytest.approx(-1.123456)
    assert d.x_np == pytest.approx(0.612345)
    assert d.dcm_dcl == pytest.approx(-1.123456 / 5.234567)


def test_parse_stability_derivatives_missing_field_raises():
    with pytest.raises(AvlParseError):
        parse_stability_derivatives("nothing useful here")


def test_parse_eigenvalues_count_and_values():
    eigs = parse_eigenvalues(SAMPLE_EIG_FILE)
    assert len(eigs) == 8
    assert eigs[0].real == pytest.approx(-12.907391)
    assert eigs[0].imag == pytest.approx(0.0)
    assert eigs[0].is_real_root
    # Fortran E-notation must parse
    assert eigs[3].real == pytest.approx(0.0092339096)
    assert not eigs[1].is_real_root


def test_parse_eigenvalues_filters_by_run_case():
    other_case = SAMPLE_EIG_FILE.replace("       1  ", "       2  ")
    with pytest.raises(AvlParseError):
        parse_eigenvalues(other_case, run_case=1)
    assert len(parse_eigenvalues(other_case, run_case=2)) == 8


def test_parse_eigenvalues_missing_raises():
    with pytest.raises(AvlParseError):
        parse_eigenvalues("no eigenvalues here at all")
