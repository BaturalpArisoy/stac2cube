"""stac2cube package init.

_use_tbb_threading_layer() must stay the first thing this package does - it
has to run before anything imports numpy. See its docstring.
"""


def _openmp_is_safe_for_lazy_imports():
    """Take MKL out of the OpenMP name race, on Windows.

    Windows resolves DLLs by base NAME, with no versioning, so the first
    module to claim ``libiomp5md.dll`` owns it for the whole process. Four
    OpenMP runtimes compete for that here: the conda env's Intel one, LLVM's
    ``libomp.dll``, MSVC's ``vcomp140.dll``, and a SECOND Intel one vendored
    inside the pip torch wheel. MKL delay-loads its copy on the first real
    numeric call - never at import - so the race is decided by whatever the
    user happens to compute first.

    Losing it is not subtle: either torch fails to load ``fbgemm.dll``
    (``OSError: [WinError 182]``, no super-resolution), or OpenMP prints
    ``Error #15`` and calls ``abort()``, killing the kernel.

    Pointing MKL at TBB removes it from the contest entirely: MKL then loads
    ``mkl_tbb_thread`` + ``tbb12`` and no OpenMP at all, leaving torch's
    runtime as the only one in the process. Measured consequences: every
    import ordering survives (before, several aborted), MKL is equal or
    faster (eigvalsh 0.129 -> 0.058 s), and outputs are bit-identical for
    every path this package uses - cloud masking, shadow masking, cube I/O
    and co-registration all verified. Only LAPACK decompositions shift, at
    ~1e-9 relative, and stac2cube calls no LAPACK.

    This is what lets torch and s2cloudless be imported lazily, which is
    worth ~2.2 s and ~415 MB off ``import stac2cube``.

    The catch, and why this returns a flag: MKL reads the variable when it is
    first loaded, which happens at ``import numpy`` - not at the first
    calculation. So if anything already imported numpy before this package,
    the pin is too late and MKL is already bound to Intel OpenMP. That is not
    an exotic case; ``import numpy as np`` above ``import stac2cube`` is the
    normal way a notebook starts.

    Returns True when the lazy imports below are safe, False when the caller
    must settle the race the old way by importing torch and s2cloudless
    eagerly.

    Off Windows this returns True untouched: there is no base-name race
    (Linux resolves by SONAME, and the torch wheels bundle a hashed
    ``libgomp``). tests/test_openmp_order.py re-checks that on every platform.
    """
    import os
    import sys

    if os.name != "nt":
        return True

    layer = os.environ.get("MKL_THREADING_LAYER")
    if layer is not None:
        # Someone chose deliberately; only TBB takes MKL out of the race.
        return layer.strip().upper() == "TBB"

    if "numpy" in sys.modules:
        return False  # too late, MKL is already bound to Intel OpenMP

    # Do not point MKL at a threading layer this env cannot provide.
    for probe in (os.path.join(sys.prefix, "Library", "bin", "tbb12.dll"),
                  os.path.join(sys.prefix, "Library", "bin", "tbb.dll")):
        if os.path.exists(probe):
            os.environ["MKL_THREADING_LAYER"] = "TBB"
            return True
    return False


if not _openmp_is_safe_for_lazy_imports():
    # MKL already owns libiomp5md.dll, or will. Import the two heavy modules
    # now, before any calculation can decide the race against us - this is
    # what the package did unconditionally before, and it costs ~2.2 s. To get
    # the fast path back, set MKL_THREADING_LAYER=TBB in the environment (or
    # the Jupyter kernelspec) so it is in place before Python starts.
    try:
        import torch  # noqa: F401
        import s2cloudless  # noqa: F401
    except Exception:
        # Nothing to gain by masking the real error here; the later, lazy
        # import at point of use will raise it with proper context.
        pass

from .vector_refiner import *
from .main import *
from .get_data import *
from .data_availability import *
from .auxiliary import *
from .stac_processing import *
from .get_spectral_indices import *
from .export_cfg import *
from .clip import *
from .time_series_tools import *
#from .get_topo import *
from .cloud_masking import *
from .shadow_masking import *
from .get_statistics import *
from .get_update import *
from .mosaic import *
from .aoi_tiler import *
from .tables import *
from .coregistration import *
from .super_resolution import *
from .gui import *