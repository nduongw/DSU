from .mmd import MaximumMeanDiscrepancy
from .dsbn import DSBN1d, DSBN2d
from .mixup import mixup
from .mixstyle import MixStyle
from .transnorm import TransNorm1d, TransNorm2d
from .sequential2 import Sequential2
from .reverse_grad import ReverseGrad
from .cross_entropy import cross_entropy
from .optimal_transport import SinkhornDivergence, MinibatchEnergyDistance
from .ilm_in import ilm_in
from .conststyle import ConstStyle
from .conststyle2 import ConstStyle2
from .conststyle3 import ConstStyle3
from .conststyle4 import ConstStyle4
from .conststyle5 import ConstStyle5
from .conststyle6 import ConstStyle6
from .efdmix import (
    EFDMix, random_efdmix, activate_efdmix, run_with_efdmix, deactivate_efdmix,
    crossdomain_efdmix, run_without_efdmix
)