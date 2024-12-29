import lal
import lalsimulation
from astropy import units as u
import torch
import numpy as np
import jax.numpy as jnp
from torch import Tensor
from phenom_p import IMRPhenomPv2
from IMRPhenomPv2 import gen_IMRPhenomPv2
from conversion import bilby_spins_to_lalsim, chirp_mass_and_mass_ratio_to_components


chirp_mass = torch.tensor([30.0])
mass_ratio = torch.tensor([0.8])
mass_1, mass_2 = chirp_mass_and_mass_ratio_to_components(chirp_mass, mass_ratio)
f_ref = 20.0
sample_rate = 2048.0

a_1 = torch.tensor([0.5])
a_2 = torch.tensor([0.8])
tilt_1 = torch.tensor([np.pi / 3])
tilt_2 = torch.tensor([np.pi / 2])
phi_12 = torch.tensor([np.pi / 4])
phi_jl = torch.tensor([np.pi / 6])
distance = torch.tensor([500.0])
theta_jn = torch.tensor([np.pi / 4])
phase = torch.tensor([7 * np.pi / 4])

inclination, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z = bilby_spins_to_lalsim(
    theta_jn,
    phi_jl,
    tilt_1,
    tilt_2,
    phi_12,
    a_1,
    a_2,
    mass_1,
    mass_2,
    f_ref,
    phase,
)

params = dict(
    m1=mass_1.item() * lal.MSUN_SI,
    m2=mass_2.item() * lal.MSUN_SI,
    S1x=chi1x.item(),
    S1y=chi1y.item(),
    S1z=chi1z.item(),
    S2x=chi2x.item(),
    S2y=chi2y.item(),
    S2z=chi2z.item(),
    distance=(distance.item() * u.Mpc).to("m").value,
    inclination=inclination.item(),
    phiRef=phase.item(),
    longAscNodes=0.0,
    eccentricity=0.0,
    meanPerAno=0.0,
    deltaF=1.0 / sample_rate,
    f_min=10.0,
    f_ref=f_ref,
    f_max=300,
    approximant=lalsimulation.IMRPhenomPv2,
    LALpars=lal.CreateDict(),
)
hp_lal, hc_lal = lalsimulation.SimInspiralChooseFDWaveform(**params)


lal_freqs = np.array(
    [hp_lal.f0 + ii * hp_lal.deltaF for ii in range(len(hp_lal.data.data))]
)

theta_ripple = jnp.array(
    [
        mass_1.item(),
        mass_2.item(),
        chi1x.item(),
        chi1y.item(),
        chi1z.item(),
        chi2x.item(),
        chi2y.item(),
        chi2z.item(),
        distance.item(),
        0,
        phase.item(),
        inclination.item(),
    ]
)

lal_mask = (lal_freqs > params["f_min"]) & (lal_freqs < params["f_max"])

lal_freqs = lal_freqs[lal_mask]
ml4gw_freqs = torch.tensor(lal_freqs, dtype=torch.float32)
ripple_freqs = jnp.array(lal_freqs)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
args = parser.parse_args()

if args.torch:
    hc_ml4gw, hp_ml4gw = IMRPhenomPv2().forward(
        ml4gw_freqs,
        chirp_mass,
        mass_ratio,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        distance,
        phase,
        inclination,
        f_ref,
        torch.tensor([0.0]),
    )
else:
    hp_ripple, hc_ripple = gen_IMRPhenomPv2(ripple_freqs, theta_ripple, f_ref)
