import lal
import torch
import numpy as np
import jax.numpy as jnp
from torch import Tensor
from phenom_p import IMRPhenomPv2
from IMRPhenomPv2 import gen_IMRPhenomPv2


def ms_to_Mc_eta(m):
    r"""
    Converts binary component masses to chirp mass and symmetric mass ratio.

    Args:
        m: the binary component masses ``(m1, m2)``

    Returns:
        :math:`(\mathcal{M}, \eta)`, with the chirp mass in the same units as
        the component masses
    """
    m1, m2 = m
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5), m1 * m2 / (m1 + m2) ** 2


# Frequency grid
T = 128
f_l = 20.0
f_sampling = 4096
f_u = f_sampling // 2
f_ref = f_l

delta_t = 1 / f_sampling
tlen = int(round(T / delta_t))
freqs = np.fft.rfftfreq(tlen, delta_t)
df = freqs[1] - freqs[0]
fs = freqs[(freqs > f_l) & (freqs < f_u)]


m1_msun = Tensor([40])
m2_msun = Tensor([25])
chirp_mass = Tensor([(m1_msun * m2_msun) ** (3 / 5) / (m1_msun + m2_msun) ** (1 / 5)])
mass_ratio = Tensor([m1_msun / m2_msun])
m1_kg = Tensor([m1_msun * lal.MSUN_SI])
m2_kg = Tensor([m2_msun * lal.MSUN_SI])
chi1z = Tensor([0.1])
chi2z = Tensor([-0.1])
chi1x = Tensor([-0.2])
chi2x = Tensor([0.2])
chi1y = Tensor([0.9])
chi2y = Tensor([-0.4])
dist_mpc = Tensor([100])
distance = Tensor([dist_mpc * 1e6 * lal.PC_SI])
tc = Tensor([0])
phic = Tensor([torch.pi])
inclination = Tensor([0])
Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun.item(), m2_msun.item()]))
theta_ripple = jnp.array(
    [
        m1_msun.item(),
        m2_msun.item(),
        chi1x.item(),
        chi1y.item(),
        chi1z.item(),
        chi2x.item(),
        chi2y.item(),
        chi2z.item(),
        dist_mpc.item(),
        tc.item(),
        phic.item(),
        inclination.item(),
    ]
)
fs_ripple = jnp.arange(f_l, f_u, df)[1:]
fs_torch = Tensor(np.arange(f_l, f_u, df)[1:])


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
args = parser.parse_args()

if args.torch:
    hp_torch, hc_torch = IMRPhenomPv2().forward(
        fs_torch,
        chirp_mass,
        mass_ratio,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        dist_mpc,
        phic,
        inclination,
        f_ref,
        tc,
    )
else:
    hp_ripple, hc_ripple = gen_IMRPhenomPv2(fs_ripple, theta_ripple, f_ref)
