from typing import Tuple

import torch
from torch import Tensor
from constants import PI, C, gt, m_per_Mpc
from phenom_d_data import QNMData_a, QNMData_fdamp, QNMData_fring
from phenom_d import FinalSpin0815, PhenomInternal_EradRational0815


MPC_SEC = m_per_Mpc / C
"""
1 Mpc in seconds.
"""



def interpolate(
    x: Tensor, xp: Tensor, fp: Tensor
) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    original_shape = x.shape
    x = x.flatten()
    xp = xp.flatten()
    fp = fp.flatten()

    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])  # slope
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.searchsorted(xp, x, right=False) - 1

    interpolated = m[indicies] * x + b[indicies]

    return interpolated.reshape(original_shape)


def ROTATEZ(angle: Tensor, x, y, z):
    tmp_x = x * torch.cos(angle) - y * torch.sin(angle)
    tmp_y = x * torch.sin(angle) + y * torch.cos(angle)
    return tmp_x, tmp_y, z


def ROTATEY(angle, x, y, z):
    tmp_x = x * torch.cos(angle) + z * torch.sin(angle)
    tmp_z = -x * torch.sin(angle) + z * torch.cos(angle)
    return tmp_x, y, tmp_z


def L2PNR(v: Tensor, eta: Tensor) -> torch.Tensor:
    eta2 = eta**2
    x = v**2
    x2 = x**2
    tmp = (
        eta
        * (
            1.0
            + (1.5 + eta / 6.0) * x
            + (3.375 - (19.0 * eta) / 8.0 - eta2 / 24.0) * x2
        )
    ) / x**0.5

    return tmp


def convert_spins(
    m1: Tensor,
    m2: Tensor,
    f_ref: Tensor,
    phiRef: Tensor,
    incl: Tensor,
    s1x: Tensor,
    s1y: Tensor,
    s1z: Tensor,
    s2x: Tensor,
    s2y: Tensor,
    s2z: Tensor,
) -> Tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:
    M = m1 + m2
    m1_2 = m1 * m1
    m2_2 = m2 * m2
    eta = m1 * m2 / (M * M)  # Symmetric mass-ratio

    # From the components in the source frame, we can easily determine
    # chi1_l, chi2_l, chip and phi_aligned, which we need to return.
    # We also compute the spherical angles of J,
    # which we need to transform to the J frame

    # Aligned spins
    chi1_l = s1z  # Dimensionless aligned spin on BH 1
    chi2_l = s2z  # Dimensionless aligned spin on BH 2

    # Magnitude of the spin projections in the orbital plane
    S1_perp = m1_2 * torch.sqrt(s1x**2 + s1y**2)
    S2_perp = m2_2 * torch.sqrt(s2x**2 + s2y**2)

    A1 = 2 + (3 * m2) / (2 * m1)
    A2 = 2 + (3 * m1) / (2 * m2)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp
    num = torch.maximum(ASp1, ASp2)
    den = A2 * m2_2  # warning: this assumes m2 > m1
    chip = num / den

    m_sec = M * gt
    piM = PI * m_sec
    v_ref = (piM * f_ref) ** (1 / 3)
    L0 = M * M * L2PNR(v_ref, eta)
    J0x_sf = m1_2 * s1x + m2_2 * s2x
    J0y_sf = m1_2 * s1y + m2_2 * s2y
    J0z_sf = L0 + m1_2 * s1z + m2_2 * s2z
    J0 = torch.sqrt(J0x_sf * J0x_sf + J0y_sf * J0y_sf + J0z_sf * J0z_sf)

    thetaJ_sf = torch.arccos(J0z_sf / J0)

    phiJ_sf = torch.arctan2(J0y_sf, J0x_sf)

    phi_aligned = -phiJ_sf

    # First we determine kappa
    # in the source frame, the components of N are given in Eq (35c) of T1500606-v6
    Nx_sf = torch.sin(incl) * torch.cos(PI / 2.0 - phiRef)
    Ny_sf = torch.sin(incl) * torch.sin(PI / 2.0 - phiRef)
    Nz_sf = torch.cos(incl)

    tmp_x = Nx_sf
    tmp_y = Ny_sf
    tmp_z = Nz_sf

    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)

    kappa = -torch.arctan2(tmp_y, tmp_x)

    # Then we determine alpha0, by rotating LN
    tmp_x, tmp_y, tmp_z = 0, 0, 1
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    alpha0 = torch.arctan2(tmp_y, tmp_x)

    # Finally we determine thetaJ, by rotating N
    tmp_x, tmp_y, tmp_z = Nx_sf, Ny_sf, Nz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)
    Nx_Jf, Nz_Jf = tmp_x, tmp_z
    thetaJN = torch.arccos(Nz_Jf)

    # Finally, we need to redefine the polarizations:
    # PhenomP's polarizations are defined following Arun et al (arXiv:0810.5336)
    # i.e. projecting the metric onto the P,Q,N triad defined with P=NxJ/|NxJ| (see (2.6) in there).
    # By contrast, the triad X,Y,N used in LAL
    # ("waveframe" in the nomenclature of T1500606-v6)
    # is defined in e.g. eq (35) of this document
    # (via its components in the source frame; note we use the defautl Omega=Pi/2).
    # Both triads differ from each other by a rotation around N by an angle \zeta
    # and we need to rotate the polarizations accordingly by 2\zeta

    Xx_sf = -torch.cos(incl) * torch.sin(phiRef)
    Xy_sf = -torch.cos(incl) * torch.cos(phiRef)
    Xz_sf = torch.sin(incl)
    tmp_x, tmp_y, tmp_z = Xx_sf, Xy_sf, Xz_sf
    tmp_x, tmp_y, tmp_z = ROTATEZ(-phiJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEY(-thetaJ_sf, tmp_x, tmp_y, tmp_z)
    tmp_x, tmp_y, tmp_z = ROTATEZ(kappa, tmp_x, tmp_y, tmp_z)

    # Now the tmp_a are the components of X in the J frame
    # We need the polar angle of that vector in the P,Q basis of Arun et al
    # P = NxJ/|NxJ| and since we put N in the (pos x)z half plane of the J frame
    PArunx_Jf = 0.0
    PAruny_Jf = -1.0
    PArunz_Jf = 0.0

    # Q = NxP
    QArunx_Jf = Nz_Jf
    QAruny_Jf = 0.0
    QArunz_Jf = -Nx_Jf

    # Calculate the dot products XdotPArun and XdotQArun
    XdotPArun = tmp_x * PArunx_Jf + tmp_y * PAruny_Jf + tmp_z * PArunz_Jf
    XdotQArun = tmp_x * QArunx_Jf + tmp_y * QAruny_Jf + tmp_z * QArunz_Jf

    zeta_polariz = torch.arctan2(XdotQArun, XdotPArun)
    return chi1_l, chi2_l, chip, thetaJN, alpha0, phi_aligned, zeta_polariz


# TODO: add input and output types
def SpinWeightedY(theta, phi, s, l, m):
    "copied from SphericalHarmonics.c in LAL"
    if s == -2:
        if l == 2:
            if m == -2:
                fac = (
                    torch.sqrt(torch.tensor(5.0 / (64.0 * PI)))
                    * (1.0 - torch.cos(theta))
                    * (1.0 - torch.cos(theta))
                )
            elif m == -1:
                fac = (
                    torch.sqrt(torch.tensor(5.0 / (16.0 * PI)))
                    * torch.sin(theta)
                    * (1.0 - torch.cos(theta))
                )
            elif m == 0:
                fac = (
                    torch.sqrt(torch.tensor(15.0 / (32.0 * PI)))
                    * torch.sin(theta)
                    * torch.sin(theta)
                )
            elif m == 1:
                fac = (
                    torch.sqrt(torch.tensor(5.0 / (16.0 * PI)))
                    * torch.sin(theta)
                    * (1.0 + torch.cos(theta))
                )
            elif m == 2:
                fac = (
                    torch.sqrt(torch.tensor(5.0 / (64.0 * PI)))
                    * (1.0 + torch.cos(theta))
                    * (1.0 + torch.cos(theta))
                )
            else:
                raise ValueError(
                    f"Invalid mode s={s}, l={l}, m={m} - require |m| <= l"
                )
            return fac * torch.complex(torch.cos(torch.tensor(m * phi)), torch.sin(torch.tensor(m * phi)))


def WignerdCoefficients(
    v: Tensor, SL: Tensor, eta: Tensor, Sp: Tensor
) -> Tuple[Tensor, Tensor]:
    # We define the shorthand s := Sp / (L + SL)
    L = L2PNR(v, eta)
    s = Sp / (L + SL)
    s2 = s**2
    cos_beta = 1.0 / (1.0 + s2) ** 0.5
    cos_beta_half = ((1.0 + cos_beta) / 2.0) ** 0.5  # cos(beta/2)
    sin_beta_half = ((1.0 - cos_beta) / 2.0) ** 0.5  # sin(beta/2)

    return cos_beta_half, sin_beta_half


def ComputeNNLOanglecoeffs(
    q: Tensor, chil: Tensor, chip: Tensor
) -> dict[str, Tensor]:
    m2 = q / (1.0 + q)
    m1 = 1.0 / (1.0 + q)
    dm = m1 - m2
    mtot = 1.0
    eta = m1 * m2  # mtot = 1
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    mtot2 = mtot * mtot
    mtot4 = mtot2 * mtot2
    mtot6 = mtot4 * mtot2
    mtot8 = mtot6 * mtot2
    chil2 = chil * chil
    chip2 = chip * chip
    chip4 = chip2 * chip2
    dm2 = dm * dm
    dm3 = dm2 * dm
    m2_2 = m2 * m2
    m2_3 = m2_2 * m2
    m2_4 = m2_3 * m2
    m2_5 = m2_4 * m2
    m2_6 = m2_5 * m2
    m2_7 = m2_6 * m2
    m2_8 = m2_7 * m2

    angcoeffs = {}
    angcoeffs["alphacoeff1"] = -0.18229166666666666 - (5 * dm) / (64.0 * m2)

    angcoeffs["alphacoeff2"] = (-15 * dm * m2 * chil) / (
        128.0 * mtot2 * eta
    ) - (35 * m2_2 * chil) / (128.0 * mtot2 * eta)

    angcoeffs["alphacoeff3"] = (
        -1.7952473958333333
        - (4555 * dm) / (7168.0 * m2)
        - (15 * chip2 * dm * m2_3) / (128.0 * mtot4 * eta2)
        - (35 * chip2 * m2_4) / (128.0 * mtot4 * eta2)
        - (515 * eta) / 384.0
        - (15 * dm2 * eta) / (256.0 * m2_2)
        - (175 * dm * eta) / (256.0 * m2)
    )

    angcoeffs["alphacoeff4"] = (
        -(35 * PI) / 48.0
        - (5 * dm * PI) / (16.0 * m2)
        + (5 * dm2 * chil) / (16.0 * mtot2)
        + (5 * dm * m2 * chil) / (3.0 * mtot2)
        + (2545 * m2_2 * chil) / (1152.0 * mtot2)
        - (5 * chip2 * dm * m2_5 * chil) / (128.0 * mtot6 * eta3)
        - (35 * chip2 * m2_6 * chil) / (384.0 * mtot6 * eta3)
        + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
        + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
    )

    angcoeffs["alphacoeff5"] = (
        4.318908476114694
        + (27895885 * dm) / (2.1676032e7 * m2)
        - (15 * chip4 * dm * m2_7) / (512.0 * mtot8 * eta4)
        - (35 * chip4 * m2_8) / (512.0 * mtot8 * eta4)
        - (485 * chip2 * dm * m2_3) / (14336.0 * mtot4 * eta2)
        + (475 * chip2 * m2_4) / (6144.0 * mtot4 * eta2)
        + (15 * chip2 * dm2 * m2_2) / (256.0 * mtot4 * eta)
        + (145 * chip2 * dm * m2_3) / (512.0 * mtot4 * eta)
        + (575 * chip2 * m2_4) / (1536.0 * mtot4 * eta)
        + (39695 * eta) / 86016.0
        + (1615 * dm2 * eta) / (28672.0 * m2_2)
        - (265 * dm * eta) / (14336.0 * m2)
        + (955 * eta2) / 576.0
        + (15 * dm3 * eta2) / (1024.0 * m2_3)
        + (35 * dm2 * eta2) / (256.0 * m2_2)
        + (2725 * dm * eta2) / (3072.0 * m2)
        - (15 * dm * m2 * PI * chil) / (16.0 * mtot2 * eta)
        - (35 * m2_2 * PI * chil) / (16.0 * mtot2 * eta)
        + (15 * chip2 * dm * m2_7 * chil2) / (128.0 * mtot8 * eta4)
        + (35 * chip2 * m2_8 * chil2) / (128.0 * mtot8 * eta4)
        + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
        + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
        + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
    )

    angcoeffs["epsiloncoeff1"] = -0.18229166666666666 - (5 * dm) / (64.0 * m2)
    angcoeffs["epsiloncoeff2"] = (-15 * dm * m2 * chil) / (
        128.0 * mtot2 * eta
    ) - (35 * m2_2 * chil) / (128.0 * mtot2 * eta)
    angcoeffs["epsiloncoeff3"] = (
        -1.7952473958333333
        - (4555 * dm) / (7168.0 * m2)
        - (515 * eta) / 384.0
        - (15 * dm2 * eta) / (256.0 * m2_2)
        - (175 * dm * eta) / (256.0 * m2)
    )
    angcoeffs["epsiloncoeff4"] = (
        -(35 * PI) / 48.0
        - (5 * dm * PI) / (16.0 * m2)
        + (5 * dm2 * chil) / (16.0 * mtot2)
        + (5 * dm * m2 * chil) / (3.0 * mtot2)
        + (2545 * m2_2 * chil) / (1152.0 * mtot2)
        + (2035 * dm * m2 * chil) / (21504.0 * mtot2 * eta)
        + (2995 * m2_2 * chil) / (9216.0 * mtot2 * eta)
    )
    angcoeffs["epsiloncoeff5"] = (
        4.318908476114694
        + (27895885 * dm) / (2.1676032e7 * m2)
        + (39695 * eta) / 86016.0
        + (1615 * dm2 * eta) / (28672.0 * m2_2)
        - (265 * dm * eta) / (14336.0 * m2)
        + (955 * eta2) / 576.0
        + (15 * dm3 * eta2) / (1024.0 * m2_3)
        + (35 * dm2 * eta2) / (256.0 * m2_2)
        + (2725 * dm * eta2) / (3072.0 * m2)
        - (15 * dm * m2 * PI * chil) / (16.0 * mtot2 * eta)
        - (35 * m2_2 * PI * chil) / (16.0 * mtot2 * eta)
        + (375 * dm2 * m2_2 * chil2) / (256.0 * mtot4 * eta)
        + (1815 * dm * m2_3 * chil2) / (256.0 * mtot4 * eta)
        + (1645 * m2_4 * chil2) / (192.0 * mtot4 * eta)
    )
    return angcoeffs


def FinalSpin_inplane(
    m1: Tensor, m2: Tensor, chi1_l: Tensor, chi2_l: Tensor, chip: Tensor
) -> Tensor:
    M = m1 + m2
    eta = m1 * m2 / (M * M)
    eta2 = eta * eta
    if m1 >= m2:
        q_factor = m1 / M
        af_parallel = FinalSpin0815(eta, eta2, chi1_l, chi2_l)
    else:
        q_factor = m2 / M
        af_parallel = FinalSpin0815(eta, eta2, chi2_l, chi1_l)
    Sperp = chip * q_factor * q_factor
    af = torch.copysign(torch.ones_like(af_parallel), af_parallel) * torch.sqrt(
        Sperp * Sperp + af_parallel * af_parallel
    )
    return af


def phP_get_fRD_fdamp(m1, m2, chi1_l, chi2_l, chip) -> Tuple[Tensor, Tensor]:
    # m1 > m2 should hold here
    finspin = FinalSpin_inplane(m1, m2, chi1_l, chi2_l, chip)
    m1_s = m1 * gt
    m2_s = m2 * gt
    M_s = m1_s + m2_s
    eta_s = m1_s * m2_s / (M_s**2.0)
    eta_s2 = eta_s * eta_s
    Erad = PhenomInternal_EradRational0815(eta_s, eta_s2, chi1_l, chi2_l)
    fRD = interpolate(finspin, QNMData_a, QNMData_fring) / (1.0 - Erad)
    fdamp = interpolate(finspin, QNMData_a, QNMData_fdamp) / (1.0 - Erad)
    return fRD / M_s, fdamp / M_s


def phP_get_transition_frequencies(
    theta: Tensor,
    gamma2: Tensor,
    gamma3: Tensor,
    chip: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # m1 > m2 should hold here

    m1, m2, chi1, chi2 = theta
    M = m1 + m2
    f_RD, f_damp = phP_get_fRD_fdamp(m1, m2, chi1, chi2, chip)

    # Phase transition frequencies
    f1 = 0.018 / (M * gt)
    f2 = 0.5 * f_RD

    # Amplitude transition frequencies
    f3 = 0.014 / (M * gt)

    def f4_gammaneg_gtr_1(
        f_RD_: Tensor, f_damp_: Tensor, gamma3_: Tensor, gamma2_: Tensor
    ) -> Tensor:
        return torch.abs(f_RD_ + (-f_damp_ * gamma3_) / gamma2_)

    def f4_gammaneg_less_1(
        f_RD_: Tensor, f_damp_: Tensor, gamma3_: Tensor, gamma2_: Tensor
    ) -> Tensor:
        return torch.abs(
            f_RD_
            + (f_damp_ * (-1 + torch.sqrt(1 - (gamma2_) ** 2.0)) * gamma3_)
            / gamma2_
        )

    f4 = Tensor(
        torch.cond(
            gamma2 >= 1,
            f4_gammaneg_gtr_1,
            f4_gammaneg_less_1,
            (
                f_RD,
                f_damp,
                gamma3,
                gamma2,
            ),
        )
    )
    return f1, f2, f3, f4, f_RD, f_damp

def get_Amp0(fM_s: Tensor, eta: Tensor) -> Tensor:
    Amp0 = (
        (2.0 / 3.0 * eta) ** (1.0 / 2.0) * (fM_s) ** (-7.0 / 6.0) * PI ** (-1.0 / 6.0)
    )
    return Amp0