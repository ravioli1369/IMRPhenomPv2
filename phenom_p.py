from phenom_p_utils import *
from phenom_d import phenom_d_phase, phenom_d_amp, rho1_fun, chiPN, fring_fdamp

def PhenomPCoreTwistUp(
    fHz: Tensor,
    hPhenom: Tensor,
    eta: Tensor,
    chi1_l: Tensor,
    chi2_l: Tensor,
    chip: Tensor,
    M: Tensor,
    angcoeffs: dict[str, Tensor],
    Y2m: Tensor,
    alphaoffset: Tensor,
    epsilonoffset: Tensor,
) -> Tuple[Tensor, Tensor]:
    assert angcoeffs is not None
    assert Y2m is not None
    f = fHz * gt * M  # Frequency in geometric units
    q = (1.0 + torch.sqrt(1.0 - 4.0 * eta) - 2.0 * eta) / (2.0 * eta)
    m1 = 1.0 / (1.0 + q)  # Mass of the smaller BH for unit total mass M=1.
    m2 = q / (1.0 + q)  # Mass of the larger BH for unit total mass M=1.
    Sperp = chip * (
        m2 * m2
    )  # Dimensionfull spin component in the orbital plane. S_perp = S_2_perp
    # chi_eff = m1 * chi1_l + m2 * chi2_l  # effective spin for M=1

    SL = chi1_l * m1 * m1 + chi2_l * m2 * m2  # Dimensionfull aligned spin.

    omega = PI * f
    logomega = torch.log(omega)
    omega_cbrt = (omega) ** (1 / 3)
    omega_cbrt2 = omega_cbrt * omega_cbrt

    alpha = (
        angcoeffs["alphacoeff1"] / omega
        + angcoeffs["alphacoeff2"] / omega_cbrt2
        + angcoeffs["alphacoeff3"] / omega_cbrt
        + angcoeffs["alphacoeff4"] * logomega
        + angcoeffs["alphacoeff5"] * omega_cbrt
    ) - alphaoffset

    epsilon = (
        angcoeffs["epsiloncoeff1"] / omega
        + angcoeffs["epsiloncoeff2"] / omega_cbrt2
        + angcoeffs["epsiloncoeff3"] / omega_cbrt
        + angcoeffs["epsiloncoeff4"] * logomega
        + angcoeffs["epsiloncoeff5"] * omega_cbrt
    ) - epsilonoffset

    cBetah, sBetah = WignerdCoefficients(omega_cbrt, SL, eta, Sperp)
    
    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah2 * cBetah
    cBetah4 = cBetah3 * cBetah
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah2 * sBetah
    sBetah4 = sBetah3 * sBetah

    hp_sum = 0
    hc_sum = 0

    cexp_i_alpha = torch.exp(1j * alpha)
    cexp_2i_alpha = cexp_i_alpha * cexp_i_alpha
    cexp_mi_alpha = 1.0 / cexp_i_alpha
    cexp_m2i_alpha = cexp_mi_alpha * cexp_mi_alpha
    T2m = (
        cexp_2i_alpha * cBetah4 * Y2m[0]
        - cexp_i_alpha * 2 * cBetah3 * sBetah * Y2m[1]
        + 1 * torch.sqrt(torch.tensor(6)) * sBetah2 * cBetah2 * Y2m[2]
        - cexp_mi_alpha * 2 * cBetah * sBetah3 * Y2m[3]
        + cexp_m2i_alpha * sBetah4 * Y2m[4]
    )
    Tm2m = (
        cexp_m2i_alpha * sBetah4 * torch.conj(Y2m[0])
        + cexp_mi_alpha * 2 * cBetah * sBetah3 * torch.conj(Y2m[1])
        + 1 * torch.sqrt(torch.tensor(6)) * sBetah2 * cBetah2 * torch.conj(Y2m[2])
        + cexp_i_alpha * 2 * cBetah3 * sBetah * torch.conj(Y2m[3])
        + cexp_2i_alpha * cBetah4 * torch.conj(Y2m[4])
    )
    hp_sum = T2m + Tm2m
    hc_sum = 1j * (T2m - Tm2m)
    
    eps_phase_hP = torch.exp(-2j * epsilon) * hPhenom / 2.0

    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return hp, hc

def PhenomPOneFrequency(
    fs, m1, m2, eta, eta2, Seta, chi1, chi2, chi12, chi22, chip, phic, M, xi, dist_mpc):
    """
    m1, m2: in solar masses
    phic: Orbital phase at the peak of the underlying non precessing model (rad)
    M: Total mass (Solar masses)
    """
    
    M_s = M * gt
    Mf = torch.outer(M_s, fs)
    fRD, _ = phP_get_fRD_fdamp(m2, m1, chi2, chi1, chip)
    MfRD = torch.outer(M_s, fRD)
    
    phase, _ = phenom_d_phase(Mf, m1, m2, eta, eta2, chi1, chi2, xi)
    Dphase = -phenom_d_phase(MfRD, m1, m2, eta, eta2, chi1, chi2, xi)[1] * M_s
    phase -= phic + PI/4.0
    Amp = phenom_d_amp(Mf, m1, m2, eta, eta2, Seta, chi1, chi2, chi12, chi22, xi, dist_mpc)[0]
    Amp0 = get_Amp0(Mf, eta)
    dist_s = dist_mpc * MPC_SEC
    Amp = Amp0 * Amp * (M_s**2.0) / dist_s

    # phase -= 2. * phic; # line 1316 ???
    hPhenom = Amp * (torch.exp(-1j * phase))
    return hPhenom, Dphase

def IMRPhenomPv2(
    fs: Tensor,
    m1: Tensor,
    m2: Tensor,
    s1x: Tensor,
    s1y: Tensor,
    s1z: Tensor,
    s2x: Tensor,
    s2y: Tensor,
    s2z: Tensor,
    dist_mpc: Tensor,
    tc: Tensor,
    phiRef: Tensor,
    incl: Tensor,
    f_ref: Tensor,
):
    """
    Thetas are waveform parameters.
    m1 must be larger than m2.
    """

    # # flip m1 m2. For some reason LAL uses this convention for PhenomPv2
    m1, m2 = m2, m1
    s1x, s2x = s2x, s1x
    s1y, s2y = s2y, s1y
    s1z, s2z = s2z, s1z
    
    
    (
        chi1_l,
        chi2_l,
        chip,
        thetaJN,
        alpha0,
        phi_aligned,
        zeta_polariz,
    ) = convert_spins(m1, m2, f_ref, phiRef, incl, s1x, s1y, s1z, s2x, s2y, s2z)

    phic = 2 * phi_aligned
    q = m2 / m1  # q>=1
    M = m1 + m2
    chi_eff = (m1 * chi1_l + m2 * chi2_l) / M
    chil = (1.0 + q) / q * chi_eff
    eta = m1 * m2 / (M * M)
    eta2 = eta * eta
    Seta = torch.sqrt(1.0 - 4.0 * eta)
    chi = chiPN(Seta, eta, chi1_l, chi2_l)
    chi22 = chi2_l * chi2_l
    chi12 = chi1_l * chi1_l
    xi = -1.0 + chi
    m_sec = M * gt
    piM = PI * m_sec

    omega_ref = piM * f_ref
    logomega_ref = torch.log(omega_ref)
    omega_ref_cbrt = (piM * f_ref) ** (1 / 3)  # == v0
    omega_ref_cbrt2 = omega_ref_cbrt * omega_ref_cbrt

    angcoeffs = ComputeNNLOanglecoeffs(q, chil, chip)

    alphaNNLOoffset = (
        angcoeffs["alphacoeff1"] / omega_ref
        + angcoeffs["alphacoeff2"] / omega_ref_cbrt2
        + angcoeffs["alphacoeff3"] / omega_ref_cbrt
        + angcoeffs["alphacoeff4"] * logomega_ref
        + angcoeffs["alphacoeff5"] * omega_ref_cbrt
    )

    epsilonNNLOoffset = (
        angcoeffs["epsiloncoeff1"] / omega_ref
        + angcoeffs["epsiloncoeff2"] / omega_ref_cbrt2
        + angcoeffs["epsiloncoeff3"] / omega_ref_cbrt
        + angcoeffs["epsiloncoeff4"] * logomega_ref
        + angcoeffs["epsiloncoeff5"] * omega_ref_cbrt
    )

    Y2m2 = SpinWeightedY(thetaJN, 0, -2, 2, -2)
    Y2m1 = SpinWeightedY(thetaJN, 0, -2, 2, -1)
    Y20 = SpinWeightedY(thetaJN, 0, -2, 2, -0)
    Y21 = SpinWeightedY(thetaJN, 0, -2, 2, 1)
    Y22 = SpinWeightedY(thetaJN, 0, -2, 2, 2)
    Y2 = torch.tensor([Y2m2, Y2m1, Y20, Y21, Y22])
    


    hPhenomDs, Dphase = PhenomPOneFrequency(
        fs, m2, m1, eta, eta2, Seta, chi2_l, chi1_l, chi12, chi22, chip, phic, M, xi, dist_mpc
    )

    hp, hc = PhenomPCoreTwistUp(
        fs,
        hPhenomDs,
        eta,
        chi1_l,
        chi2_l,
        chip,
        M,
        angcoeffs,
        Y2,
        alphaNNLOoffset - alpha0,
        epsilonNNLOoffset,
    )
    t0 = (Dphase) / (2 * PI)
    phase_corr = torch.cos(2 * PI * fs * (t0)) - 1j * torch.sin(2 * PI * fs * (t0))
    M_s = (m1 + m2) * gt
    phase_corr_tc = torch.exp(-1j * fs * M_s * tc)
    hp *= phase_corr * phase_corr_tc
    hc *= phase_corr * phase_corr_tc

    c2z = torch.cos(2 * zeta_polariz)
    s2z = torch.sin(2 * zeta_polariz)
    hplus = c2z * hp + s2z * hc
    hcross = c2z * hc - s2z * hp
    return hplus, hcross