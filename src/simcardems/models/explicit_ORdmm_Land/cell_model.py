# flake8: noqa: F841
import math

import numpy

from .em_model import EMCoupling


def parameter_index(name: str) -> int:
    """Return the index of the parameter with the given name

    Arguments
    ---------
    name : str
        The name of the parameter

    Returns
    -------
    int
        The index of the parameter

    Raises
    ------
    KeyError
        If the name is not a valid parameter
    """

    data = {
        "Aff": 0,
        "Ahf": 1,
        "BSLmax": 2,
        "BSRmax": 3,
        "Beta0": 4,
        "Beta1": 5,
        "CaMKo": 6,
        "Esac_ns": 7,
        "F": 8,
        "GKb": 9,
        "GNa": 10,
        "Gncx": 11,
        "GpCa": 12,
        "Gsac_k": 13,
        "Gsac_ns": 14,
        "Gto": 15,
        "H": 16,
        "Khp": 17,
        "Kki": 18,
        "Kko": 19,
        "KmBSL": 20,
        "KmBSR": 21,
        "KmCaAct": 22,
        "KmCaM": 23,
        "KmCaMK": 24,
        "Kmgatp": 25,
        "Kmn": 26,
        "Knai0": 27,
        "Knao0": 28,
        "Knap": 29,
        "Kxkur": 30,
        "L": 31,
        "MgADP": 32,
        "MgATP": 33,
        "PCab": 34,
        "PKNa": 35,
        "PNab": 36,
        "Pnak": 37,
        "R": 38,
        "T": 39,
        "Tot_A": 40,
        "Tref": 41,
        "Trpn50": 42,
        "aCaMK": 43,
        "amp": 44,
        "bCaMK": 45,
        "bt": 46,
        "calib": 47,
        "cao": 48,
        "cat50_ref": 49,
        "celltype": 50,
        "cmdnmax": 51,
        "csqnmax": 52,
        "dLambda": 53,
        "delta": 54,
        "delta_epi": 55,
        "duration": 56,
        "eP": 57,
        "emcoupling": 58,
        "etal": 59,
        "etas": 60,
        "gammas": 61,
        "gammaw": 62,
        "isacs": 63,
        "k1m": 64,
        "k1p": 65,
        "k2m": 66,
        "k2n": 67,
        "k2p": 68,
        "k3m": 69,
        "k3p": 70,
        "k4m": 71,
        "k4p": 72,
        "kasymm": 73,
        "kcaoff": 74,
        "kcaon": 75,
        "kmcmdn": 76,
        "kmcsqn": 77,
        "kmtrpn": 78,
        "kna1": 79,
        "kna2": 80,
        "kna3": 81,
        "ko": 82,
        "ktrpn": 83,
        "ku": 84,
        "kuw": 85,
        "kws": 86,
        "lambda_max": 87,
        "lmbda": 88,
        "mode": 89,
        "nao": 90,
        "ntm": 91,
        "ntrpn": 92,
        "p_a": 93,
        "p_b": 94,
        "p_k": 95,
        "phi": 96,
        "qca": 97,
        "qna": 98,
        "rad": 99,
        "rs": 100,
        "rw": 101,
        "scale_HF_CaMKa": 102,
        "scale_HF_GK1": 103,
        "scale_HF_GNaL": 104,
        "scale_HF_Gncx": 105,
        "scale_HF_Gto": 106,
        "scale_HF_Jleak": 107,
        "scale_HF_Jrel_inf": 108,
        "scale_HF_Jup": 109,
        "scale_HF_Pnak": 110,
        "scale_HF_cat50_ref": 111,
        "scale_HF_thL": 112,
        "scale_ICaL": 113,
        "scale_IK1": 114,
        "scale_IKr": 115,
        "scale_IKs": 116,
        "scale_INaL": 117,
        "scale_drug_ICaL": 118,
        "scale_drug_ICab": 119,
        "scale_drug_IK1": 120,
        "scale_drug_IKb": 121,
        "scale_drug_IKr": 122,
        "scale_drug_IKs": 123,
        "scale_drug_INa": 124,
        "scale_drug_INaL": 125,
        "scale_drug_INab": 126,
        "scale_drug_IpCa": 127,
        "scale_drug_Isack": 128,
        "scale_drug_Isacns": 129,
        "scale_drug_Ito": 130,
        "thL": 131,
        "tjca": 132,
        "trpnmax": 133,
        "wca": 134,
        "wna": 135,
        "wnaca": 136,
        "zca": 137,
        "zk": 138,
    }
    return data[name]


def state_index(name: str) -> int:
    """Return the index of the state with the given name

    Arguments
    ---------
    name : str
        The name of the state

    Returns
    -------
    int
        The index of the state

    Raises
    ------
    KeyError
        If the name is not a valid state
    """

    data = {
        "hL": 0,
        "a": 1,
        "ap": 2,
        "d": 3,
        "ff": 4,
        "fs": 5,
        "hf": 6,
        "hs": 7,
        "m": 8,
        "xrf": 9,
        "xrs": 10,
        "xs1": 11,
        "CaMKt": 12,
        "xk1": 13,
        "Zetaw": 14,
        "XW": 15,
        "TmB": 16,
        "hLp": 17,
        "iF": 18,
        "iS": 19,
        "fcaf": 20,
        "fcas": 21,
        "jca": 22,
        "j": 23,
        "fcafp": 24,
        "ffp": 25,
        "hsp": 26,
        "jp": 27,
        "mL": 28,
        "xs2": 29,
        "Zetas": 30,
        "nca": 31,
        "CaTrpn": 32,
        "XS": 33,
        "iFp": 34,
        "iSp": 35,
        "cajsr": 36,
        "cansr": 37,
        "Cd": 38,
        "kss": 39,
        "Jrelnp": 40,
        "Jrelp": 41,
        "ki": 42,
        "cass": 43,
        "nass": 44,
        "cai": 45,
        "nai": 46,
        "v": 47,
    }
    return data[name]


def init_parameter_values(**values):
    """Initialize parameter values"""
    # Aff=0.6, Ahf=0.99, BSLmax=1.124, BSRmax=0.047, Beta0=2.3
    # Beta1=-2.4, CaMKo=0.05, Esac_ns=-10, F=96485.0, GKb=0.003
    # GNa=31, Gncx=0.0008, GpCa=0.0005, Gsac_k=(0.2882*800)/210
    # Gsac_ns=0.006, Gto=0.02, H=1e-07, Khp=1.698e-07, Kki=0.5
    # Kko=0.3582, KmBSL=0.0087, KmBSR=0.00087, KmCaAct=0.00015
    # KmCaM=0.0015, KmCaMK=0.15, Kmgatp=1.698e-07, Kmn=0.002
    # Knai0=9.073, Knao0=27.78, Knap=224.0, Kxkur=292.0, L=0.01
    # MgADP=0.05, MgATP=9.8, PCab=2.5e-08, PKNa=0.01833
    # PNab=3.75e-10, Pnak=30, R=8314.0, T=310.0, Tot_A=25, Tref=120
    # Trpn50=0.35, aCaMK=0.05, amp=-80.0, bCaMK=0.00068, bt=4.75
    # calib=1, cao=1.8, cat50_ref=0.805, celltype=0, cmdnmax=0.05
    # csqnmax=10.0, dLambda=0, delta=-0.155, delta_epi=1.0
    # duration=0.5, eP=4.2, emcoupling=1, etal=200, etas=20
    # gammas=0.0085, gammaw=0.615, isacs=0, k1m=182.4, k1p=949.5
    # k2m=39.4, k2n=1000.0, k2p=687.2, k3m=79300.0, k3p=1899.0
    # k4m=40.0, k4p=639.0, kasymm=12.5, kcaoff=5000.0
    # kcaon=1500000.0, kmcmdn=0.00238, kmcsqn=0.8, kmtrpn=0.0005
    # kna1=15.0, kna2=5.0, kna3=88.12, ko=5.4, ktrpn=0.1, ku=0.04
    # kuw=0.182, kws=0.012, lambda_max=1.1, lmbda=1, mode=1
    # nao=140.0, ntm=2.4, ntrpn=2, p_a=2.1, p_b=9.1, p_k=7
    # phi=2.23, qca=0.167, qna=0.5224, rad=0.0011, rs=0.25, rw=0.5
    # scale_HF_CaMKa=1.0, scale_HF_GK1=1.0, scale_HF_GNaL=1.0
    # scale_HF_Gncx=1.0, scale_HF_Gto=1.0, scale_HF_Jleak=1.0
    # scale_HF_Jrel_inf=1.0, scale_HF_Jup=1.0, scale_HF_Pnak=1.0
    # scale_HF_cat50_ref=1.0, scale_HF_thL=1.0, scale_ICaL=1.018
    # scale_IK1=1.414, scale_IKr=1.119, scale_IKs=1.648
    # scale_INaL=2.274, scale_drug_ICaL=1.0, scale_drug_ICab=1.0
    # scale_drug_IK1=1.0, scale_drug_IKb=1.0, scale_drug_IKr=1.0
    # scale_drug_IKs=1.0, scale_drug_INa=1.0, scale_drug_INaL=1.0
    # scale_drug_INab=1.0, scale_drug_IpCa=1.0
    # scale_drug_Isack=1.0, scale_drug_Isacns=1.0
    # scale_drug_Ito=1.0, thL=200.0, tjca=75.0, trpnmax=0.07
    # wca=60000.0, wna=60000.0, wnaca=5000.0, zca=2.0, zk=1.0

    parameters = numpy.array(
        [
            0.6,
            0.99,
            1.124,
            0.047,
            2.3,
            -2.4,
            0.05,
            -10,
            96485.0,
            0.003,
            31,
            0.0008,
            0.0005,
            (0.2882 * 800) / 210,
            0.006,
            0.02,
            1e-07,
            1.698e-07,
            0.5,
            0.3582,
            0.0087,
            0.00087,
            0.00015,
            0.0015,
            0.15,
            1.698e-07,
            0.002,
            9.073,
            27.78,
            224.0,
            292.0,
            0.01,
            0.05,
            9.8,
            2.5e-08,
            0.01833,
            3.75e-10,
            30,
            8314.0,
            310.0,
            25,
            120,
            0.35,
            0.05,
            -80.0,
            0.00068,
            4.75,
            1,
            1.8,
            0.805,
            0,
            0.05,
            10.0,
            0,
            -0.155,
            1.0,
            0.5,
            4.2,
            1,
            200,
            20,
            0.0085,
            0.615,
            0,
            182.4,
            949.5,
            39.4,
            1000.0,
            687.2,
            79300.0,
            1899.0,
            40.0,
            639.0,
            12.5,
            5000.0,
            1500000.0,
            0.00238,
            0.8,
            0.0005,
            15.0,
            5.0,
            88.12,
            5.4,
            0.1,
            0.04,
            0.182,
            0.012,
            1.1,
            1,
            1,
            140.0,
            2.4,
            2,
            2.1,
            9.1,
            7,
            2.23,
            0.167,
            0.5224,
            0.0011,
            0.25,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.018,
            1.414,
            1.119,
            1.648,
            2.274,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            200.0,
            75.0,
            0.07,
            60000.0,
            60000.0,
            5000.0,
            2.0,
            1.0,
        ],
    )

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # hL=1, a=0, ap=0, d=0, ff=1, fs=1, hf=1, hs=1, m=0, xrf=0
    # xrs=0, xs1=0, CaMKt=0, xk1=1, Zetaw=0, XW=0, TmB=1, hLp=1
    # iF=1, iS=1, fcaf=1, fcas=1, jca=1, j=1, fcafp=1, ffp=1, hsp=1
    # jp=1, mL=0, xs2=0, Zetas=0, nca=0, CaTrpn=0, XS=0, iFp=1
    # iSp=1, cajsr=1.2, cansr=1.2, Cd=0, kss=145, Jrelnp=0, Jrelp=0
    # ki=145, cass=0.0001, nass=7, cai=0.0001, nai=7, v=-87

    states = numpy.array(
        [
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1.2,
            1.2,
            0,
            145,
            0,
            0,
            145,
            0.0001,
            7,
            0.0001,
            7,
            -87,
        ],
    )

    for key, value in values.items():
        states[state_index(key)] = value

    return states


def rhs(t, states, parameters):
    # Assign states
    hL = states[0]
    a = states[1]
    ap = states[2]
    d = states[3]
    ff = states[4]
    fs = states[5]
    hf = states[6]
    hs = states[7]
    m = states[8]
    xrf = states[9]
    xrs = states[10]
    xs1 = states[11]
    CaMKt = states[12]
    xk1 = states[13]
    Zetaw = states[14]
    XW = states[15]
    TmB = states[16]
    hLp = states[17]
    iF = states[18]
    iS = states[19]
    fcaf = states[20]
    fcas = states[21]
    jca = states[22]
    j = states[23]
    fcafp = states[24]
    ffp = states[25]
    hsp = states[26]
    jp = states[27]
    mL = states[28]
    xs2 = states[29]
    Zetas = states[30]
    nca = states[31]
    CaTrpn = states[32]
    XS = states[33]
    iFp = states[34]
    iSp = states[35]
    cajsr = states[36]
    cansr = states[37]
    Cd = states[38]
    kss = states[39]
    Jrelnp = states[40]
    Jrelp = states[41]
    ki = states[42]
    cass = states[43]
    nass = states[44]
    cai = states[45]
    nai = states[46]
    v = states[47]

    # Assign parameters
    Aff = parameters[0]
    Ahf = parameters[1]
    BSLmax = parameters[2]
    BSRmax = parameters[3]
    Beta0 = parameters[4]
    Beta1 = parameters[5]
    CaMKo = parameters[6]
    Esac_ns = parameters[7]
    F = parameters[8]
    GKb = parameters[9]
    GNa = parameters[10]
    Gncx = parameters[11]
    GpCa = parameters[12]
    Gsac_k = parameters[13]
    Gsac_ns = parameters[14]
    Gto = parameters[15]
    H = parameters[16]
    Khp = parameters[17]
    Kki = parameters[18]
    Kko = parameters[19]
    KmBSL = parameters[20]
    KmBSR = parameters[21]
    KmCaAct = parameters[22]
    KmCaM = parameters[23]
    KmCaMK = parameters[24]
    Kmgatp = parameters[25]
    Kmn = parameters[26]
    Knai0 = parameters[27]
    Knao0 = parameters[28]
    Knap = parameters[29]
    Kxkur = parameters[30]
    L = parameters[31]
    MgADP = parameters[32]
    MgATP = parameters[33]
    PCab = parameters[34]
    PKNa = parameters[35]
    PNab = parameters[36]
    Pnak = parameters[37]
    R = parameters[38]
    T = parameters[39]
    Tot_A = parameters[40]
    Tref = parameters[41]
    Trpn50 = parameters[42]
    aCaMK = parameters[43]
    amp = parameters[44]
    bCaMK = parameters[45]
    bt = parameters[46]
    calib = parameters[47]
    cao = parameters[48]
    cat50_ref = parameters[49]
    celltype = parameters[50]
    cmdnmax = parameters[51]
    csqnmax = parameters[52]
    dLambda = parameters[53]
    delta = parameters[54]
    delta_epi = parameters[55]
    duration = parameters[56]
    eP = parameters[57]
    emcoupling = parameters[58]
    etal = parameters[59]
    etas = parameters[60]
    gammas = parameters[61]
    gammaw = parameters[62]
    isacs = parameters[63]
    k1m = parameters[64]
    k1p = parameters[65]
    k2m = parameters[66]
    k2n = parameters[67]
    k2p = parameters[68]
    k3m = parameters[69]
    k3p = parameters[70]
    k4m = parameters[71]
    k4p = parameters[72]
    kasymm = parameters[73]
    kcaoff = parameters[74]
    kcaon = parameters[75]
    kmcmdn = parameters[76]
    kmcsqn = parameters[77]
    kmtrpn = parameters[78]
    kna1 = parameters[79]
    kna2 = parameters[80]
    kna3 = parameters[81]
    ko = parameters[82]
    ktrpn = parameters[83]
    ku = parameters[84]
    kuw = parameters[85]
    kws = parameters[86]
    lambda_max = parameters[87]
    lmbda = parameters[88]
    mode = parameters[89]
    nao = parameters[90]
    ntm = parameters[91]
    ntrpn = parameters[92]
    p_a = parameters[93]
    p_b = parameters[94]
    p_k = parameters[95]
    phi = parameters[96]
    qca = parameters[97]
    qna = parameters[98]
    rad = parameters[99]
    rs = parameters[100]
    rw = parameters[101]
    scale_HF_CaMKa = parameters[102]
    scale_HF_GK1 = parameters[103]
    scale_HF_GNaL = parameters[104]
    scale_HF_Gncx = parameters[105]
    scale_HF_Gto = parameters[106]
    scale_HF_Jleak = parameters[107]
    scale_HF_Jrel_inf = parameters[108]
    scale_HF_Jup = parameters[109]
    scale_HF_Pnak = parameters[110]
    scale_HF_cat50_ref = parameters[111]
    scale_HF_thL = parameters[112]
    scale_ICaL = parameters[113]
    scale_IK1 = parameters[114]
    scale_IKr = parameters[115]
    scale_IKs = parameters[116]
    scale_INaL = parameters[117]
    scale_drug_ICaL = parameters[118]
    scale_drug_ICab = parameters[119]
    scale_drug_IK1 = parameters[120]
    scale_drug_IKb = parameters[121]
    scale_drug_IKr = parameters[122]
    scale_drug_IKs = parameters[123]
    scale_drug_INa = parameters[124]
    scale_drug_INaL = parameters[125]
    scale_drug_INab = parameters[126]
    scale_drug_IpCa = parameters[127]
    scale_drug_Isack = parameters[128]
    scale_drug_Isacns = parameters[129]
    scale_drug_Ito = parameters[130]
    thL = parameters[131]
    tjca = parameters[132]
    trpnmax = parameters[133]
    wca = parameters[134]
    wna = parameters[135]
    wnaca = parameters[136]
    zca = parameters[137]
    zk = parameters[138]

    # Assign expressions
    values = numpy.zeros_like(states)
    zna = 1.0
    Afcaf = 0.3 + 0.6 / (numpy.exp((v - 10.0) / 10.0) + 1.0)
    AiF = 1.0 / (numpy.exp((v - 213.6) / 151.2) + 1.0)
    Axrf = 1.0 / (numpy.exp((v + 54.81) / 38.21) + 1.0)
    ass = 1.0 / (numpy.exp((-(v - 14.34)) / 14.82) + 1.0)
    assp = 1.0 / (numpy.exp((-(v - 24.34)) / 14.82) + 1.0)
    dss = 1.0 / (numpy.exp((-(v + 3.94)) / 4.23) + 1.0)
    dti_develop = 1.354 + 0.0001 / (
        numpy.exp((-(v - 12.23)) / 0.2154) + numpy.exp((v - 167.4) / 15.89)
    )
    dti_recover = 1.0 - 0.5 / (numpy.exp((v + 70.0) / 20.0) + 1.0)
    fss = 1.0 / (numpy.exp((v + 19.58) / 3.696) + 1.0)
    hLss = 1.0 / (numpy.exp((v + 87.61) / 7.488) + 1.0)
    hLssp = 1.0 / (numpy.exp((v + 93.81) / 7.488) + 1.0)
    hss = 1.0 / (numpy.exp((v + 78.5) / 6.22) + 1)
    hssp = 1.0 / (numpy.exp(((v + 78.5) + 6.2) / 6.22) + 1)
    iss = 1.0 / (numpy.exp((v + 43.94) / 5.711) + 1.0)
    mLss = 1.0 / (numpy.exp((-(v + 42.85)) / 5.264) + 1.0)
    mss = 1.0 / (numpy.exp((-((v + 39.57) + 9.4)) / 7.5) + 1.0)
    rkr = (1.0 * (1.0 / (numpy.exp((v + 55.0) / 75.0) + 1.0))) / (
        numpy.exp((v - 10.0) / 30.0) + 1.0
    )
    ta = 1.0515 / (
        1.0 / ((1.2089 * (numpy.exp((-(v - 18.4099)) / 29.3814) + 1.0)))
        + 3.5 / (numpy.exp((v + 100.0) / 29.3814) + 1.0)
    )
    td = 0.6 + 1.0 / (numpy.exp((-0.05) * (v + 6.0)) + numpy.exp(0.09 * (v + 14.0)))
    tfcaf = 7.0 + 1.0 / (
        0.04 * numpy.exp((-(v - 4.0)) / 7.0) + 0.04 * numpy.exp((v - 4.0) / 7.0)
    )
    tfcas = 100.0 + 1.0 / (
        0.00012 * numpy.exp((-v) / 3.0) + 0.00012 * numpy.exp(v / 7.0)
    )
    tff = 7.0 + 1.0 / (
        0.0045 * numpy.exp((-(v + 20.0)) / 10.0) + 0.0045 * numpy.exp((v + 20.0) / 10.0)
    )
    tfs = 1000.0 + 1.0 / (
        3.5e-05 * numpy.exp((-(v + 5.0)) / 4.0) + 3.5e-05 * numpy.exp((v + 5.0) / 6.0)
    )
    thf = 1.0 / (
        6.149 * numpy.exp((v + 0.5096) / 20.27)
        + 1.432e-05 * numpy.exp((-(v + 1.196)) / 6.285)
    )
    ths = 1.0 / (
        0.009794 * numpy.exp((-(v + 17.95)) / 28.05)
        + 0.3343 * numpy.exp((v + 5.73) / 56.66)
    )
    tj = 2.038 + 1.0 / (
        0.3052 * numpy.exp((v + 0.9941) / 38.45)
        + 0.02136 * numpy.exp((-(v + 100.6)) / 8.281)
    )
    tm = 1.0 / (
        6.765 * numpy.exp((v + 11.64) / 34.77)
        + 8.552 * numpy.exp((-(v + 77.42)) / 5.955)
    )
    txk1 = 122.2 / (numpy.exp((-(v + 127.2)) / 20.36) + numpy.exp((v + 236.8) / 69.33))
    txrf = 12.98 + 1.0 / (
        4.123e-05 * numpy.exp((-(v - 47.78)) / 20.38)
        + 0.3652 * numpy.exp((v - 31.66) / 3.869)
    )
    txrs = 1.865 + 1.0 / (
        1.128e-05 * numpy.exp((-(v - 29.74)) / 25.94)
        + 0.06629 * numpy.exp((v - 34.7) / 7.355)
    )
    txs1 = 817.3 + 1.0 / (
        0.0002326 * numpy.exp((v + 48.28) / 17.8)
        + 0.001292 * numpy.exp((-(v + 210.0)) / 230.0)
    )
    txs2 = 1.0 / (
        0.01 * numpy.exp((v - 50.0) / 20.0) + 0.0193 * numpy.exp((-(v + 66.54)) / 31.0)
    )
    xkb = 1.0 / (numpy.exp((-(v - 14.48)) / 18.34) + 1.0)
    xrss = 1.0 / (numpy.exp((-(v + 8.337)) / 6.789) + 1.0)
    xs1ss = 1.0 / (numpy.exp((-(v + 11.6)) / 8.932) + 1.0)
    Afs = 1.0 - Aff
    Ageo = L * ((2 * 3.14) * rad) + rad * ((2 * 3.14) * rad)
    vcell = L * (rad * ((3.14 * 1000) * rad))
    Ahs = 1.0 - Ahf
    Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
    Bcai = 1.0 / ((cmdnmax * kmcmdn) / (cai + kmcmdn) ** 2.0 + 1.0)
    Jupnp = (0.004375 * cai) / (cai + 0.00092)
    Jupp = ((0.004375 * 2.75) * cai) / ((cai + 0.00092) - 0.00017)
    KsCa = 1.0 + 0.6 / ((3.8e-05 / cai) ** 1.4 + 1.0)
    Bcajsr = 1.0 / ((csqnmax * kmcsqn) / (cajsr + kmcsqn) ** 2.0 + 1.0)
    Bcass = 1.0 / (
        (BSLmax * KmBSL) / (KmBSL + cass) ** 2.0
        + ((BSRmax * KmBSR) / (KmBSR + cass) ** 2.0 + 1.0)
    )
    Jdiff = (-cai + cass) / 0.2
    CaMKb = (CaMKo * (1.0 - CaMKt)) / (KmCaM / cass + 1.0)
    CaTrpn_max = numpy.where((CaTrpn > 0), CaTrpn, 0)
    rk1 = 1.0 / (numpy.exp((-2.6 * ko + (v + 105.8)) / 9.493) + 1.0)
    xk1ss = 1.0 / (
        numpy.exp((-((2.5538 * ko + v) + 144.59)) / (1.5692 * ko + 3.8115)) + 1.0
    )
    vffrt = (F * (F * v)) / ((R * T))
    vfrt = (F * v) / ((R * T))
    EK = ((R * T) / F) * numpy.log(ko / ki)
    EKs = ((R * T) / F) * numpy.log((PKNa * nao + ko) / (PKNa * nai + ki))
    ENa = ((R * T) / F) * numpy.log(nao / nai)
    GK1 = scale_HF_GK1 * ((0.1908 * scale_IK1) * scale_drug_IK1)
    GKr = (0.046 * scale_IKr) * scale_drug_IKr
    GKs = (0.0034 * scale_IKs) * scale_drug_IKs
    GNaL = scale_HF_GNaL * ((0.0075 * scale_INaL) * scale_drug_INaL)
    km2n = 1.0 * jca
    IpCa = (cai * (GpCa * scale_drug_IpCa)) / (cai + 0.0005)
    lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
    Isac_P_k = numpy.where(
        (lmbda < 1.0),
        0.0,
        (Gsac_k * ((lmbda - 1.0) / (lambda_max - 1.0)))
        * (1.0 / (numpy.exp((19.05 - v) / 29.98) + 1.0)),
    )
    Isac_P_ns = numpy.where(
        (lmbda < 1.0),
        0.0,
        (Gsac_ns * ((lmbda - 1.0) / (lambda_max - 1.0))) * (-Esac_ns + v),
    )
    Istim = numpy.where((duration >= t), amp, 0)
    JdiffK = (-ki + kss) / 2.0
    JdiffNa = (-nai + nass) / 2.0
    Jtr = (-cajsr + cansr) / 100.0
    Jleak = ((0.0039375 * cansr) * scale_HF_Jleak) / 15.0
    Knai = Knai0 * numpy.exp((F * (delta * v)) / (((3.0 * R) * T)))
    Knao = Knao0 * numpy.exp((F * (v * (1.0 - delta))) / (((3.0 * R) * T)))
    P = eP / (((H / Khp + 1.0) + nai / Knap) + ki / Kxkur)
    PCa = (0.0001 * scale_ICaL) * scale_drug_ICaL
    XS_max = numpy.where((XS > 0), XS, 0)
    zetas1 = Zetas * numpy.where((Zetas > 0), 1.0, 0.0)
    zetas2 = (-Zetas - 1.0) * numpy.where((Zetas < -1.0), 1.0, 0.0)
    XW_max = numpy.where((XW > 0), XW, 0)
    XU = -XW + (-XS + (1 - TmB))
    a2 = k2p
    a4 = ((MgATP * k4p) / Kmgatp) / (1.0 + MgATP / Kmgatp)
    a_rel = 0.5 * bt
    btp = 1.25 * bt
    tau_rel_tmp = bt / (1.0 + 0.0123 / cajsr)
    allo_i = 1.0 / ((KmCaAct / cai) ** 2.0 + 1.0)
    allo_ss = 1.0 / ((KmCaAct / cass) ** 2.0 + 1.0)
    b1 = MgADP * k1m
    cs = ((kws * phi) * (rw * (1 - rs))) / rs
    ksu = (kws * rw) * (-1 + 1 / rs)
    cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
    kwu = kuw * (-1 + 1 / rw) - kws
    gammawu = gammaw * numpy.abs(Zetaw)
    h10 = (nao / kna1) * (1 + nao / kna2) + (kasymm + 1.0)
    h10_i = (nao / kna1) * (1.0 + nao / kna2) + (kasymm + 1.0)
    h4 = (nass / kna1) * (1 + nass / kna2) + 1.0
    h4_i = (nai / kna1) * (1 + nai / kna2) + 1.0
    hca = numpy.exp((F * (qca * v)) / ((R * T)))
    hna = numpy.exp((F * (qna * v)) / ((R * T)))
    k2 = kcaoff
    k2_i = kcaoff
    k5 = kcaoff
    k5_i = kcaoff
    kb = (Trpn50**ntm * ku) / (-rw * (1 - rs) + (1 - rs))
    thLp = scale_HF_thL * (3.0 * thL)
    tiF = (
        delta_epi
        * (
            1
            / (
                0.3933 * numpy.exp((-(v + 100.0)) / 100.0)
                + 0.08004 * numpy.exp((v + 50.0) / 16.59)
            )
        )
        + 4.562
    )
    tiS = (
        delta_epi
        * (
            1
            / (
                0.001416 * numpy.exp((-(v + 96.52)) / 59.05)
                + 1.78e-08 * numpy.exp((v + 114.1) / 8.079)
            )
        )
        + 23.62
    )
    Afcas = 1.0 - Afcaf
    AiS = 1.0 - AiF
    Axrs = 1.0 - Axrf
    fcass = fss
    dhL_dt = (-hL + hLss) / ((scale_HF_thL * thL))
    values[0] = dhL_dt
    jss = hss
    da_dt = (-a + ass) / ta
    values[1] = da_dt
    dap_dt = (-ap + assp) / ta
    values[2] = dap_dt
    dd_dt = (-d + dss) / td
    values[3] = dd_dt
    tfcafp = 2.5 * tfcaf
    tffp = 2.5 * tff
    dff_dt = (-ff + fss) / tff
    values[4] = dff_dt
    dfs_dt = (-fs + fss) / tfs
    values[5] = dfs_dt
    dhf_dt = (-hf + hss) / thf
    values[6] = dhf_dt
    thsp = 3.0 * ths
    dhs_dt = (-hs + hss) / ths
    values[7] = dhs_dt
    tjp = 1.46 * tj
    tmL = tm
    dm_dt = (-m + mss) / tm
    values[8] = dm_dt
    dxrf_dt = (-xrf + xrss) / txrf
    values[9] = dxrf_dt
    dxrs_dt = (-xrs + xrss) / txrs
    values[10] = dxrs_dt
    xs2ss = xs1ss
    dxs1_dt = (-xs1 + xs1ss) / txs1
    values[11] = dxs1_dt
    f = Aff * ff + Afs * fs
    fp = Aff * ffp + Afs * fs
    Acap = 2 * Ageo
    vjsr = 0.0048 * vcell
    vmyo = 0.68 * vcell
    vnsr = 0.0552 * vcell
    vss = 0.02 * vcell
    h = Ahf * hf + Ahs * hs
    hp = Ahf * hf + Ahs * hsp
    As = Aw
    CaMKa = scale_HF_CaMKa * (CaMKb + CaMKt)
    dCaMKt_dt = -CaMKt * bCaMK + (CaMKb * aCaMK) * (CaMKb + CaMKt)
    values[12] = dCaMKt_dt
    dxk1_dt = (-xk1 + xk1ss) / txk1
    values[13] = dxk1_dt
    ICab = (
        (vffrt * (4.0 * (PCab * scale_drug_ICab)))
        * (cai * numpy.exp(2.0 * vfrt) - 0.341 * cao)
    ) / (numpy.exp(2.0 * vfrt) - 1.0)
    INab = ((vffrt * (PNab * scale_drug_INab)) * (nai * numpy.exp(vfrt) - nao)) / (
        numpy.exp(vfrt) - 1.0
    )
    PhiCaK = ((1.0 * vffrt) * (-0.75 * ko + (0.75 * kss) * numpy.exp(1.0 * vfrt))) / (
        numpy.exp(1.0 * vfrt) - 1.0
    )
    PhiCaL = ((4.0 * vffrt) * (-0.341 * cao + cass * numpy.exp(2.0 * vfrt))) / (
        numpy.exp(2.0 * vfrt) - 1.0
    )
    PhiCaNa = (
        (1.0 * vffrt) * (-0.75 * nao + (0.75 * nass) * numpy.exp(1.0 * vfrt))
    ) / (numpy.exp(1.0 * vfrt) - 1.0)
    IKb = (xkb * (GKb * scale_drug_IKb)) * (-EK + v)
    IK1 = (xk1 * (rk1 * (GK1 * math.sqrt(ko)))) * (-EK + v)
    IKs = (xs2 * (xs1 * (GKs * KsCa))) * (-EKs + v)
    anca = 1.0 / (k2n / km2n + (Kmn / cass + 1.0) ** 4.0)
    C = lambda_min12 - 1
    cat50 = scale_HF_cat50_ref * (Beta1 * (lambda_min12 - 1) + cat50_ref)
    lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
    a1 = (k1p * (nai / Knai) ** 3.0) / (
        ((1.0 + ki / Kki) ** 2.0 + (1.0 + nai / Knai) ** 3.0) - 1.0
    )
    b4 = (k4m * (ki / Kki) ** 2.0) / (
        ((1.0 + ki / Kki) ** 2.0 + (1.0 + nai / Knai) ** 3.0) - 1.0
    )
    a3 = (k3p * (ko / Kko) ** 2.0) / (
        ((1.0 + ko / Kko) ** 2.0 + (1.0 + nao / Knao) ** 3.0) - 1.0
    )
    b2 = (k2m * (nao / Knao) ** 3.0) / (
        ((1.0 + ko / Kko) ** 2.0 + (1.0 + nao / Knao) ** 3.0) - 1.0
    )
    b3 = (H * (P * k3m)) / (1.0 + MgATP / Kmgatp)
    PCaK = 0.0003574 * PCa
    PCaNa = 0.00125 * PCa
    PCap = 1.1 * PCa
    gammasu = gammas * numpy.where((zetas1 > zetas2), zetas1, zetas2)
    a_relp = 0.5 * btp
    tau_relp_tmp = btp / (1.0 + 0.0123 / cajsr)
    tau_rel = numpy.where((tau_rel_tmp < 0.001), 0.001, tau_rel_tmp)
    dZetaw_dt = Aw * dLambda - Zetaw * cw
    values[14] = dZetaw_dt
    dXW_dt = -XW * gammawu + (-XW * kws + (XU * kuw - XW * kwu))
    values[15] = dXW_dt
    h11 = (nao * nao) / ((kna2 * (h10 * kna1)))
    h12 = 1.0 / h10
    h11_i = (nao * nao) / ((kna2 * (h10_i * kna1)))
    h12_i = 1.0 / h10_i
    h5 = (nass * nass) / ((kna2 * (h4 * kna1)))
    h6 = 1.0 / h4
    h5_i = (nai * nai) / ((kna2 * (h4_i * kna1)))
    h6_i = 1.0 / h4_i
    h1 = (nass / kna3) * (hna + 1) + 1
    h1_i = (nai / kna3) * (hna + 1) + 1
    h7 = (nao / kna3) * (1.0 + 1.0 / hna) + 1.0
    h7_i = (nao / kna3) * (1.0 + 1.0 / hna) + 1.0
    dTmB_dt = -TmB * CaTrpn ** (ntm / 2) * ku + XU * (
        kb
        * numpy.where((CaTrpn ** (-1 / 2 * ntm) < 100), CaTrpn ** (-1 / 2 * ntm), 100)
    )
    values[16] = dTmB_dt
    dhLp_dt = (-hLp + hLssp) / thLp
    values[17] = dhLp_dt
    tiFp = tiF * (dti_develop * dti_recover)
    diF_dt = (-iF + iss) / tiF
    values[18] = diF_dt
    tiSp = tiS * (dti_develop * dti_recover)
    diS_dt = (-iS + iss) / tiS
    values[19] = diS_dt
    fca = Afcaf * fcaf + Afcas * fcas
    fcap = Afcaf * fcafp + Afcas * fcas
    i = AiF * iF + AiS * iS
    ip = AiF * iFp + AiS * iSp
    xr = Axrf * xrf + Axrs * xrs
    dfcaf_dt = (-fcaf + fcass) / tfcaf
    values[20] = dfcaf_dt
    dfcas_dt = (-fcas + fcass) / tfcas
    values[21] = dfcas_dt
    djca_dt = (fcass - jca) / tjca
    values[22] = djca_dt
    dj_dt = (-j + jss) / tj
    values[23] = dj_dt
    dfcafp_dt = (-fcafp + fcass) / tfcafp
    values[24] = dfcafp_dt
    dffp_dt = (-ffp + fss) / tffp
    values[25] = dffp_dt
    dhsp_dt = (-hsp + hssp) / thsp
    values[26] = dhsp_dt
    djp_dt = (-jp + jss) / tjp
    values[27] = djp_dt
    dmL_dt = (-mL + mLss) / tmL
    values[28] = dmL_dt
    dxs2_dt = (-xs2 + xs2ss) / txs2
    values[29] = dxs2_dt
    dZetas_dt = As * dLambda - Zetas * cs
    values[30] = dZetas_dt
    fICaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
    fINaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
    fINap = 1.0 / (1.0 + KmCaMK / CaMKa)
    fItop = 1.0 / (1.0 + KmCaMK / CaMKa)
    fJrelp = 1.0 / (1.0 + KmCaMK / CaMKa)
    fJupp = 1.0 / (1.0 + KmCaMK / CaMKa)
    dnca_dt = anca * k2n - km2n * nca
    values[31] = dnca_dt
    F1 = numpy.exp(C * p_b) - 1
    dCd = C - Cd
    dCaTrpn_dt = ktrpn * (-CaTrpn + ((1000 * cai) / cat50) ** ntrpn * (1 - CaTrpn))
    values[32] = dCaTrpn_dt
    h_lambda_prima = Beta0 * ((lambda_min087 + lambda_min12) - 1.87) + 1
    x2 = b4 * (a2 * a3) + (b4 * (a3 * b1) + (a3 * (a1 * a2) + b4 * (b1 * b2)))
    x1 = a2 * (a1 * b3) + (b3 * (a2 * b4) + (a2 * (a1 * a4) + b3 * (b2 * b4)))
    x3 = b1 * (a3 * a4) + (a4 * (b1 * b2) + (a4 * (a2 * a3) + b1 * (b2 * b3)))
    x4 = a1 * (b2 * b3) + (a1 * (a4 * b2) + (a1 * (a3 * a4) + b2 * (b3 * b4)))
    PCaKp = 0.0003574 * PCap
    PCaNap = 0.00125 * PCap
    dXS_dt = -XS * gammasu + (-XS * ksu + XW * kws)
    values[33] = dXS_dt
    tau_relp = numpy.where((tau_relp_tmp < 0.001), 0.001, tau_relp_tmp)
    k1 = kcaon * (cao * h12)
    k1_i = kcaon * (cao * h12_i)
    k6 = kcaon * (cass * h6)
    k6_i = kcaon * (cai * h6_i)
    h2 = (hna * nass) / ((h1 * kna3))
    h3 = 1.0 / h1
    h2_i = (hna * nai) / ((h1_i * kna3))
    h3_i = 1.0 / h1_i
    h8 = nao / ((h7 * (hna * kna3)))
    h9 = 1.0 / h7
    h8_i = nao / ((h7_i * (hna * kna3)))
    h9_i = 1.0 / h7_i
    diFp_dt = (-iFp + iss) / tiFp
    values[34] = diFp_dt
    diSp_dt = (-iSp + iss) / tiSp
    values[35] = diSp_dt
    IKr = (rkr * (xr * (GKr * (0.4303314829119352 * math.sqrt(ko))))) * (-EK + v)
    ICaL = (d * (PhiCaL * (PCa * (1.0 - fICaLp)))) * (
        f * (1.0 - nca) + nca * (fca * jca)
    ) + (d * (PhiCaL * (PCap * fICaLp))) * (fp * (1.0 - nca) + nca * (fcap * jca))
    INaL = (mL * (GNaL * (-ENa + v))) * (fINaLp * hLp + hL * (1.0 - fINaLp))
    INa = (m**3.0 * ((GNa * scale_drug_INa) * (-ENa + v))) * (
        j * (h * (1.0 - fINap)) + jp * (fINap * hp)
    )
    Ito = ((scale_HF_Gto * (Gto * scale_drug_Ito)) * (-EK + v)) * (
        i * (a * (1.0 - fItop)) + ip * (ap * fItop)
    )
    Jrel = Jrelnp * (1.0 - fJrelp) + Jrelp * fJrelp
    Jup = -Jleak + (Jupnp * (1.0 - fJupp) + scale_HF_Jup * (Jupp * fJupp))
    eta = numpy.where((dCd < 0), etas, etal)
    J_TRPN = dCaTrpn_dt * trpnmax
    h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
    E1 = x1 / (x4 + (x3 + (x1 + x2)))
    E2 = x2 / (x4 + (x3 + (x1 + x2)))
    E3 = x3 / (x4 + (x3 + (x1 + x2)))
    E4 = x4 / (x4 + (x3 + (x1 + x2)))
    ICaK = (d * (PhiCaK * (PCaK * (1.0 - fICaLp)))) * (
        f * (1.0 - nca) + nca * (fca * jca)
    ) + (d * (PhiCaK * (PCaKp * fICaLp))) * (fp * (1.0 - nca) + nca * (fcap * jca))
    ICaNa = (d * (PhiCaNa * (PCaNa * (1.0 - fICaLp)))) * (
        f * (1.0 - nca) + nca * (fca * jca)
    ) + (d * (PhiCaNa * (PCaNap * fICaLp))) * (fp * (1.0 - nca) + nca * (fcap * jca))
    k4pp = h2 * wnaca
    k7 = wna * (h2 * h5)
    k4p_ss = (h3 * wca) / hca
    k4pp_i = h2_i * wnaca
    k7_i = wna * (h2_i * h5_i)
    k4p_i = (h3_i * wca) / hca
    k3pp = h8 * wnaca
    k8 = wna * (h11 * h8)
    k3p_ss = h9 * wca
    k3pp_i = h8_i * wnaca
    k8_i = wna * (h11_i * h8_i)
    k3p_i = h9_i * wca
    Jrel_inf = ((-ICaL) * a_rel) / (((1.5 * scale_HF_Jrel_inf) / cajsr) ** 8.0 + 1.0)
    Jrel_infp = ((-ICaL) * a_relp) / (((1.5 * scale_HF_Jrel_inf) / cajsr) ** 8.0 + 1.0)
    dcajsr_dt = Bcajsr * (-Jrel + Jtr)
    values[36] = dcajsr_dt
    dcansr_dt = Jup - (Jtr * vjsr) / vnsr
    values[37] = dcansr_dt
    Fd = dCd * eta
    dCd_dt = (p_k * (C - Cd)) / eta
    values[38] = dCd_dt
    Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
    JnakNa = 3.0 * (E1 * a3 - E2 * b3)
    JnakK = 2.0 * (-E3 * a1 + E4 * b1)
    dkss_dt = -JdiffK + (Acap * (-ICaK)) / ((F * vss))
    values[39] = dkss_dt
    k4 = k4p_ss + k4pp
    k4_i = k4p_i + k4pp_i
    k3 = k3p_ss + k3pp
    k3_i = k3p_i + k3pp_i
    dJrelnp_dt = (Jrel_inf - Jrelnp) / tau_rel
    values[40] = dJrelnp_dt
    dJrelp_dt = (Jrel_infp - Jrelp) / tau_relp
    values[41] = dJrelp_dt
    Tp = p_a * (F1 + Fd)
    INaK = (Pnak * scale_HF_Pnak) * (JnakK * zk + JnakNa * zna)
    x2_ss = (k1 * k7) * (k4 + k5) + (k4 * k6) * (k1 + k8)
    x2_i = (k1_i * k7_i) * (k4_i + k5_i) + (k4_i * k6_i) * (k1_i + k8_i)
    x1_ss = (k2 * k4) * (k6 + k7) + (k5 * k7) * (k2 + k3)
    x3_ss = (k1 * k3) * (k6 + k7) + (k6 * k8) * (k2 + k3)
    x4_ss = (k2 * k8) * (k4 + k5) + (k3 * k5) * (k1 + k8)
    x1_i = (k2_i * k4_i) * (k6_i + k7_i) + (k5_i * k7_i) * (k2_i + k3_i)
    x3_i = (k1_i * k3_i) * (k6_i + k7_i) + (k6_i * k8_i) * (k2_i + k3_i)
    x4_i = (k2_i * k8_i) * (k4_i + k5_i) + (k3_i * k5_i) * (k1_i + k8_i)
    Ttot = Ta + Tp
    dki_dt = (
        Acap
        * (
            -(
                -2.0 * INaK
                + (
                    Istim
                    + (Isac_P_ns / 3 + (Isac_P_k + (IKb + (IK1 + (IKs + (IKr + Ito))))))
                )
            )
        )
    ) / ((F * vmyo)) + (JdiffK * vss) / vmyo
    values[42] = dki_dt
    E1_ss = x1_ss / (x4_ss + (x3_ss + (x1_ss + x2_ss)))
    E2_ss = x2_ss / (x4_ss + (x3_ss + (x1_ss + x2_ss)))
    E3_ss = x3_ss / (x4_ss + (x3_ss + (x1_ss + x2_ss)))
    E4_ss = x4_ss / (x4_ss + (x3_ss + (x1_ss + x2_ss)))
    E1_i = x1_i / (x4_i + (x3_i + (x1_i + x2_i)))
    E2_i = x2_i / (x4_i + (x3_i + (x1_i + x2_i)))
    E3_i = x3_i / (x4_i + (x3_i + (x1_i + x2_i)))
    E4_i = x4_i / (x4_i + (x3_i + (x1_i + x2_i)))
    JncxCa_ss = -E1_ss * k1 + E2_ss * k2
    JncxNa_ss = -E2_ss * k3pp + (E3_ss * k4pp + 3.0 * (-E1_ss * k8 + E4_ss * k7))
    JncxCa_i = -E1_i * k1_i + E2_i * k2_i
    JncxNa_i = -E2_i * k3pp_i + (E3_i * k4pp_i + 3.0 * (-E1_i * k8_i + E4_i * k7_i))
    INaCa_ss = (allo_ss * ((0.2 * Gncx) * scale_HF_Gncx)) * (
        JncxCa_ss * zca + JncxNa_ss * zna
    )
    INaCa_i = (allo_i * ((0.8 * Gncx) * scale_HF_Gncx)) * (
        JncxCa_i * zca + JncxNa_i * zna
    )
    dcass_dt = Bcass * (
        -Jdiff
        + (
            (Acap * (-(ICaL - 2.0 * INaCa_ss))) / (((2.0 * F) * vss))
            + (Jrel * vjsr) / vss
        )
    )
    values[43] = dcass_dt
    dnass_dt = -JdiffNa + (Acap * (-(ICaNa + 3.0 * INaCa_ss))) / ((F * vss))
    values[44] = dnass_dt
    dcai_dt = Bcai * (
        -J_TRPN
        + (
            (
                (Acap * (-(Isac_P_ns / 3 + (-2.0 * INaCa_i + (ICab + IpCa)))))
                / (((2.0 * F) * vmyo))
                - (Jup * vnsr) / vmyo
            )
            + (Jdiff * vss) / vmyo
        )
    )
    values[45] = dcai_dt
    dnai_dt = (
        Acap
        * (-(Isac_P_ns / 3 + (INab + (3.0 * INaK + (3.0 * INaCa_i + (INa + INaL))))))
    ) / ((F * vmyo)) + (JdiffNa * vss) / vmyo
    values[46] = dnai_dt
    dv_dt = -(
        Isac_P_k
        + (
            Isac_P_ns
            + (
                Istim
                + (
                    ICab
                    + (
                        IpCa
                        + (
                            IKb
                            + (
                                INab
                                + (
                                    INaK
                                    + (
                                        INaCa_ss
                                        + (
                                            INaCa_i
                                            + (
                                                IK1
                                                + (
                                                    IKs
                                                    + (
                                                        IKr
                                                        + (
                                                            ICaK
                                                            + (
                                                                ICaNa
                                                                + (
                                                                    ICaL
                                                                    + (
                                                                        Ito
                                                                        + (INa + INaL)
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    values[47] = dv_dt

    return values


def forward_generalized_rush_larsen(coupling: EMCoupling):
    def forward_generalized_rush_larsen_(states, t, dt, parameters):
        lmbda = coupling.lmbda_ep.vector().get_local()
        dLambda = coupling.dLambda_ep.vector().get_local()
        # Assign states
        hL = states[0]
        a = states[1]
        ap = states[2]
        d = states[3]
        ff = states[4]
        fs = states[5]
        hf = states[6]
        hs = states[7]
        m = states[8]
        xrf = states[9]
        xrs = states[10]
        xs1 = states[11]
        CaMKt = states[12]
        xk1 = states[13]
        Zetaw = states[14]
        XW = states[15]
        TmB = states[16]
        hLp = states[17]
        iF = states[18]
        iS = states[19]
        fcaf = states[20]
        fcas = states[21]
        jca = states[22]
        j = states[23]
        fcafp = states[24]
        ffp = states[25]
        hsp = states[26]
        jp = states[27]
        mL = states[28]
        xs2 = states[29]
        Zetas = states[30]
        nca = states[31]
        CaTrpn = states[32]
        XS = states[33]
        iFp = states[34]
        iSp = states[35]
        cajsr = states[36]
        cansr = states[37]
        Cd = states[38]
        kss = states[39]
        Jrelnp = states[40]
        Jrelp = states[41]
        ki = states[42]
        cass = states[43]
        nass = states[44]
        cai = states[45]
        nai = states[46]
        v = states[47]

        # Assign parameters
        Aff = parameters[0]
        Ahf = parameters[1]
        BSLmax = parameters[2]
        BSRmax = parameters[3]
        Beta0 = parameters[4]
        Beta1 = parameters[5]
        CaMKo = parameters[6]
        Esac_ns = parameters[7]
        F = parameters[8]
        GKb = parameters[9]
        GNa = parameters[10]
        Gncx = parameters[11]
        GpCa = parameters[12]
        Gsac_k = parameters[13]
        Gsac_ns = parameters[14]
        Gto = parameters[15]
        H = parameters[16]
        Khp = parameters[17]
        Kki = parameters[18]
        Kko = parameters[19]
        KmBSL = parameters[20]
        KmBSR = parameters[21]
        KmCaAct = parameters[22]
        KmCaM = parameters[23]
        KmCaMK = parameters[24]
        Kmgatp = parameters[25]
        Kmn = parameters[26]
        Knai0 = parameters[27]
        Knao0 = parameters[28]
        Knap = parameters[29]
        Kxkur = parameters[30]
        L = parameters[31]
        MgADP = parameters[32]
        MgATP = parameters[33]
        PCab = parameters[34]
        PKNa = parameters[35]
        PNab = parameters[36]
        Pnak = parameters[37]
        R = parameters[38]
        T = parameters[39]
        Tot_A = parameters[40]
        Tref = parameters[41]
        Trpn50 = parameters[42]
        aCaMK = parameters[43]
        amp = parameters[44]
        bCaMK = parameters[45]
        bt = parameters[46]
        calib = parameters[47]
        cao = parameters[48]
        cat50_ref = parameters[49]
        celltype = parameters[50]
        cmdnmax = parameters[51]
        csqnmax = parameters[52]
        # dLambda = parameters[53]
        delta = parameters[54]
        delta_epi = parameters[55]
        duration = parameters[56]
        eP = parameters[57]
        emcoupling = parameters[58]
        etal = parameters[59]
        etas = parameters[60]
        gammas = parameters[61]
        gammaw = parameters[62]
        isacs = parameters[63]
        k1m = parameters[64]
        k1p = parameters[65]
        k2m = parameters[66]
        k2n = parameters[67]
        k2p = parameters[68]
        k3m = parameters[69]
        k3p = parameters[70]
        k4m = parameters[71]
        k4p = parameters[72]
        kasymm = parameters[73]
        kcaoff = parameters[74]
        kcaon = parameters[75]
        kmcmdn = parameters[76]
        kmcsqn = parameters[77]
        kmtrpn = parameters[78]
        kna1 = parameters[79]
        kna2 = parameters[80]
        kna3 = parameters[81]
        ko = parameters[82]
        ktrpn = parameters[83]
        ku = parameters[84]
        kuw = parameters[85]
        kws = parameters[86]
        lambda_max = parameters[87]
        # lmbda = parameters[88]
        mode = parameters[89]
        nao = parameters[90]
        ntm = parameters[91]
        ntrpn = parameters[92]
        p_a = parameters[93]
        p_b = parameters[94]
        p_k = parameters[95]
        phi = parameters[96]
        qca = parameters[97]
        qna = parameters[98]
        rad = parameters[99]
        rs = parameters[100]
        rw = parameters[101]
        scale_HF_CaMKa = parameters[102]
        scale_HF_GK1 = parameters[103]
        scale_HF_GNaL = parameters[104]
        scale_HF_Gncx = parameters[105]
        scale_HF_Gto = parameters[106]
        scale_HF_Jleak = parameters[107]
        scale_HF_Jrel_inf = parameters[108]
        scale_HF_Jup = parameters[109]
        scale_HF_Pnak = parameters[110]
        scale_HF_cat50_ref = parameters[111]
        scale_HF_thL = parameters[112]
        scale_ICaL = parameters[113]
        scale_IK1 = parameters[114]
        scale_IKr = parameters[115]
        scale_IKs = parameters[116]
        scale_INaL = parameters[117]
        scale_drug_ICaL = parameters[118]
        scale_drug_ICab = parameters[119]
        scale_drug_IK1 = parameters[120]
        scale_drug_IKb = parameters[121]
        scale_drug_IKr = parameters[122]
        scale_drug_IKs = parameters[123]
        scale_drug_INa = parameters[124]
        scale_drug_INaL = parameters[125]
        scale_drug_INab = parameters[126]
        scale_drug_IpCa = parameters[127]
        scale_drug_Isack = parameters[128]
        scale_drug_Isacns = parameters[129]
        scale_drug_Ito = parameters[130]
        thL = parameters[131]
        tjca = parameters[132]
        trpnmax = parameters[133]
        wca = parameters[134]
        wna = parameters[135]
        wnaca = parameters[136]
        zca = parameters[137]
        zk = parameters[138]

        # Assign expressions
        values = numpy.zeros_like(states)
        zna = 1.0
        Afcaf = 0.3 + 0.6 / (numpy.exp((v - 10.0) / 10.0) + 1.0)
        AiF = 1.0 / (numpy.exp((v - 213.6) / 151.2) + 1.0)
        Axrf = 1.0 / (numpy.exp((v + 54.81) / 38.21) + 1.0)
        ass = 1.0 / (numpy.exp((-(v - 14.34)) / 14.82) + 1.0)
        assp = 1.0 / (numpy.exp((-(v - 24.34)) / 14.82) + 1.0)
        dss = 1.0 / (numpy.exp((-(v + 3.94)) / 4.23) + 1.0)
        dti_develop = 1.354 + 0.0001 / (
            numpy.exp((-(v - 12.23)) / 0.2154) + numpy.exp((v - 167.4) / 15.89)
        )
        dti_recover = 1.0 - 0.5 / (numpy.exp((v + 70.0) / 20.0) + 1.0)
        fss = 1.0 / (numpy.exp((v + 19.58) / 3.696) + 1.0)
        hLss = 1.0 / (numpy.exp((v + 87.61) / 7.488) + 1.0)
        hLssp = 1.0 / (numpy.exp((v + 93.81) / 7.488) + 1.0)
        hss = 1.0 / (numpy.exp((v + 78.5) / 6.22) + 1)
        hssp = 1.0 / (numpy.exp(((v + 78.5) + 6.2) / 6.22) + 1)
        iss = 1.0 / (numpy.exp((v + 43.94) / 5.711) + 1.0)
        mLss = 1.0 / (numpy.exp((-(v + 42.85)) / 5.264) + 1.0)
        mss = 1.0 / (numpy.exp((-((v + 39.57) + 9.4)) / 7.5) + 1.0)
        rkr = (1.0 * (1.0 / (numpy.exp((v + 55.0) / 75.0) + 1.0))) / (
            numpy.exp((v - 10.0) / 30.0) + 1.0
        )
        ta = 1.0515 / (
            1.0 / ((1.2089 * (numpy.exp((-(v - 18.4099)) / 29.3814) + 1.0)))
            + 3.5 / (numpy.exp((v + 100.0) / 29.3814) + 1.0)
        )
        td = 0.6 + 1.0 / (numpy.exp((-0.05) * (v + 6.0)) + numpy.exp(0.09 * (v + 14.0)))
        tfcaf = 7.0 + 1.0 / (
            0.04 * numpy.exp((-(v - 4.0)) / 7.0) + 0.04 * numpy.exp((v - 4.0) / 7.0)
        )
        tfcas = 100.0 + 1.0 / (
            0.00012 * numpy.exp((-v) / 3.0) + 0.00012 * numpy.exp(v / 7.0)
        )
        tff = 7.0 + 1.0 / (
            0.0045 * numpy.exp((-(v + 20.0)) / 10.0)
            + 0.0045 * numpy.exp((v + 20.0) / 10.0)
        )
        tfs = 1000.0 + 1.0 / (
            3.5e-05 * numpy.exp((-(v + 5.0)) / 4.0)
            + 3.5e-05 * numpy.exp((v + 5.0) / 6.0)
        )
        thf = 1.0 / (
            6.149 * numpy.exp((v + 0.5096) / 20.27)
            + 1.432e-05 * numpy.exp((-(v + 1.196)) / 6.285)
        )
        ths = 1.0 / (
            0.009794 * numpy.exp((-(v + 17.95)) / 28.05)
            + 0.3343 * numpy.exp((v + 5.73) / 56.66)
        )
        tj = 2.038 + 1.0 / (
            0.3052 * numpy.exp((v + 0.9941) / 38.45)
            + 0.02136 * numpy.exp((-(v + 100.6)) / 8.281)
        )
        tm = 1.0 / (
            6.765 * numpy.exp((v + 11.64) / 34.77)
            + 8.552 * numpy.exp((-(v + 77.42)) / 5.955)
        )
        txk1 = 122.2 / (
            numpy.exp((-(v + 127.2)) / 20.36) + numpy.exp((v + 236.8) / 69.33)
        )
        txrf = 12.98 + 1.0 / (
            4.123e-05 * numpy.exp((-(v - 47.78)) / 20.38)
            + 0.3652 * numpy.exp((v - 31.66) / 3.869)
        )
        txrs = 1.865 + 1.0 / (
            1.128e-05 * numpy.exp((-(v - 29.74)) / 25.94)
            + 0.06629 * numpy.exp((v - 34.7) / 7.355)
        )
        txs1 = 817.3 + 1.0 / (
            0.0002326 * numpy.exp((v + 48.28) / 17.8)
            + 0.001292 * numpy.exp((-(v + 210.0)) / 230.0)
        )
        txs2 = 1.0 / (
            0.01 * numpy.exp((v - 50.0) / 20.0)
            + 0.0193 * numpy.exp((-(v + 66.54)) / 31.0)
        )
        xkb = 1.0 / (numpy.exp((-(v - 14.48)) / 18.34) + 1.0)
        xrss = 1.0 / (numpy.exp((-(v + 8.337)) / 6.789) + 1.0)
        xs1ss = 1.0 / (numpy.exp((-(v + 11.6)) / 8.932) + 1.0)
        Afs = 1.0 - Aff
        Ageo = L * ((2 * 3.14) * rad) + rad * ((2 * 3.14) * rad)
        vcell = L * (rad * ((3.14 * 1000) * rad))
        Ahs = 1.0 - Ahf
        Aw = (Tot_A * rs) / (rs + rw * (1 - rs))
        Bcai = 1.0 / ((cmdnmax * kmcmdn) / (cai + kmcmdn) ** 2.0 + 1.0)
        Jupnp = (0.004375 * cai) / (cai + 0.00092)
        Jupp = ((0.004375 * 2.75) * cai) / ((cai + 0.00092) - 0.00017)
        KsCa = 1.0 + 0.6 / ((3.8e-05 / cai) ** 1.4 + 1.0)
        Bcajsr = 1.0 / ((csqnmax * kmcsqn) / (cajsr + kmcsqn) ** 2.0 + 1.0)
        Bcass = 1.0 / (
            (BSLmax * KmBSL) / (KmBSL + cass) ** 2.0
            + ((BSRmax * KmBSR) / (KmBSR + cass) ** 2.0 + 1.0)
        )
        Jdiff = (-cai + cass) / 0.2
        CaMKb = (CaMKo * (1.0 - CaMKt)) / (KmCaM / cass + 1.0)
        CaTrpn_max = numpy.where((CaTrpn > 0), CaTrpn, 0)
        rk1 = 1.0 / (numpy.exp((-2.6 * ko + (v + 105.8)) / 9.493) + 1.0)
        xk1ss = 1.0 / (
            numpy.exp((-((2.5538 * ko + v) + 144.59)) / (1.5692 * ko + 3.8115)) + 1.0
        )
        vffrt = (F * (F * v)) / ((R * T))
        vfrt = (F * v) / ((R * T))
        EK = ((R * T) / F) * numpy.log(ko / ki)
        EKs = ((R * T) / F) * numpy.log((PKNa * nao + ko) / (PKNa * nai + ki))
        ENa = ((R * T) / F) * numpy.log(nao / nai)
        GK1 = scale_HF_GK1 * ((0.1908 * scale_IK1) * scale_drug_IK1)
        GKr = (0.046 * scale_IKr) * scale_drug_IKr
        GKs = (0.0034 * scale_IKs) * scale_drug_IKs
        GNaL = scale_HF_GNaL * ((0.0075 * scale_INaL) * scale_drug_INaL)
        km2n = 1.0 * jca
        IpCa = (cai * (GpCa * scale_drug_IpCa)) / (cai + 0.0005)
        lambda_min12 = numpy.where((lmbda < 1.2), lmbda, 1.2)
        Isac_P_k = numpy.where(
            (lmbda < 1.0),
            0.0,
            (Gsac_k * ((lmbda - 1.0) / (lambda_max - 1.0)))
            * (1.0 / (numpy.exp((19.05 - v) / 29.98) + 1.0)),
        )
        Isac_P_ns = numpy.where(
            (lmbda < 1.0),
            0.0,
            (Gsac_ns * ((lmbda - 1.0) / (lambda_max - 1.0))) * (-Esac_ns + v),
        )
        Istim = numpy.where((duration >= t), amp, 0)
        JdiffK = (-ki + kss) / 2.0
        JdiffNa = (-nai + nass) / 2.0
        Jtr = (-cajsr + cansr) / 100.0
        Jleak = ((0.0039375 * cansr) * scale_HF_Jleak) / 15.0
        Knai = Knai0 * numpy.exp((F * (delta * v)) / (((3.0 * R) * T)))
        Knao = Knao0 * numpy.exp((F * (v * (1.0 - delta))) / (((3.0 * R) * T)))
        P = eP / (((H / Khp + 1.0) + nai / Knap) + ki / Kxkur)
        PCa = (0.0001 * scale_ICaL) * scale_drug_ICaL
        XS_max = numpy.where((XS > 0), XS, 0)
        zetas1 = Zetas * numpy.where((Zetas > 0), 1.0, 0.0)
        zetas2 = (-Zetas - 1.0) * numpy.where((Zetas < -1.0), 1.0, 0.0)
        XW_max = numpy.where((XW > 0), XW, 0)
        XU = -XW + (-XS + (1 - TmB))
        a2 = k2p
        a4 = ((MgATP * k4p) / Kmgatp) / (1.0 + MgATP / Kmgatp)
        a_rel = 0.5 * bt
        btp = 1.25 * bt
        tau_rel_tmp = bt / (1.0 + 0.0123 / cajsr)
        allo_i = 1.0 / ((KmCaAct / cai) ** 2.0 + 1.0)
        allo_ss = 1.0 / ((KmCaAct / cass) ** 2.0 + 1.0)
        b1 = MgADP * k1m
        cs = ((kws * phi) * (rw * (1 - rs))) / rs
        ksu = (kws * rw) * (-1 + 1 / rs)
        cw = ((kuw * phi) * ((1 - rs) * (1 - rw))) / ((rw * (1 - rs)))
        kwu = kuw * (-1 + 1 / rw) - kws
        gammawu = gammaw * numpy.abs(Zetaw)
        h10 = (nao / kna1) * (1 + nao / kna2) + (kasymm + 1.0)
        h10_i = (nao / kna1) * (1.0 + nao / kna2) + (kasymm + 1.0)
        h4 = (nass / kna1) * (1 + nass / kna2) + 1.0
        h4_i = (nai / kna1) * (1 + nai / kna2) + 1.0
        hca = numpy.exp((F * (qca * v)) / ((R * T)))
        hna = numpy.exp((F * (qna * v)) / ((R * T)))
        k2 = kcaoff
        k2_i = kcaoff
        k5 = kcaoff
        k5_i = kcaoff
        kb = (Trpn50**ntm * ku) / (-rw * (1 - rs) + (1 - rs))
        thLp = scale_HF_thL * (3.0 * thL)
        tiF = (
            delta_epi
            * (
                1
                / (
                    0.3933 * numpy.exp((-(v + 100.0)) / 100.0)
                    + 0.08004 * numpy.exp((v + 50.0) / 16.59)
                )
            )
            + 4.562
        )
        tiS = (
            delta_epi
            * (
                1
                / (
                    0.001416 * numpy.exp((-(v + 96.52)) / 59.05)
                    + 1.78e-08 * numpy.exp((v + 114.1) / 8.079)
                )
            )
            + 23.62
        )
        Afcas = 1.0 - Afcaf
        AiS = 1.0 - AiF
        Axrs = 1.0 - Axrf
        fcass = fss
        dhL_dt = (-hL + hLss) / ((scale_HF_thL * thL))
        dhL_dt_linearized = -1 / (scale_HF_thL * thL)
        values[0] = (
            dhL_dt * (numpy.exp(dhL_dt_linearized * dt) - 1) / dhL_dt_linearized + hL
        )
        jss = hss
        da_dt = (-a + ass) / ta
        da_dt_linearized = -1 / ta
        values[1] = (
            a + da_dt * (numpy.exp(da_dt_linearized * dt) - 1) / da_dt_linearized
        )
        dap_dt = (-ap + assp) / ta
        dap_dt_linearized = -1 / ta
        values[2] = (
            ap + dap_dt * (numpy.exp(dap_dt_linearized * dt) - 1) / dap_dt_linearized
        )
        dd_dt = (-d + dss) / td
        dd_dt_linearized = -1 / td
        values[3] = (
            d + dd_dt * (numpy.exp(dd_dt_linearized * dt) - 1) / dd_dt_linearized
        )
        tfcafp = 2.5 * tfcaf
        tffp = 2.5 * tff
        dff_dt = (-ff + fss) / tff
        dff_dt_linearized = -1 / tff
        values[4] = (
            dff_dt * (numpy.exp(dff_dt_linearized * dt) - 1) / dff_dt_linearized + ff
        )
        dfs_dt = (-fs + fss) / tfs
        dfs_dt_linearized = -1 / tfs
        values[5] = (
            dfs_dt * (numpy.exp(dfs_dt_linearized * dt) - 1) / dfs_dt_linearized + fs
        )
        dhf_dt = (-hf + hss) / thf
        dhf_dt_linearized = -1 / thf
        values[6] = (
            dhf_dt * (numpy.exp(dhf_dt_linearized * dt) - 1) / dhf_dt_linearized + hf
        )
        thsp = 3.0 * ths
        dhs_dt = (-hs + hss) / ths
        dhs_dt_linearized = -1 / ths
        values[7] = (
            dhs_dt * (numpy.exp(dhs_dt_linearized * dt) - 1) / dhs_dt_linearized + hs
        )
        tjp = 1.46 * tj
        tmL = tm
        dm_dt = (-m + mss) / tm
        dm_dt_linearized = -1 / tm
        values[8] = (
            dm_dt * (numpy.exp(dm_dt_linearized * dt) - 1) / dm_dt_linearized + m
        )
        dxrf_dt = (-xrf + xrss) / txrf
        dxrf_dt_linearized = -1 / txrf
        values[9] = (
            dxrf_dt * (numpy.exp(dt * dxrf_dt_linearized) - 1) / dxrf_dt_linearized
            + xrf
        )
        dxrs_dt = (-xrs + xrss) / txrs
        dxrs_dt_linearized = -1 / txrs
        values[10] = (
            dxrs_dt * (numpy.exp(dt * dxrs_dt_linearized) - 1) / dxrs_dt_linearized
            + xrs
        )
        xs2ss = xs1ss
        dxs1_dt = (-xs1 + xs1ss) / txs1
        dxs1_dt_linearized = -1 / txs1
        values[11] = (
            dxs1_dt * (numpy.exp(dt * dxs1_dt_linearized) - 1) / dxs1_dt_linearized
            + xs1
        )
        f = Aff * ff + Afs * fs
        fp = Aff * ffp + Afs * fs
        Acap = 2 * Ageo
        vjsr = 0.0048 * vcell
        vmyo = 0.68 * vcell
        vnsr = 0.0552 * vcell
        vss = 0.02 * vcell
        h = Ahf * hf + Ahs * hs
        hp = Ahf * hf + Ahs * hsp
        As = Aw
        CaMKa = scale_HF_CaMKa * (CaMKb + CaMKt)
        dCaMKt_dt = -CaMKt * bCaMK + (CaMKb * aCaMK) * (CaMKb + CaMKt)
        dCaMKt_dt_linearized = CaMKb * aCaMK - bCaMK
        values[12] = CaMKt + numpy.where(
            numpy.logical_or(
                (dCaMKt_dt_linearized > 1e-08),
                (dCaMKt_dt_linearized < -1e-08),
            ),
            dCaMKt_dt
            * (numpy.exp(dCaMKt_dt_linearized * dt) - 1)
            / dCaMKt_dt_linearized,
            dCaMKt_dt * dt,
        )
        dxk1_dt = (-xk1 + xk1ss) / txk1
        dxk1_dt_linearized = -1 / txk1
        values[13] = (
            dxk1_dt * (numpy.exp(dt * dxk1_dt_linearized) - 1) / dxk1_dt_linearized
            + xk1
        )
        ICab = (
            (vffrt * (4.0 * (PCab * scale_drug_ICab)))
            * (cai * numpy.exp(2.0 * vfrt) - 0.341 * cao)
        ) / (numpy.exp(2.0 * vfrt) - 1.0)
        INab = ((vffrt * (PNab * scale_drug_INab)) * (nai * numpy.exp(vfrt) - nao)) / (
            numpy.exp(vfrt) - 1.0
        )
        PhiCaK = (
            (1.0 * vffrt) * (-0.75 * ko + (0.75 * kss) * numpy.exp(1.0 * vfrt))
        ) / (numpy.exp(1.0 * vfrt) - 1.0)
        PhiCaL = ((4.0 * vffrt) * (-0.341 * cao + cass * numpy.exp(2.0 * vfrt))) / (
            numpy.exp(2.0 * vfrt) - 1.0
        )
        PhiCaNa = (
            (1.0 * vffrt) * (-0.75 * nao + (0.75 * nass) * numpy.exp(1.0 * vfrt))
        ) / (numpy.exp(1.0 * vfrt) - 1.0)
        IKb = (xkb * (GKb * scale_drug_IKb)) * (-EK + v)
        IK1 = (xk1 * (rk1 * (GK1 * math.sqrt(ko)))) * (-EK + v)
        IKs = (xs2 * (xs1 * (GKs * KsCa))) * (-EKs + v)
        anca = 1.0 / (k2n / km2n + (Kmn / cass + 1.0) ** 4.0)
        C = lambda_min12 - 1
        cat50 = scale_HF_cat50_ref * (Beta1 * (lambda_min12 - 1) + cat50_ref)
        lambda_min087 = numpy.where((lambda_min12 < 0.87), lambda_min12, 0.87)
        a1 = (k1p * (nai / Knai) ** 3.0) / (
            ((1.0 + ki / Kki) ** 2.0 + (1.0 + nai / Knai) ** 3.0) - 1.0
        )
        b4 = (k4m * (ki / Kki) ** 2.0) / (
            ((1.0 + ki / Kki) ** 2.0 + (1.0 + nai / Knai) ** 3.0) - 1.0
        )
        a3 = (k3p * (ko / Kko) ** 2.0) / (
            ((1.0 + ko / Kko) ** 2.0 + (1.0 + nao / Knao) ** 3.0) - 1.0
        )
        b2 = (k2m * (nao / Knao) ** 3.0) / (
            ((1.0 + ko / Kko) ** 2.0 + (1.0 + nao / Knao) ** 3.0) - 1.0
        )
        b3 = (H * (P * k3m)) / (1.0 + MgATP / Kmgatp)
        PCaK = 0.0003574 * PCa
        PCaNa = 0.00125 * PCa
        PCap = 1.1 * PCa
        gammasu = gammas * numpy.where((zetas1 > zetas2), zetas1, zetas2)
        a_relp = 0.5 * btp
        tau_relp_tmp = btp / (1.0 + 0.0123 / cajsr)
        tau_rel = numpy.where((tau_rel_tmp < 0.001), 0.001, tau_rel_tmp)
        dZetaw_dt = Aw * dLambda - Zetaw * cw
        dZetaw_dt_linearized = -cw
        values[14] = Zetaw + numpy.where(
            numpy.logical_or(
                (dZetaw_dt_linearized > 1e-08),
                (dZetaw_dt_linearized < -1e-08),
            ),
            dZetaw_dt
            * (numpy.exp(dZetaw_dt_linearized * dt) - 1)
            / dZetaw_dt_linearized,
            dZetaw_dt * dt,
        )
        dXW_dt = -XW * gammawu + (-XW * kws + (XU * kuw - XW * kwu))
        dXW_dt_linearized = -gammawu - kws - kwu
        values[15] = XW + numpy.where(
            numpy.logical_or((dXW_dt_linearized > 1e-08), (dXW_dt_linearized < -1e-08)),
            dXW_dt * (numpy.exp(dXW_dt_linearized * dt) - 1) / dXW_dt_linearized,
            dXW_dt * dt,
        )
        h11 = (nao * nao) / ((kna2 * (h10 * kna1)))
        h12 = 1.0 / h10
        h11_i = (nao * nao) / ((kna2 * (h10_i * kna1)))
        h12_i = 1.0 / h10_i
        h5 = (nass * nass) / ((kna2 * (h4 * kna1)))
        h6 = 1.0 / h4
        h5_i = (nai * nai) / ((kna2 * (h4_i * kna1)))
        h6_i = 1.0 / h4_i
        h1 = (nass / kna3) * (hna + 1) + 1
        h1_i = (nai / kna3) * (hna + 1) + 1
        h7 = (nao / kna3) * (1.0 + 1.0 / hna) + 1.0
        h7_i = (nao / kna3) * (1.0 + 1.0 / hna) + 1.0
        dTmB_dt = -TmB * CaTrpn ** (ntm / 2) * ku + XU * (
            kb
            * numpy.where(
                (CaTrpn ** (-1 / 2 * ntm) < 100),
                CaTrpn ** (-1 / 2 * ntm),
                100,
            )
        )
        dTmB_dt_linearized = -(CaTrpn ** (ntm / 2)) * ku
        values[16] = TmB + numpy.where(
            numpy.logical_or(
                (dTmB_dt_linearized > 1e-08),
                (dTmB_dt_linearized < -1e-08),
            ),
            dTmB_dt * (numpy.exp(dTmB_dt_linearized * dt) - 1) / dTmB_dt_linearized,
            dTmB_dt * dt,
        )
        dhLp_dt = (-hLp + hLssp) / thLp
        dhLp_dt_linearized = -1 / thLp
        values[17] = (
            dhLp_dt * (numpy.exp(dhLp_dt_linearized * dt) - 1) / dhLp_dt_linearized
            + hLp
        )
        tiFp = tiF * (dti_develop * dti_recover)
        diF_dt = (-iF + iss) / tiF
        diF_dt_linearized = -1 / tiF
        values[18] = (
            diF_dt * (numpy.exp(diF_dt_linearized * dt) - 1) / diF_dt_linearized + iF
        )
        tiSp = tiS * (dti_develop * dti_recover)
        diS_dt = (-iS + iss) / tiS
        diS_dt_linearized = -1 / tiS
        values[19] = (
            diS_dt * (numpy.exp(diS_dt_linearized * dt) - 1) / diS_dt_linearized + iS
        )
        fca = Afcaf * fcaf + Afcas * fcas
        fcap = Afcaf * fcafp + Afcas * fcas
        i = AiF * iF + AiS * iS
        ip = AiF * iFp + AiS * iSp
        xr = Axrf * xrf + Axrs * xrs
        dfcaf_dt = (-fcaf + fcass) / tfcaf
        dfcaf_dt_linearized = -1 / tfcaf
        values[20] = (
            dfcaf_dt * (numpy.exp(dfcaf_dt_linearized * dt) - 1) / dfcaf_dt_linearized
            + fcaf
        )
        dfcas_dt = (-fcas + fcass) / tfcas
        dfcas_dt_linearized = -1 / tfcas
        values[21] = (
            dfcas_dt * (numpy.exp(dfcas_dt_linearized * dt) - 1) / dfcas_dt_linearized
            + fcas
        )
        djca_dt = (fcass - jca) / tjca
        djca_dt_linearized = -1 / tjca
        values[22] = (
            djca_dt * (numpy.exp(djca_dt_linearized * dt) - 1) / djca_dt_linearized
            + jca
        )
        dj_dt = (-j + jss) / tj
        dj_dt_linearized = -1 / tj
        values[23] = (
            dj_dt * (numpy.exp(dj_dt_linearized * dt) - 1) / dj_dt_linearized + j
        )
        dfcafp_dt = (-fcafp + fcass) / tfcafp
        dfcafp_dt_linearized = -1 / tfcafp
        values[24] = (
            dfcafp_dt
            * (numpy.exp(dfcafp_dt_linearized * dt) - 1)
            / dfcafp_dt_linearized
            + fcafp
        )
        dffp_dt = (-ffp + fss) / tffp
        dffp_dt_linearized = -1 / tffp
        values[25] = (
            dffp_dt * (numpy.exp(dffp_dt_linearized * dt) - 1) / dffp_dt_linearized
            + ffp
        )
        dhsp_dt = (-hsp + hssp) / thsp
        dhsp_dt_linearized = -1 / thsp
        values[26] = (
            dhsp_dt * (numpy.exp(dhsp_dt_linearized * dt) - 1) / dhsp_dt_linearized
            + hsp
        )
        djp_dt = (-jp + jss) / tjp
        djp_dt_linearized = -1 / tjp
        values[27] = (
            djp_dt * (numpy.exp(djp_dt_linearized * dt) - 1) / djp_dt_linearized + jp
        )
        dmL_dt = (-mL + mLss) / tmL
        dmL_dt_linearized = -1 / tmL
        values[28] = (
            dmL_dt * (numpy.exp(dmL_dt_linearized * dt) - 1) / dmL_dt_linearized + mL
        )
        dxs2_dt = (-xs2 + xs2ss) / txs2
        dxs2_dt_linearized = -1 / txs2
        values[29] = (
            dxs2_dt * (numpy.exp(dt * dxs2_dt_linearized) - 1) / dxs2_dt_linearized
            + xs2
        )
        dZetas_dt = As * dLambda - Zetas * cs
        dZetas_dt_linearized = -cs
        values[30] = Zetas + numpy.where(
            numpy.logical_or(
                (dZetas_dt_linearized > 1e-08),
                (dZetas_dt_linearized < -1e-08),
            ),
            dZetas_dt
            * (numpy.exp(dZetas_dt_linearized * dt) - 1)
            / dZetas_dt_linearized,
            dZetas_dt * dt,
        )
        fICaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
        fINaLp = 1.0 / (1.0 + KmCaMK / CaMKa)
        fINap = 1.0 / (1.0 + KmCaMK / CaMKa)
        fItop = 1.0 / (1.0 + KmCaMK / CaMKa)
        fJrelp = 1.0 / (1.0 + KmCaMK / CaMKa)
        fJupp = 1.0 / (1.0 + KmCaMK / CaMKa)
        dnca_dt = anca * k2n - km2n * nca
        dnca_dt_linearized = -km2n
        values[31] = nca + numpy.where(
            numpy.logical_or(
                (dnca_dt_linearized > 1e-08),
                (dnca_dt_linearized < -1e-08),
            ),
            dnca_dt * (numpy.exp(dnca_dt_linearized * dt) - 1) / dnca_dt_linearized,
            dnca_dt * dt,
        )
        F1 = numpy.exp(C * p_b) - 1
        dCd = C - Cd
        dCaTrpn_dt = ktrpn * (-CaTrpn + ((1000 * cai) / cat50) ** ntrpn * (1 - CaTrpn))
        dCaTrpn_dt_linearized = ktrpn * (-(((1000 * cai) / cat50) ** ntrpn) - 1)
        values[32] = CaTrpn + numpy.where(
            numpy.logical_or(
                (dCaTrpn_dt_linearized > 1e-08),
                (dCaTrpn_dt_linearized < -1e-08),
            ),
            dCaTrpn_dt
            * (numpy.exp(dCaTrpn_dt_linearized * dt) - 1)
            / dCaTrpn_dt_linearized,
            dCaTrpn_dt * dt,
        )
        h_lambda_prima = Beta0 * ((lambda_min087 + lambda_min12) - 1.87) + 1
        x2 = b4 * (a2 * a3) + (b4 * (a3 * b1) + (a3 * (a1 * a2) + b4 * (b1 * b2)))
        x1 = a2 * (a1 * b3) + (b3 * (a2 * b4) + (a2 * (a1 * a4) + b3 * (b2 * b4)))
        x3 = b1 * (a3 * a4) + (a4 * (b1 * b2) + (a4 * (a2 * a3) + b1 * (b2 * b3)))
        x4 = a1 * (b2 * b3) + (a1 * (a4 * b2) + (a1 * (a3 * a4) + b2 * (b3 * b4)))
        PCaKp = 0.0003574 * PCap
        PCaNap = 0.00125 * PCap
        dXS_dt = -XS * gammasu + (-XS * ksu + XW * kws)
        dXS_dt_linearized = -gammasu - ksu
        values[33] = XS + numpy.where(
            numpy.logical_or((dXS_dt_linearized > 1e-08), (dXS_dt_linearized < -1e-08)),
            dXS_dt * (numpy.exp(dXS_dt_linearized * dt) - 1) / dXS_dt_linearized,
            dXS_dt * dt,
        )
        tau_relp = numpy.where((tau_relp_tmp < 0.001), 0.001, tau_relp_tmp)
        k1 = kcaon * (cao * h12)
        k1_i = kcaon * (cao * h12_i)
        k6 = kcaon * (cass * h6)
        k6_i = kcaon * (cai * h6_i)
        h2 = (hna * nass) / ((h1 * kna3))
        h3 = 1.0 / h1
        h2_i = (hna * nai) / ((h1_i * kna3))
        h3_i = 1.0 / h1_i
        h8 = nao / ((h7 * (hna * kna3)))
        h9 = 1.0 / h7
        h8_i = nao / ((h7_i * (hna * kna3)))
        h9_i = 1.0 / h7_i
        diFp_dt = (-iFp + iss) / tiFp
        diFp_dt_linearized = -1 / tiFp
        values[34] = (
            diFp_dt * (numpy.exp(diFp_dt_linearized * dt) - 1) / diFp_dt_linearized
            + iFp
        )
        diSp_dt = (-iSp + iss) / tiSp
        diSp_dt_linearized = -1 / tiSp
        values[35] = (
            diSp_dt * (numpy.exp(diSp_dt_linearized * dt) - 1) / diSp_dt_linearized
            + iSp
        )
        IKr = (rkr * (xr * (GKr * (0.4303314829119352 * math.sqrt(ko))))) * (-EK + v)
        ICaL = (d * (PhiCaL * (PCa * (1.0 - fICaLp)))) * (
            f * (1.0 - nca) + nca * (fca * jca)
        ) + (d * (PhiCaL * (PCap * fICaLp))) * (fp * (1.0 - nca) + nca * (fcap * jca))
        INaL = (mL * (GNaL * (-ENa + v))) * (fINaLp * hLp + hL * (1.0 - fINaLp))
        INa = (m**3.0 * ((GNa * scale_drug_INa) * (-ENa + v))) * (
            j * (h * (1.0 - fINap)) + jp * (fINap * hp)
        )
        Ito = ((scale_HF_Gto * (Gto * scale_drug_Ito)) * (-EK + v)) * (
            i * (a * (1.0 - fItop)) + ip * (ap * fItop)
        )
        Jrel = Jrelnp * (1.0 - fJrelp) + Jrelp * fJrelp
        Jup = -Jleak + (Jupnp * (1.0 - fJupp) + scale_HF_Jup * (Jupp * fJupp))
        eta = numpy.where((dCd < 0), etas, etal)
        J_TRPN = dCaTrpn_dt * trpnmax
        h_lambda = numpy.where((h_lambda_prima > 0), h_lambda_prima, 0)
        E1 = x1 / (x4 + (x3 + (x1 + x2)))
        E2 = x2 / (x4 + (x3 + (x1 + x2)))
        E3 = x3 / (x4 + (x3 + (x1 + x2)))
        E4 = x4 / (x4 + (x3 + (x1 + x2)))
        ICaK = (d * (PhiCaK * (PCaK * (1.0 - fICaLp)))) * (
            f * (1.0 - nca) + nca * (fca * jca)
        ) + (d * (PhiCaK * (PCaKp * fICaLp))) * (fp * (1.0 - nca) + nca * (fcap * jca))
        ICaNa = (d * (PhiCaNa * (PCaNa * (1.0 - fICaLp)))) * (
            f * (1.0 - nca) + nca * (fca * jca)
        ) + (d * (PhiCaNa * (PCaNap * fICaLp))) * (
            fp * (1.0 - nca) + nca * (fcap * jca)
        )
        k4pp = h2 * wnaca
        k7 = wna * (h2 * h5)
        k4p_ss = (h3 * wca) / hca
        k4pp_i = h2_i * wnaca
        k7_i = wna * (h2_i * h5_i)
        k4p_i = (h3_i * wca) / hca
        k3pp = h8 * wnaca
        k8 = wna * (h11 * h8)
        k3p_ss = h9 * wca
        k3pp_i = h8_i * wnaca
        k8_i = wna * (h11_i * h8_i)
        k3p_i = h9_i * wca
        Jrel_inf = ((-ICaL) * a_rel) / (
            ((1.5 * scale_HF_Jrel_inf) / cajsr) ** 8.0 + 1.0
        )
        Jrel_infp = ((-ICaL) * a_relp) / (
            ((1.5 * scale_HF_Jrel_inf) / cajsr) ** 8.0 + 1.0
        )
        dcajsr_dt = Bcajsr * (-Jrel + Jtr)
        values[36] = cajsr + dcajsr_dt * dt
        dcansr_dt = Jup - (Jtr * vjsr) / vnsr
        values[37] = cansr + dcansr_dt * dt
        Fd = dCd * eta
        dCd_dt = (p_k * (C - Cd)) / eta
        dCd_dt_linearized = -p_k / eta
        values[38] = Cd + numpy.where(
            numpy.logical_or((dCd_dt_linearized > 1e-08), (dCd_dt_linearized < -1e-08)),
            dCd_dt * (numpy.exp(dCd_dt_linearized * dt) - 1) / dCd_dt_linearized,
            dCd_dt * dt,
        )
        Ta = (h_lambda * (Tref / rs)) * (XS * (Zetas + 1) + XW * Zetaw)
        JnakNa = 3.0 * (E1 * a3 - E2 * b3)
        JnakK = 2.0 * (-E3 * a1 + E4 * b1)
        dkss_dt = -JdiffK + (Acap * (-ICaK)) / ((F * vss))
        values[39] = dkss_dt * dt + kss
        k4 = k4p_ss + k4pp
        k4_i = k4p_i + k4pp_i
        k3 = k3p_ss + k3pp
        k3_i = k3p_i + k3pp_i
        dJrelnp_dt = (Jrel_inf - Jrelnp) / tau_rel
        dJrelnp_dt_linearized = -1 / tau_rel
        values[40] = (
            Jrelnp
            + dJrelnp_dt
            * (numpy.exp(dJrelnp_dt_linearized * dt) - 1)
            / dJrelnp_dt_linearized
        )
        dJrelp_dt = (Jrel_infp - Jrelp) / tau_relp
        dJrelp_dt_linearized = -1 / tau_relp
        values[41] = (
            Jrelp
            + dJrelp_dt
            * (numpy.exp(dJrelp_dt_linearized * dt) - 1)
            / dJrelp_dt_linearized
        )
        Tp = p_a * (F1 + Fd)
        INaK = (Pnak * scale_HF_Pnak) * (JnakK * zk + JnakNa * zna)
        x2_ss = (k1 * k7) * (k4 + k5) + (k4 * k6) * (k1 + k8)
        x2_i = (k1_i * k7_i) * (k4_i + k5_i) + (k4_i * k6_i) * (k1_i + k8_i)
        x1_ss = (k2 * k4) * (k6 + k7) + (k5 * k7) * (k2 + k3)
        x3_ss = (k1 * k3) * (k6 + k7) + (k6 * k8) * (k2 + k3)
        x4_ss = (k2 * k8) * (k4 + k5) + (k3 * k5) * (k1 + k8)
        x1_i = (k2_i * k4_i) * (k6_i + k7_i) + (k5_i * k7_i) * (k2_i + k3_i)
        x3_i = (k1_i * k3_i) * (k6_i + k7_i) + (k6_i * k8_i) * (k2_i + k3_i)
        x4_i = (k2_i * k8_i) * (k4_i + k5_i) + (k3_i * k5_i) * (k1_i + k8_i)
        Ttot = Ta + Tp
        dki_dt = (
            Acap
            * (
                -(
                    -2.0 * INaK
                    + (
                        Istim
                        + (
                            Isac_P_ns / 3
                            + (Isac_P_k + (IKb + (IK1 + (IKs + (IKr + Ito)))))
                        )
                    )
                )
            )
        ) / ((F * vmyo)) + (JdiffK * vss) / vmyo
        values[42] = dki_dt * dt + ki
        E1_ss = x1_ss / (x4_ss + (x3_ss + (x1_ss + x2_ss)))
        E2_ss = x2_ss / (x4_ss + (x3_ss + (x1_ss + x2_ss)))
        E3_ss = x3_ss / (x4_ss + (x3_ss + (x1_ss + x2_ss)))
        E4_ss = x4_ss / (x4_ss + (x3_ss + (x1_ss + x2_ss)))
        E1_i = x1_i / (x4_i + (x3_i + (x1_i + x2_i)))
        E2_i = x2_i / (x4_i + (x3_i + (x1_i + x2_i)))
        E3_i = x3_i / (x4_i + (x3_i + (x1_i + x2_i)))
        E4_i = x4_i / (x4_i + (x3_i + (x1_i + x2_i)))
        JncxCa_ss = -E1_ss * k1 + E2_ss * k2
        JncxNa_ss = -E2_ss * k3pp + (E3_ss * k4pp + 3.0 * (-E1_ss * k8 + E4_ss * k7))
        JncxCa_i = -E1_i * k1_i + E2_i * k2_i
        JncxNa_i = -E2_i * k3pp_i + (E3_i * k4pp_i + 3.0 * (-E1_i * k8_i + E4_i * k7_i))
        INaCa_ss = (allo_ss * ((0.2 * Gncx) * scale_HF_Gncx)) * (
            JncxCa_ss * zca + JncxNa_ss * zna
        )
        INaCa_i = (allo_i * ((0.8 * Gncx) * scale_HF_Gncx)) * (
            JncxCa_i * zca + JncxNa_i * zna
        )
        dcass_dt = Bcass * (
            -Jdiff
            + (
                (Acap * (-(ICaL - 2.0 * INaCa_ss))) / (((2.0 * F) * vss))
                + (Jrel * vjsr) / vss
            )
        )
        values[43] = cass + dcass_dt * dt
        dnass_dt = -JdiffNa + (Acap * (-(ICaNa + 3.0 * INaCa_ss))) / ((F * vss))
        values[44] = dnass_dt * dt + nass
        dcai_dt = Bcai * (
            -J_TRPN
            + (
                (
                    (Acap * (-(Isac_P_ns / 3 + (-2.0 * INaCa_i + (ICab + IpCa)))))
                    / (((2.0 * F) * vmyo))
                    - (Jup * vnsr) / vmyo
                )
                + (Jdiff * vss) / vmyo
            )
        )
        values[45] = cai + dcai_dt * dt
        dnai_dt = (
            Acap
            * (
                -(
                    Isac_P_ns / 3
                    + (INab + (3.0 * INaK + (3.0 * INaCa_i + (INa + INaL))))
                )
            )
        ) / ((F * vmyo)) + (JdiffNa * vss) / vmyo
        values[46] = dnai_dt * dt + nai
        dv_dt = -(
            Isac_P_k
            + (
                Isac_P_ns
                + (
                    Istim
                    + (
                        ICab
                        + (
                            IpCa
                            + (
                                IKb
                                + (
                                    INab
                                    + (
                                        INaK
                                        + (
                                            INaCa_ss
                                            + (
                                                INaCa_i
                                                + (
                                                    IK1
                                                    + (
                                                        IKs
                                                        + (
                                                            IKr
                                                            + (
                                                                ICaK
                                                                + (
                                                                    ICaNa
                                                                    + (
                                                                        ICaL
                                                                        + (
                                                                            Ito
                                                                            + (
                                                                                INa
                                                                                + INaL
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        values[47] = dt * dv_dt + v

        return values

    return forward_generalized_rush_larsen_
