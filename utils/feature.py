import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert

EPS = 1e-12

def _to_1d(x):
    x = np.asarray(x, dtype=np.float32)
    return x.flatten()

# ---------- 时域特征 ----------
def time_features(x):
    x = _to_1d(x)
    N = len(x)

    mean_val = np.mean(x)
    std_val  = np.std(x)
    rms_val  = np.sqrt(np.mean(x**2))
    pp_val   = np.ptp(x)        # max - min
    max_val  = np.max(x)
    min_val  = np.min(x)

    skew_val = skew(x)
    kurt_val = kurtosis(x)

    abs_x = np.abs(x)
    mean_abs = np.mean(abs_x) + EPS
    sqrt_abs = np.sqrt(abs_x)
    mean_sqrt_abs = np.mean(sqrt_abs) + EPS

    crest_factor   = max_val / (rms_val + EPS)
    shape_factor   = rms_val / mean_abs
    impulse_factor = max_val / mean_abs
    margin_factor  = max_val / (mean_sqrt_abs**2 + EPS)

    feats = {
        "time_mean": mean_val,
        "time_std": std_val,
        "time_rms": rms_val,
        "time_pp": pp_val,
        "time_max": max_val,
        "time_min": min_val,
        "time_skew": skew_val,
        "time_kurt": kurt_val,
        "time_crest_factor": crest_factor,
        "time_shape_factor": shape_factor,
        "time_impulse_factor": impulse_factor,
        "time_margin_factor": margin_factor,
    }
    return feats

# ---------- 包络时域特征 ----------
def envelope_time_features(x):
    """
    Hilbert 包络，再做一遍时域特征，字段前缀 env_。
    """
    x = _to_1d(x)
    analytic = hilbert(x)
    env = np.abs(analytic)

    tfeats = time_features(env)
    efeats = {}
    for k, v in tfeats.items():
        new_k = "env_" + k[len("time_"):] if k.startswith("time_") else "env_" + k
        efeats[new_k] = v
    return efeats

def freq_features(x, fs=12000.0):
    """
    基于 FFT 的简单频域特征。
    """
    x = _to_1d(x)
    x = x - np.mean(x)
    N = len(x)

    Xf = np.fft.rfft(x)
    mag = np.abs(Xf) + EPS
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # 能量相关
    spec_energy = np.sum(mag**2)

    # 归一化为概率分布
    p = mag / np.sum(mag)

    # 质心 & 带宽
    centroid = np.sum(freqs * p)
    bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * p))

    # 主峰频率 & 幅值
    idx_peak = np.argmax(mag)
    peak_freq = freqs[idx_peak]
    peak_amp  = mag[idx_peak]

    # 谱熵
    spec_entropy = -np.sum(p * np.log(p + EPS))

    feats = {
        "freq_energy": spec_energy,
        "freq_centroid": centroid,
        "freq_bandwidth": bandwidth,
        "freq_peak_freq": peak_freq,
        "freq_peak_amp": peak_amp,
        "freq_entropy": spec_entropy,
    }
    return feats

def envelope_spectrum_features(x, fs=12000.0):
    """
    简单包络谱特征：对包络做 FFT，提取跟 freq_features 类似的一组。
    """
    x = _to_1d(x)
    analytic = hilbert(x)
    env = np.abs(analytic)

    # 直接调用 freq_features 但换个前缀
    ffeats = freq_features(env, fs=fs)
    efeats = {}
    for k, v in ffeats.items():
        efeats["envspec_" + k[5:]] = v  # freq_ 开头的去掉前缀换 envspec_
    return efeats

from scipy.signal import stft

#简化版 后期可以补充
import pywt
# ---------- 时频谱熵 ----------  
def spectral_entropy(x, fs=12000.0, nperseg=256):
    """
    计算时频谱熵，衡量信号的复杂度。
    """
    x = _to_1d(x)
    f, t, Zxx = stft(x, fs=fs, nperseg=nperseg)
    S = np.abs(Zxx)**2  # 能量谱

    # 归一化为概率分布
    p = S / np.sum(S)

    # 计算谱熵
    entropy = -np.sum(p * np.log(p + EPS))

    return {"spectral_entropy": entropy}

# ---------- STFT 时频能量特征 ----------  
def stft_band_energy_features(x, fs=12000.0,
                              nperseg=256,
                              bands=((0,1000),(1000,3000),(3000,6000))):
    """
    STFT 时频能量特征：按预设频带求平均能量。
    bands: 列表，每个元素为 (f_low, f_high)
    """
    x = _to_1d(x)
    f, t, Zxx = stft(x, fs=fs, nperseg=nperseg)
    S = np.abs(Zxx)**2  # (freq_bins, time_frames)

    feats = {}
    for i, (fl, fh) in enumerate(bands):
        mask = (f >= fl) & (f < fh)
        band_energy = S[mask].mean() if np.any(mask) else 0.0
        feats[f"stft_band{i}_mean_energy_{int(fl)}_{int(fh)}Hz"] = band_energy
    return feats
# ---------- 小波变换时频特征 ----------  
def cwt_features(x, fs=12000.0, wavelet="cmor"):
    """
    计算小波变换的时频特征
    x: 输入信号
    fs: 采样频率
    wavelet: 使用的母小波类型
    """
    x = _to_1d(x)
    scales = np.arange(1, 50)  # 选择小波变换的尺度
    coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=1/fs)
    
    # 计算 CWT 的能量特征
    energy = np.sum(np.abs(coeffs)**2, axis=1)  # 每个尺度的能量

    # 计算能量谱熵（可以看做时频图的复杂度）
    entropy = -np.sum((energy / np.sum(energy)) * np.log(energy / np.sum(energy) + EPS))

    # 选取能量最大的尺度作为特征
    max_energy_scale = np.argmax(energy)

    return {
        "cwt_max_energy_scale": max_energy_scale,
        "cwt_energy_entropy": entropy,
        "cwt_mean_energy": np.mean(energy),
        "cwt_peak_energy": np.max(energy),
    }



def extract_all_features_one_extended(x, fs=12000.0,
                                      use_envelope=True,
                                      use_cwt=True,
                                      use_stft=True):
    """
    结合时域、频域、时频特征提取，增加了小波变换（CWT）和时频谱熵。
    """
    feats = {}
    feats.update(time_features(x))  # 时域特征
    feats.update(freq_features(x, fs=fs))  # 频域特征

    if use_envelope:
        feats.update(envelope_time_features(x))  # 包络时域特征
        feats.update(envelope_spectrum_features(x, fs=fs))  # 包络频域特征
    if use_cwt:
        feats.update(cwt_features(x, fs=fs))  # CWT 时频特征
    if use_stft:
        feats.update(stft_band_energy_features(x, fs=fs))  # STFT 时频能量特征

    # 时频谱熵特征
    feats.update(spectral_entropy(x, fs=fs))

    return feats

def extract_all_features_batch_extended(X, fs=12000.0,
                                         use_envelope=True,
                                         use_cwt=True,
                                         use_stft=True,
                                         return_df=True):
    X = np.asarray(X)
    feat_list = []
    for i in range(X.shape[0]):
        f = extract_all_features_one_extended(X[i], fs=fs,
                                              use_envelope=use_envelope,
                                              use_cwt=use_cwt,
                                              use_stft=use_stft)
        feat_list.append(f)

    if return_df:
        import pandas as pd
        return pd.DataFrame(feat_list)
    else:
        return feat_list
