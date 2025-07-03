import math
## temp!!!
import numpy as np
import torch

"""
    Implements the 1-D phase correlation from the first-order 
    scattering transform coefficients 
    (ACROSS CHANNELS 0 and 1!!  The default implementation 
    does computation over the same channel)

    Parameters
    ----------
    out_U_1 : Tensor
        a torch Tensor of size `(B, J, N)` where `N` is the temporal size
    psi1 : a dictionary of filters (in the Fourier domain)
    phi : dictionary
        a dictionary of filters of scale :math:`2^J` with keys (`j`)
        where :math:`2^j` is the downsampling factor.
        The array `phi[j]` is a real-valued filter.
    k0 : int
        subsampling scale
    ind_start : dictionary of ints, optional
        indices to truncate the signal to recover only the
        parts which correspond to the actual signal after padding and
        downsampling. Defaults to None
    ind_end : dictionary of ints, optional
        See description of ind_start    
    backend : the backend implementation
"""


def phase_correlation_cross_channel(out_U_1, phi, psi1, k0, ind_start, ind_end, backend):
  eps = 1e-14
  filtered_signal = backend.ifft(out_U_1)
  # filtered_signal = filtered_signal / \
  #                   (eps + backend.norm(filtered_signal, axis=-2, keepdims=True))

  # print('filtered_signal' + str(filtered_signal.shape))
  [mag, phase] = backend.to_polar(filtered_signal)
  accelerated_comparisons = []
  e_slice = slice(0, None, 2)
  o_slice = slice(1, None, 2)
  # k = 0;
  for n1 in range(len(psi1)):
    xi1 = psi1[n1]['xi']
    # not necessary, only for test
    # signal1 = filtered_signal[...,n1,:]
    #

    # debug
    # mag1 = mag[...,n1,:]
    mag1 = mag[e_slice, ..., n1, :]
    mag1 += eps
    # debug
    # phase1 = phase[...,n1,:]
    phase1 = phase[e_slice, ..., n1, :]
    # print('Shape of signal1' + str(signal1.shape))
    for n2 in range(len(psi1)):
      xi2 = psi1[n2]['xi']
      if xi2 >= xi1:
        # not necessary, only for test
        # signal2 = filtered_signal[...,n2,:]

        # polar approach
        # debug
        # mag2 = mag[...,n2,:]
        # phase2 = phase[...,n2,:]
        mag2 = mag[o_slice, ..., n2, :]
        phase2 = phase[o_slice, ..., n2, :]
        # print('Shape of signal2' + str(signal2.shape))
        power = xi2 / xi1
        # accelerate the phase
        phase1_power = backend.multiply(phase1, power)
        # phase1_power_cart = backend.to_cartesian(1.0, phase1_power)
        # phase1_cart = backend.to_cartesian(1.0, phase1_power)
        # correlate signal1 and signal2 complex conjugate, so multiply mag and subtract phase
        comparison_mag = backend.multiply(mag1, mag2)
        comparison_phase = phase1_power - phase2
        comparison = backend.to_cartesian(comparison_mag, comparison_phase)

        # cartesian approach
        # cart_mag1 = backend.cast_complex(mag1)
        # cart_phase1 = backend.divide(signal1, cart_mag1)
        # cart_phase1_power = backend.pow(cart_phase1, power)
        # cart_accelerated1 = backend.multiply(cart_mag1, cart_phase1_power)
        # cart_signal2_conj = backend.conj(signal2)
        # cart_comparison = backend.multiply(cart_accelerated1, cart_signal2_conj)
        # backend.rms_diff(backend.modulus(comparison), backend.modulus(cart_comparison))

        accelerated_comparisons.append(comparison)
        # print('k:' + str(k) + ' n1:' + str(n1) + ' n2:' + str(n2) + ' xi1:' + str(xi1) + ' xi2:' + str(xi2) + ' pow: ' + str(power))
        # k += 1

  accelerated_comparisons = backend.concatenate(accelerated_comparisons)
  # print('Shape of accelerated_comparisons' + str(accelerated_comparisons.shape))
  f_accelerated_comparisons = backend.fft(accelerated_comparisons)
  f_smoothed_accelerated_comparisons = backend.cdgmm(f_accelerated_comparisons, phi['levels'][0])
  f_smoothed_accelerated_comparisons_hat = backend.subsample_fourier(f_smoothed_accelerated_comparisons, 2 ** k0)

  smoothed_accelerated_comparisons = backend.irfft(f_smoothed_accelerated_comparisons_hat)
  # print('Shape of smoothed_accelerated_comparisons' + str(smoothed_accelerated_comparisons.shape))
  smoothed_accelerated_comparisons_unpad = backend.unpad(smoothed_accelerated_comparisons, ind_start[k0], ind_end[k0])
  # print('Shape of smoothed_accelerated_comparisons_unpad' + str(smoothed_accelerated_comparisons_unpad.shape))
  smoothed_accelerated_comparisons_unpad = backend.reshape(smoothed_accelerated_comparisons_unpad,
                                                           (-1, accelerated_comparisons.shape[-2],
                                                            smoothed_accelerated_comparisons_unpad.shape[-1]))
  # print('Shape of (reshaped) smoothed_accelerated_comparisons_unpad' + str(smoothed_accelerated_comparisons_unpad.shape))
  return smoothed_accelerated_comparisons_unpad

"""
    Implements the 1-D phase correlation from the first-order 
    scattering transform coefficients

    Parameters
    ----------
    out_U_1 : Tensor
        a torch Tensor of size `(B, J, N)` where `N` is the temporal size
    psi1 : a dictionary of filters (in the Fourier domain)
    phi : dictionary
        a dictionary of filters of scale :math:`2^J` with keys (`j`)
        where :math:`2^j` is the downsampling factor.
        The array `phi[j]` is a real-valued filter.
    k0 : int
        subsampling scale
    ind_start : dictionary of ints, optional
        indices to truncate the signal to recover only the
        parts which correspond to the actual signal after padding and
        downsampling. Defaults to None
    ind_end : dictionary of ints, optional
        See description of ind_start    
    backend : the backend implementation
"""
def phase_correlation(out_U_1, phi, psi1, k0, ind_start, ind_end, backend):
    filtered_signal = backend.ifft(out_U_1)
    #print('filtered_signal' + str(filtered_signal.shape))
    [mag, phase] = backend.to_polar(filtered_signal)
    accelerated_comparisons = []
    #k = 0;
    for n1 in range(len(psi1)):
      xi1 = psi1[n1]['xi']
      # not necessary, only for test
      #signal1 = filtered_signal[...,n1,:]
      #
      mag1 = mag[...,n1,:]
      mag1 += 1e-14
      phase1 = phase[...,n1,:]
      #print('Shape of signal1' + str(signal1.shape))
      for n2 in range(len(psi1)):
        xi2 = psi1[n2]['xi']
        if xi2 >= xi1:
          # not necessary, only for test
          #signal2 = filtered_signal[...,n2,:]
          
          # polar approach
          mag2 = mag[...,n2,:]
          phase2 = phase[...,n2,:]
          #print('Shape of signal2' + str(signal2.shape))
          power = xi2 / xi1
          # accelerate the phase
          phase1_power = backend.multiply(phase1, power)
          #phase1_power_cart = backend.to_cartesian(1.0, phase1_power)
          #phase1_cart = backend.to_cartesian(1.0, phase1_power)
          # correlate signal1 and signal2 complex conjugate, so multiply mag and subtract phase
          comparison_mag = backend.multiply(mag1, mag2)
          comparison_phase = phase1_power - phase2
          comparison = backend.to_cartesian(comparison_mag, comparison_phase)

          # cartesian approach          
          # cart_mag1 = backend.cast_complex(mag1)
          # cart_phase1 = backend.divide(signal1, cart_mag1)
          # cart_phase1_power = backend.pow(cart_phase1, power)
          # cart_accelerated1 = backend.multiply(cart_mag1, cart_phase1_power)
          # cart_signal2_conj = backend.conj(signal2)
          # cart_comparison = backend.multiply(cart_accelerated1, cart_signal2_conj)
          # backend.rms_diff(backend.modulus(comparison), backend.modulus(cart_comparison))
          
          accelerated_comparisons.append(comparison)          
          #print('k:' + str(k) + ' n1:' + str(n1) + ' n2:' + str(n2) + ' xi1:' + str(xi1) + ' xi2:' + str(xi2) + ' pow: ' + str(power))
          #k += 1
        
    accelerated_comparisons = backend.concatenate(accelerated_comparisons)
    #print('Shape of accelerated_comparisons' + str(accelerated_comparisons.shape))
    f_accelerated_comparisons = backend.fft(accelerated_comparisons)
    f_smoothed_accelerated_comparisons = backend.cdgmm(f_accelerated_comparisons, phi['levels'][0])
    f_smoothed_accelerated_comparisons_hat = backend.subsample_fourier(f_smoothed_accelerated_comparisons, 2**k0)

    smoothed_accelerated_comparisons = backend.irfft(f_smoothed_accelerated_comparisons_hat)
    #print('Shape of smoothed_accelerated_comparisons' + str(smoothed_accelerated_comparisons.shape))
    smoothed_accelerated_comparisons_unpad = backend.unpad(smoothed_accelerated_comparisons, ind_start[k0], ind_end[k0])
    #print('Shape of smoothed_accelerated_comparisons_unpad' + str(smoothed_accelerated_comparisons_unpad.shape))
    smoothed_accelerated_comparisons_unpad = backend.reshape(smoothed_accelerated_comparisons_unpad, 
                                                             (-1, accelerated_comparisons.shape[-2], smoothed_accelerated_comparisons_unpad.shape[-1]))
    #print('Shape of (reshaped) smoothed_accelerated_comparisons_unpad' + str(smoothed_accelerated_comparisons_unpad.shape))
    return smoothed_accelerated_comparisons_unpad

def scattering1d(x, pad, unpad, backend, J, T, psi1, psi2, phi, pad_left=0,
        pad_right=0, ind_start=None, ind_end=None, oversampling=0,
        max_order=2, average=True, size_scattering=(0, 0, 0),
        vectorize=False, out_type='array',
        do_phase_correlation=False,
        do_phase_correlation_cross=False):
    """
    Main function implementing the 1-D scattering transform.

    Parameters
    ----------
    x : Tensor
        a torch Tensor of size `(B, 1, N)` where `N` is the temporal size
    psi1 : dictionary
        a dictionary of filters (in the Fourier domain), with keys (`j`, `q`).
        `j` corresponds to the downsampling factor for
        :math:`x \\ast psi1[(j, q)]``, and `q` corresponds to a pitch class
        (chroma).
        * psi1[(j, n)] is itself a dictionary, with keys corresponding to the
        dilation factors: psi1[(j, n)][j2] corresponds to a support of size
        :math:`2^{J_\\text{max} - j_2}`, where :math:`J_\\text{max}` has been
        defined a priori (`J_max = size` of the padding support of the input)
        * psi1[(j, n)] only has real values;
        the tensors are complex so that broadcasting applies
    psi2 : dictionary
        a dictionary of filters, with keys (j2, n2). Same remarks as for psi1
    phi : dictionary
        a dictionary of filters of scale :math:`2^J` with keys (`j`)
        where :math:`2^j` is the downsampling factor.
        The array `phi[j]` is a real-valued filter.
    J : int
        scale of the scattering
    T : int
        temporal support of low-pass filter, controlling amount of imposed
        time-shift invariance and subsampling
    pad_left : int, optional
        how much to pad the signal on the left. Defaults to `0`
    pad_right : int, optional
        how much to pad the signal on the right. Defaults to `0`
    ind_start : dictionary of ints, optional
        indices to truncate the signal to recover only the
        parts which correspond to the actual signal after padding and
        downsampling. Defaults to None
    ind_end : dictionary of ints, optional
        See description of ind_start
    oversampling : int, optional
        how much to oversample the scattering (with respect to :math:`2^J`):
        the higher, the larger the resulting scattering
        tensor along time. Defaults to `0`
    order2 : boolean, optional
        Whether to compute the 2nd order or not. Defaults to `False`.
    average_U1 : boolean, optional
        whether to average the first order vector. Defaults to `True`
    size_scattering : tuple
        Contains the number of channels of the scattering, precomputed for
        speed-up. Defaults to `(0, 0, 0)`.
    vectorize : boolean, optional
        whether to return a dictionary or a tensor. Defaults to False.
    do_phase_correlation : boolean, optional
        whether to compute phase correlation. Defaults to False.

    """
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    

    # S is simply a dictionary if we do not perform the averaging...
    batch_size = x.shape[0]
    kJ = max(J - oversampling, 0)
    temporal_size = ind_end[kJ] - ind_start[kJ]
    out_S_0, out_S_1, out_S_2 = [], [], []
    
    # PAW
    out_U_1 = []

    # pad to a dyadic size and make it complex
    U_0 = pad(x, pad_left=pad_left, pad_right=pad_right)
    # compute the Fourier transform
    U_0_hat = rfft(U_0)

    log2_T = math.floor(math.log2(T))

    # Get S0
    k0 = max(log2_T - oversampling, 0)

    if average:
        S_0_c = cdgmm(U_0_hat, phi['levels'][0])
        S_0_hat = subsample_fourier(S_0_c, 2**k0)
        S_0_r = irfft(S_0_hat)

        S_0 = unpad(S_0_r, ind_start[k0], ind_end[k0])
    else:
        S_0 = x
    out_S_0.append({'coef': S_0,
                    'j': (),
                    'n': ()})

    # First order:
    for n1 in range(len(psi1)):
        # Convolution + downsampling
        j1 = psi1[n1]['j']

        k1 = max(min(j1 - oversampling, log2_T - oversampling), 0)

        assert psi1[n1]['xi'] < 0.5 / (2**k1)
        U_1_c = cdgmm(U_0_hat, psi1[n1]['levels'][0])
       #PAW
        out_U_1.append(U_1_c[..., 0, :])
        
        U_1_hat = subsample_fourier(U_1_c, 2**k1)
        U_1_c = ifft(U_1_hat)
               
        # Take the modulus
        U_1_m = modulus(U_1_c)
        
        if average or max_order > 1:
            U_1_hat = rfft(U_1_m)

        if average:
            # Convolve with phi_J
            k1_J = max(log2_T - k1 - oversampling, 0)
            S_1_c = cdgmm(U_1_hat, phi['levels'][k1])
            S_1_hat = subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = irfft(S_1_hat)

            S_1 = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
        else:
            S_1 = unpad(U_1_m, ind_start[k1], ind_end[k1])

        out_S_1.append({'coef': S_1,
                        'j': (j1,),
                        'n': (n1,)})

        if max_order == 2:
            # 2nd order
            for n2 in range(len(psi2)):
                j2 = psi2[n2]['j']

                if j2 > j1:
                    assert psi2[n2]['xi'] < psi1[n1]['xi']

                    # convolution + downsampling
                    k2 = max(min(j2 - k1 - oversampling,
                                 log2_T - k1 - oversampling), 0)

                    U_2_c = cdgmm(U_1_hat, psi2[n2]['levels'][k1])
                    U_2_hat = subsample_fourier(U_2_c, 2**k2)
                    # take the modulus
                    U_2_c = ifft(U_2_hat)

                    U_2_m = modulus(U_2_c)

                    if average:
                        U_2_hat = rfft(U_2_m)

                        # Convolve with phi_J
                        k2_J = max(log2_T - k2 - k1 - oversampling, 0)

                        S_2_c = cdgmm(U_2_hat, phi['levels'][k1 + k2])
                        S_2_hat = subsample_fourier(S_2_c, 2**k2_J)
                        S_2_r = irfft(S_2_hat)

                        S_2 = unpad(S_2_r, ind_start[k1 + k2 + k2_J], ind_end[k1 + k2 + k2_J])
                    else:
                        S_2 = unpad(U_2_m, ind_start[k1 + k2], ind_end[k1 + k2])

                    out_S_2.append({'coef': S_2,
                                    'j': (j1, j2),
                                    'n': (n1, n2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array' and vectorize:
        out_S = concatenate([x['coef'] for x in out_S])
    elif out_type == 'array' and not vectorize:
        out_S = {x['n']: x['coef'] for x in out_S}
    elif out_type == 'list':
        # NOTE: This overrides the vectorize flag.
        for x in out_S:
            x.pop('n')

    if do_phase_correlation:
      phase_corr = phase_correlation(concatenate(out_U_1), phi, psi1, k0, ind_start, ind_end, backend)
      out_S = [out_S, phase_corr]
    elif do_phase_correlation_cross:
      phase_corr = phase_correlation_cross_channel(concatenate(out_U_1), phi, psi1, k0, ind_start, ind_end, backend)
      phase_corr = [phase_corr, phase_corr]
      phase_corr = concatenate(phase_corr)
      phase_corr = backend.reshape(phase_corr, [-1, phase_corr.shape[-2], phase_corr.shape[-1]])
      out_S = [out_S, phase_corr]
    else:
      out_S = [out_S, out_S]
      pass
    
    return out_S

__all__ = ['scattering1d']
