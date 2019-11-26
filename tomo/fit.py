import numpy as np
from scipy import signal, optimize
from physics import phase_low, dphase_low 


def fit_xat0(profiles):
    ref_idx = profiles.machine.beam_ref_frame
    ref_prof = profiles.waterfall[ref_idx] 
    ref_turn = ref_idx * profiles.machine.dturns

    tfoot_up, tfoot_low = _calc_tangentfeet(ref_prof)
    bunch_duration = (tfoot_up - tfoot_low) * profiles.machine.dtbin
    
    bunch_phaselength = (profiles.machine.h_num * bunch_duration
                         * profiles.machine.omega_rev0[ref_turn])

    x0 = profiles.machine.phi0[ref_turn] - bunch_phaselength / 2.0
    phil = optimize.newton(
            func=phase_low, x0=x0, fprime=dphase_low, tol=0.0001, maxiter=100,
            args=(profiles.machine, bunch_phaselength, ref_turn))
    
    fitted_xat0 = (tfoot_low + (profiles.machine.phi0[ref_turn] - phil)
                   / (profiles.machine.h_num
                      * profiles.machine.omega_rev0[ref_turn]
                      * profiles.machine.dtbin))

    return fitted_xat0, tfoot_low, tfoot_up


# Find foot tangents of profile. Needed to estimate bunch duration
# when performing a fit to find xat0
def _calc_tangentfeet(ref_prof):       
    nbins = len(ref_prof)
    index_array = np.arange(nbins) + 0.5

    tanbin_up, tanbin_low = _calc_tangentbins(ref_prof, nbins)

    [bl, al] = np.polyfit(index_array[tanbin_low - 2: tanbin_low + 2],
                          ref_prof[tanbin_low - 2: tanbin_low + 2], deg=1)

    [bu, au] = np.polyfit(index_array[tanbin_up - 1: tanbin_up + 3],
                          ref_prof[tanbin_up - 1: tanbin_up + 3], deg=1)

    tanfoot_low = -1 * al / bl
    tanfoot_up = -1 * au / bu

    return tanfoot_up, tanfoot_low


# return index of last bins to the left and right of max valued bin,
# with value over the threshold.
def _calc_tangentbins(ref_profile, nbins, threshold_coeff=0.15):
    threshold = threshold_coeff * np.max(ref_profile)
    maxbin = np.argmax(ref_profile)
    for ibin in range(maxbin, 0, -1):
        if ref_profile[ibin] < threshold:
            tangent_bin_low = ibin + 1
            break
    for ibin in range(maxbin, nbins):
        if ref_profile[ibin] < threshold:
            tangent_bin_up = ibin - 1
            break

    return tangent_bin_up, tangent_bin_low