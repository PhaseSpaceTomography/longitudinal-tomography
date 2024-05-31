# coding: utf-8
'''
tomographics.m translated in py
Work in progress
A. Lasheen

TODO: include double RF
TODO: documentation
TODO: include plotting routine matching the mathematica output (draft already exists)
TODO: docstrings
TODO: unit tests
'''
from __future__ import annotations

# General imports
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import TYPE_CHECKING

# Local imports
from .. import exceptions as exc

if TYPE_CHECKING:
    from typing import Iterable, Tuple
    from ..tracking.machine_base import MachineABC

    FloatArr = np.ndarray[float]
    IntArray = np.ndarray[int]


def foot_tangent_fit_dq(x: Iterable[float], y: Iterable[float],
                        t_rf: float, apply_filter: bool=True)\
                                                        -> Tuple[float, float]:
    """
    Compute the x-intercepts of the foot tangent fit

    Parameters
    ----------
    x : Iterable[float]
        The x-coordinates of the profile.
    y : Iterable[float]
        The y-coordinates of the profile.
    t_rf : float
        The RF period.
    apply_filter : bool, optional
        DESCRIPTION. The default is True.

    Raises
    ------
    InvalidProfileError
        If the profile is too long for a valid fit, or the resulting fit
        goes past an RF period, an InvalidProfileError is raised.

    Returns
    -------
    Tuple[float, float]
        The x coordinates of the lower and upper intercepts.
    """
    if apply_filter:
        y = scipy.signal.savgol_filter(y, 5, 4)

    # Threshold at 15% of the peak
    threshold = 0.15 * np.max(y)
    indices_above = np.where(y >= threshold)[0]

    # Find the indices corresponding to 15%
    left_index = indices_above[0]
    right_index = indices_above[-1]

    if (left_index-3 < 0) or (right_index+3 >= len(x)):
        raise exc.InvalidProfileError("The input profile is too long for a "
                                      +"foot tangent fit ")

    # Polynomial fits of order 1 (straight lines) to find the two
    # tangent lines.  For each fit, 6 points of the y are used, 3 to the
    # left of the 15% point and 3 to the right of the 15% point.
    coefficient_1 = np.polyfit(x[left_index-3:left_index+3],
                               y[left_index-3:left_index+3], 1)
    coefficient_2 = np.polyfit(x[right_index-2:right_index+4],
                               y[right_index-2:right_index+4], 1)

    # Find the intersections of the two tangents with the baseline, here
    # supposed at 0.
    x_min = - coefficient_1[1] / coefficient_1[0]
    x_max = - coefficient_2[1] / coefficient_2[0]

    # Obvious checks, but just to be sure...
    if (0 <= (x_max - x_min) < t_rf):
        return [x_min, x_max]
    else:
        raise exc.InvalidProfileError("The computed x-intercepts go beyond"
                                      +" one RF period")


def foot_tangent_fit(y: Iterable[float], dx: float = 1) -> Tuple[float, float]:
    """
    Compute the foot tangent fit

    Parameters
    ----------
    y : Iterable[float]
        The bunch profile to be fitted.
    dx : float, optional
        The bin width of the profile. The default is 1.

    Returns
    -------
    Tuple[float, float]
        The positions of the lower and upper points of the fit in
        units of dx.
    """

    threshold = 0.15 * np.max(y)
    max_index = np.where(y == np.max(y))[0][0]
    indices_below = np.where(y < threshold)[0]

    left_index = indices_below[indices_below < max_index][-1]
    right_index = indices_below[indices_below > max_index][0]

    y_left = y[left_index - 1:left_index + 3]
    y_right = y[right_index - 2:right_index + 2]

    foot_left = left_index - 5*np.sum(y_left)/(-6*y_left[0] - 2*y_left[1]
                                               +2*y_left[2] + 6*y_left[3])
    foot_right = right_index - 6*np.sum(y_right)/(-6*y_right[0] - 2*y_right[1]
                                                  +2*y_right[2] + 6*y_right[3])

    return foot_left * dx, foot_right * dx


# TODO: What's this for?
def foot_tangent_fit_density(y: Iterable[float], x: Iterable[float],
                             level: float = 0.15) -> float:

    if not (0 < level < 1):
        raise ValueError("Level must be between 0 and 1")

    y_zero = y-np.min(y)

    threshold = level * np.max(y_zero)
    max_index = np.where(y_zero == np.max(y_zero))[0][0]
    indices_below = np.where(y_zero < threshold)[0]

    right_index = indices_below[indices_below > max_index][0]

    y_right = y_zero[right_index - 2:right_index + 2]

    foot_right = right_index - 6*np.sum(y_right)/(-6*y_right[0] - 2*y_right[1]
                                                  +2*y_right[2] + 6*y_right[3])

    foot_density = np.interp(foot_right, np.arange(len(x)), x)

    return foot_density


def _urf_length_at_level(phi_array: Iterable[float], urf: Iterable[float],
                         phi_0: float, urf_level: float)\
                                                 -> Tuple[float, float, float]:

    urf_left = urf[phi_array < phi_0] - urf_level
    urf_right = urf[phi_array > phi_0] - urf_level

    phi_left = np.interp(0, urf_left, phi_array[phi_array < phi_0])
    phi_right = np.interp(0, urf_right[np.argsort(urf_right)],
                          phi_array[phi_array > phi_0][np.argsort(urf_right)])
    phi_tot = phi_right - phi_left

    return phi_tot, phi_right, phi_left


def _residue_urf_length(urf_level: float,
                        phi_target: float, phi_array: float,
                        urf: float, phi_0: float) -> float:

    phi_tot = _urf_length_at_level(phi_array, urf, phi_0, urf_level)[0]
    residue = (phi_tot - phi_target)**2.

    return residue


def matched_area_calc(tomomachine: MachineABC, bunch_length: float,
                      idx_frame: int=None) -> float:
    """
    Compute the matched area for a given bunch length at the specified
    profile number.

    Parameters
    ----------
    tomomachine : MachineABC
        The machine object used for the reconstruction.
    bunch_length : float
        #TODO: Confirm length unit
        The bunch length in seconds.
    idx_frame : int, optional
        The profile number to compute the emittance at.
        The default is None.
        If None, the machine reference frame is used.

    Returns
    -------
    float
        The computed matched area in eVs.
    """

    if idx_frame is None:
        idx_frame = tomomachine.machine_ref_frame

    turn = int(idx_frame * tomomachine.dturns)

    phi_0 = tomomachine.phi0[turn]
    energy = tomomachine.e0[turn]
    eta0 = tomomachine.eta0[turn]
    omega_rev0 = tomomachine.omega_rev0[turn]

    gamma0 = energy / tomomachine.e_rest
    bunch_length_phase = bunch_length * omega_rev0 * tomomachine.h_num

    phi_array = phi_0 + np.linspace(-np.pi, np.pi, 1000)
    urf = (tomomachine.vrf1 * (np.cos(phi_0) - np.cos(phi_array))
           + (phi_0 - phi_array) * tomomachine.vrf1 * np.sin(phi_0))

    urf_level_opti = minimize(_residue_urf_length,
                              (np.max(urf) - np.min(urf))/2 + np.min(urf),
                              args=(bunch_length_phase, phi_array, urf, phi_0),
                              method='Powell')['x']

    phi_tot, phi_right, phi_left = _urf_length_at_level(phi_array, urf, phi_0,
                                                        urf_level_opti)
    phi_array_matched = np.linspace(phi_left, phi_right, 1000)

    dE_height_array = np.abs(np.sqrt(tomomachine.q*(gamma0 - 1/gamma0)
                                     * tomomachine.e_rest/(np.pi
                                                           * tomomachine.h_num
                                                           * eta0)
                                     * (tomomachine.vrf1
                                        * (np.cos(phi_array_matched)
                                           - np.cos(phi_left))
                                        + (phi_array_matched - phi_left)
                                        * tomomachine.vrf1 * np.sin(phi_0))
                                     )
                             )

    dE_height_array[np.isnan(dE_height_array)] = 0

    matched_area = (2 / (tomomachine.h_num*omega_rev0)
                    * np.trapz(dE_height_array,
                               dx=phi_array_matched[1] - phi_array_matched[0]))

    return matched_area


def _cumulative_density_calc(tomo_image: Iterable[float], dt: float,
                             dE: float) -> Tuple[np.ndarray]:

    cumulative_density = np.cumsum(-np.sort(-tomo_image.flatten()))
    cumulative_density_x_array = np.arange(len(cumulative_density)) * dt * dE

    return cumulative_density, cumulative_density_x_array


def emittance_density_calc(tomo_image: Iterable[float], dt: float,
                           dE: float, density_target: float=0.9) -> float:
    """
    Compute the emittance that contains a specified fraction of the beam
    density.

    Parameters
    ----------
    tomo_image : Iterable[float]
        The 2-array defining the phase space distribution.
    # TODO: Confirm units
    dt : float
        The time in seconds of one bin.
    dE : float
        The energy in eV of one bin.
    density_target : float, optional
        The fraction of beam density to compute the emittance for.
        The default is 0.9.

    Raises
    ------
    ValueError
        If density_target is not between 0 and 1, a ValueError is raised.

    Returns
    -------
    float
        The computed emittance in eVs.
    """

    if not (0 < density_target < 1):
        raise ValueError("density_target must be between 0 and 1")

    cumulative_density, cumulative_density_x_array = _cumulative_density_calc(
                                                            tomo_image, dt, dE)

    emittance = np.interp(density_target, cumulative_density,
                          cumulative_density_x_array)

    return emittance


def rms_params(tomo_image: Iterable[float], dt: float, dE: float)\
                                   -> Tuple[float, float, float, float, float]:
    """
    Compute the RMS emittance, mean dt, RMS dt, mean dE and RMS dE

    Parameters
    ----------
    tomo_image : Iterable[float]
        The 2-array containing the phase space distribution.
    # TODO: Confirm units
    dt : float
        The time in seconds of one bin.
    dE : float
        The energy in eV of one bin.

    Returns
    -------
    Tuple[float, float, float, float, float]
        RMS emittance, mean dt, RMS dt, mean dE and RMS dE.
    """

    y_matrix_tomo1, x_matrix_tomo1 = np.meshgrid(np.arange(tomo_image.shape[0]),
                                                 np.arange(tomo_image.shape[1]))
    xbar = np.sum(tomo_image * x_matrix_tomo1)
    xms = np.sum(tomo_image * x_matrix_tomo1**2.)
    ybar = np.sum(tomo_image * y_matrix_tomo1)
    yms = np.sum(tomo_image * y_matrix_tomo1**2.)
    xybar = np.sum(tomo_image * x_matrix_tomo1 * y_matrix_tomo1)

    rmsemittance = np.pi*dt*dE*np.sqrt((xms - xbar**2.)
                                       * (yms - ybar**2.)
                                       - (xybar - xbar*ybar)**2.)

    mean_dt = xbar * dt
    sigma_dt = np.sqrt(xms - xbar**2.) * dt

    mean_dE = (ybar - tomo_image.shape[0]/2) * dE
    sigma_dE = np.sqrt(yms - ybar**2.) * dE

    return rmsemittance, mean_dt, sigma_dt, mean_dE, sigma_dE


def density_vs_emittance(tomomachine: MachineABC, tomo_image: Iterable[float],
                         time_array: Iterable[float], dE_array: Iterable[float],
                         n_points_amplitude: int=100, idx_frame: int=None)\
                                   -> Tuple[float, float, float, float, float]:
    """
    TODO: Good description?

    Parameters
    ----------
    tomomachine : MachineABC
        The machine object used for the reconstruction.
    tomo_image : Iterable[float]
        The 2-array defining the phase space distribution.
    time_array : Iterable[float]
        The time axis of the phase space.
    dE_array : Iterable[float]
        The energy axis of the phase space.
    n_points_amplitude : int, optional
        TODO: DESCRIPTION. The default is 100.
    idx_frame : int, optional
        The profile number to compute at.
        The default is None.
        If None, the machine reference frame is used.

    Returns
    -------
    Tuple[float, float, float, float, float]
        TODO: DESCRIPTION.
    """

    if idx_frame is None:
        idx_frame = tomomachine.machine_ref_frame

    turn = int(idx_frame * tomomachine.dturns)

    phi_0 = tomomachine.phi0[turn]
    energy = tomomachine.e0[turn]
    eta0 = tomomachine.eta0[turn]
    h_num = tomomachine.h_num
    omega_rev0 = tomomachine.omega_rev0[turn]
    gamma0 = energy / tomomachine.e_rest

    phi_array = phi_0 + np.linspace(-np.pi, np.pi, 1000)
# TODO: Make urf a function to avoid code duplication
    urf = (tomomachine.vrf1 * (np.cos(phi_0) - np.cos(phi_array))
           + (phi_0 - phi_array) * tomomachine.vrf1 * np.sin(phi_0))
    sync_time = tomomachine.synch_part_x * (time_array[1] - time_array[0])
    phi_array_matched = (time_array-sync_time) * (h_num*omega_rev0) + phi_0

    amplitude_array = np.linspace(0, np.min(urf), n_points_amplitude)
    summed_density_final = np.zeros(len(amplitude_array))
    local_density_final = np.zeros(len(amplitude_array))
    local_density_min_final = np.zeros(len(amplitude_array))
    local_density_max_final = np.zeros(len(amplitude_array))
    emittance_density = np.zeros(len(amplitude_array))

    for idx_amplitude, amplitude in enumerate(amplitude_array):

        phi_left = _urf_length_at_level(phi_array, urf, phi_0, amplitude)[-1]

# TODO: Make dE_height_array function to avoid code duplication
        dE_height_array = np.abs(np.sqrt(tomomachine.q * (gamma0 - 1/gamma0)
                                         * tomomachine.e_rest
                                         / (np.pi*tomomachine.h_num*eta0)
                                         * (tomomachine.vrf1
                                            * (np.cos(phi_array_matched)
                                               -np.cos(phi_left))
                                            + (phi_array_matched - phi_left)
                                            * tomomachine.vrf1*np.sin(phi_0))
                                         )
                                 )

        dE_height_array[np.isnan(dE_height_array)] = 0

        dE_resolution = dE_array[1] - dE_array[0]
        summed_density = 0
        n_points_local = 0
        local_density = 0
        local_density_min = np.inf
        local_density_max = 0

        for idx_time in range(len(time_array)):
            if dE_height_array[idx_time] == 0:
                continue

            good_idx = np.where((dE_array > -dE_height_array[idx_time])
                                *(dE_array < dE_height_array[idx_time]))[0]

            summed_density += np.trapz(tomo_image[idx_time, good_idx])

            cross_dE_top = np.interp(dE_height_array[idx_time],
                                     dE_array[good_idx[-1] : good_idx[-1]+2],
                                     tomo_image[idx_time,
                                                good_idx[-1] : good_idx[-1]+2])

            summed_density += np.trapz([tomo_image[idx_time, good_idx[-1]],
                                        cross_dE_top],
                                       dx=((dE_height_array[idx_time]
                                            -dE_array[good_idx[-1]])
                                           /dE_resolution))

            cross_dE_bottom = np.interp(-dE_height_array[idx_time],
                                        dE_array[good_idx[0]-1 : good_idx[0]+1],
                                        tomo_image[idx_time, good_idx[0]-1
                                                             : good_idx[0]+1])

            summed_density += np.trapz([cross_dE_bottom,
                                        tomo_image[idx_time, good_idx[0]]],
                                       dx=((dE_array[good_idx[0]]
                                            -(-dE_height_array[idx_time]))
                                            /dE_resolution))

            local_density += (cross_dE_top + cross_dE_bottom) / 2
            local_density_min = np.min([local_density_min, cross_dE_top,
                                        cross_dE_bottom])
            local_density_max = np.max([local_density_max, cross_dE_top,
                                        cross_dE_bottom])

            n_points_local += 1

        if n_points_local != 0:
            summed_density_final[idx_amplitude] = (summed_density
                                                   / np.sum(tomo_image))
            local_density_final[idx_amplitude] = local_density / n_points_local
            local_density_min_final[idx_amplitude] = local_density_min
            local_density_max_final[idx_amplitude] = local_density_max
            emittance_density[idx_amplitude] = (2 / (tomomachine.h_num
                                                     *omega_rev0)
                                                *np.trapz(dE_height_array,
                                                      dx=phi_array_matched[1]
                                                        -phi_array_matched[0]))

    if local_density_final[0] == 0:
        slope_extrapolate = ((local_density_final[2]-local_density_final[1])
                             / (emittance_density[2] - emittance_density[1]))
        origin_extrapolate = (local_density_final[1]
                              - slope_extrapolate
                              * emittance_density[1])
        local_density_final[0] = origin_extrapolate
        local_density_min_final[0] = local_density_final[0]
        local_density_max_final[0] = local_density_final[0]

    return (emittance_density, local_density_final, local_density_min_final,
            local_density_max_final, summed_density_final)


def tomo_weight_clipping(tomomachine: MachineABC, time_array: Iterable[float],
                         dE_array: Iterable[float], map_tomo_x: Iterable[float],
                         map_tomo_y: Iterable[float],
                         weight_tomo: Iterable[float], emittance_target: float,
                         n_points_amplitude: int=100, idx_frame: int=None)\
                                        -> Tuple[FloatArr, FloatArr, IntArray]:
    """


    Parameters
    ----------
    tomomachine : MachineABC
        The machine object used for the reconstruction.
    time_array : Iterable[float]
        The time axis of the phase space.
    dE_array : Iterable[float]
        The energy axis of the phase space.
    map_tomo_x : Iterable[float]
        TODO: DESCRIPTION.
    map_tomo_y : Iterable[float]
        TODO: DESCRIPTION.
    weight_tomo : Iterable[float]
        TODO: DESCRIPTION.
    emittance_target : float
        TODO: DESCRIPTION.
    n_points_amplitude : int, optional
        TODO: DESCRIPTION. The default is 100.
    idx_frame : int, optional
        The profile number to compute at.
        The default is None.
        If None, the machine reference frame is used.

    Returns
    -------
    Tuple[FloatArr, FloatArr, IntArray]
        TODO: DESCRIPTION.
    """

    if idx_frame is None:
        idx_frame = tomomachine.machine_ref_frame

    turn = int(idx_frame * tomomachine.dturns)

    new_weight_tomo = np.array(weight_tomo)

    phi_0 = tomomachine.phi0[turn]
    energy = tomomachine.e0[turn]
    eta0 = tomomachine.eta0[turn]
    h_num = tomomachine.h_num
    omega_rev0 = tomomachine.omega_rev0[turn]
    gamma0 = energy / tomomachine.e_rest

    phi_array = phi_0 + np.linspace(-np.pi, np.pi, 1000)
    urf = (tomomachine.vrf1 * (np.cos(phi_0)-np.cos(phi_array))
           + (phi_0-phi_array) * tomomachine.vrf1 * np.sin(phi_0))
    sync_time = tomomachine.synch_part_x * (time_array[1] - time_array[0])
    phi_array_matched = (time_array-sync_time) * (h_num*omega_rev0) + phi_0

    amplitude_array = np.linspace(0, np.min(urf), n_points_amplitude)

    dt_res = time_array[1] - time_array[0]
    dE_res = dE_array[1] - dE_array[0]

    for amplitude in amplitude_array:

        phi_left = _urf_length_at_level(phi_array, urf, phi_0, amplitude)[-1]

        dE_height_array = np.abs(np.sqrt(tomomachine.q * (gamma0 - 1/gamma0)
                                         * tomomachine.e_rest
                                         / (np.pi*tomomachine.h_num*eta0)
                                         * (tomomachine.vrf1
                                            * (np.cos(phi_array_matched)
                                               - np.cos(phi_left))
                                            + (phi_array_matched-phi_left)
                                            * tomomachine.vrf1 * np.sin(phi_0))
                                         )
                                 )

        dE_height_array[np.isnan(dE_height_array)] = 0

        emittance_density = (2 / (tomomachine.h_num*omega_rev0)
                             * np.trapz(dE_height_array,
                                        dx=phi_array_matched[1]
                                           - phi_array_matched[0]))

        if emittance_density >= emittance_target:
            break

    idx_minmax = np.array([], dtype=int)

    for idx_time in range(len(time_array) - 1):

        idx_out = np.where(
            ((map_tomo_x[:, idx_frame] >= ((time_array[idx_time]
                                            - time_array[0]) / dt_res))
             *(map_tomo_x[:, idx_frame] <= ((time_array[idx_time+1]
                                             - time_array[0]) / dt_res)))
            *((map_tomo_y[:, idx_frame] >= ((dE_height_array[idx_time]
                                             - dE_array[0]) / dE_res))
              | (map_tomo_y[:, idx_frame] <= ((-dE_height_array[idx_time]
                                               - dE_array[0]) / dE_res))))
        new_weight_tomo[idx_out] = 0

        idx_in = np.where(
            ((map_tomo_x[:, idx_frame] >= ((time_array[idx_time]
                                            - time_array[0]) / dt_res))
             * (map_tomo_x[:, idx_frame] <= ((time_array[idx_time + 1]
                                              - time_array[0]) / dt_res)))
            * ((map_tomo_y[:, idx_frame] < ((dE_height_array[idx_time]
                                             - dE_array[0]) / dE_res))
               * (map_tomo_y[:, idx_frame] > ((-dE_height_array[idx_time]
                                               - dE_array[0]) / dE_res))))[0]

        if len(idx_in) > 1:
            idx_max = idx_in[map_tomo_y[idx_in, idx_frame]
                             == np.max(map_tomo_y[idx_in, idx_frame])][0]
            idx_min = idx_in[map_tomo_y[idx_in, idx_frame]
                             == np.min(map_tomo_y[idx_in, idx_frame])][0]

            idx_minmax = np.append(idx_minmax, idx_min)
            idx_minmax = np.append(idx_minmax, idx_max)

        elif len(idx_in) == 1:
            idx_minmax.append(np.where(map_tomo_y
                                       == np.max(map_tomo_y[idx_in,
                                                            idx_frame]))[0][0])

    new_weight_tomo = (new_weight_tomo
                       / np.sum(new_weight_tomo)
                       * np.sum(weight_tomo))

    return new_weight_tomo, dE_height_array, idx_minmax
