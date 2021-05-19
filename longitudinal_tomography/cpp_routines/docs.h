/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file docs.h
 *
 * Python docstrings for pybind11 wrappers. Can be viewed on the Python side
 * using help(libtomo) as with any other python module.
 */

#ifndef TOMO_DOCS_H
#define TOMO_DOCS_H

const char* kick_docs = R"pbdoc(
    Wrapper for C++ kick function.

    Particle kick for **one** machine turn.

    Used in the :mod:`tomo.tracking.tracking` module.

    Parameters
    ----------
    machine: Machine
        Object holding machine parameters.
    denergy: ndarray
        1D array holding the energy difference relative to the synchronous
        particle for each particle at a turn.
    dphi: ndarray
        1D array holding the phase difference relative to the synchronous
        particle for each particle at a turn.
    rfv1: ndarray
        1D array holding the radio frequency voltage at RF station 1 for
        each turn, multiplied with the charge state of the particles.
    rfv2: ndarray
        1D array holding the radio frequency voltage at RF station 2 for
        each turn, multiplied with the charge state of the particles.
    npart: int
        The number of tracked particles.
    turn: int
        The current machine turn.
    up: boolean, optional, default=True
        Direction of tracking. Up=True tracks towards last machine turn,
        up=False tracks toward first machine turn.

    Returns
    -------
    denergy: ndarray
        1D array containing the new energy of each particle after voltage kick.
)pbdoc";

const char * kick_up_docs = R"pbdoc(

)pbdoc";

const char * kick_down_docs = R"pbdoc(

)pbdoc";

const char * drift_docs = R"pbdoc(
    Wrapper for C++ drift function.

    Particle drift for **one** machine turn

    Used in the :mod:`~tomo.tracking.tracking` module.

    Parameters
    ----------
    denergy: ndarray
        1D array holding the energy difference relative to the synchronous
        particle for each particle at a turn.
    dphi: ndarray
        1D array holding the phase difference relative to the synchronous
        particle for each particle at a turn.
    drift_coef: ndarray
        1D array of drift coefficient at each machine turn.
    npart: int
        The number of tracked particles.
    turn: int
        The current machine turn.
    up: boolean, optional, default=True
        Direction of tracking. Up=True tracks towards last machine turn,
        up=False tracks toward first machine turn.

    Returns
    -------
    dphi: ndarray
        1D array containing the new phase for each particle
        after drifting for a machine turn.
)pbdoc";

const char * drift_up_docs = R"pbdoc(

)pbdoc";

const char * drift_down_docs = R"pbdoc(

)pbdoc";

const char * kick_and_drift_docs = R"pbdoc(
    Wrapper for full kick and drift algorithm written in C++.

    Tracks all particles from the time frame to be recreated,
    trough all machine turns.

    Used in the :mod:`tomo.tracking.tracking` module.

    Parameters
    ----------
    xp: ndarray
        2D array large enough to hold the phase of each particle
        at every time frame. Shape: (nprofiles, nparts)
    yp: ndarray
        2D array large enough to hold the energy of each particle
        at every time frame. Shape: (nprofiles, nparts)
    denergy: ndarray
        1D array holding the energy difference relative to the synchronous
        particle for each particle the initial turn.
    dphi: ndarray
        1D array holding the phase difference relative to the synchronous
        particle for each particle the initial turn.
    rfv1: ndarray
        Array holding the radio frequency voltage at RF station 1 for each
        turn, multiplied with the charge state of the particles.
    rfv2: ndarray
        Array holding the radio frequency voltage at RF station 2 for each
        turn, multiplied with the charge state of the particles.
    rec_prof: int
        Index of profile to be reconstructed.
    nturns: int
        Total number of machine turns.
    nparts: int
        The number of particles.
    args: tuple
        Arguments can be provided via the args if a machine object is not to
        be used. In this case, the args should be:

        - phi0
        - deltaE0
        - omega_rev0
        - drift_coef
        - phi12
        - h_ratio
        - dturns

        The args will not be used if a Machine object is provided.

    machine: Machine, optional, default=False
        Object containing machine parameters.
ftn_out: boolean, optional, default=False
        Flag to enable printing of status of tracking to stdout.
        The format will be similar to the Fortran version.
        Note that the **information regarding lost particles
        are not valid**.

    Returns
    -------
    xp: ndarray
        2D array holding every particles coordinates in phase [rad]
        at every time frame. Shape: (nprofiles, nparts)
    yp: ndarray
        2D array holding every particles coordinates in energy [eV]
        at every time frame. Shape: (nprofiles, nparts)
)pbdoc";

const char * project_docs = R"pbdoc(
    Wrapper projection routine written in C++.
    Used in the :mod:`~tomo.tomography.tomography` module.

    Parameters
    ----------
    recreated: ndarray
        2D array with the shape of the waterfall to be recreated,
        initiated as zero. Shape: (nprofiles, nbins)
    flat_points: ndarray
        2D array containing particle coordinates as integers, pointing
        at flattened versions of the waterfall. Shape: (nparts, nprofiles)
    weights: ndarray
        1D array containing the weight of each particle.
    nparts: int
        Number of tracked particles.
    nprofs: int
        Number of profiles.
    nbins: int
        Number of bins in profiles.

    Returns
    -------
    recreated: ndarray
        2D array containing the projected profiles as waterfall.
        Shape: (nprofiles, nbins)
)pbdoc";

const char * back_project_docs = R"pbdoc(
    Wrapper for back projection routine written in C++.
    Used in the :mod:`~tomo.tomography.tomography` module.

    Parameters
    ----------
    weights: ndarray
        1D array containing the weight of each particle.
    flat_points: ndarray
        2D array containing particle coordinates as integers, pointing
        at flattened versions of the waterfall. Shape: (nparts, nprofiles)
    flat_profiles: ndarray
        1D array containing a flattened waterfall.
    nparts: int
        Number of tracked particles.
    nprofs: int
        Number of profiles.

    Returns
    -------
    weights: ndarray
        1D array containing the **new weight** of each particle.
)pbdoc";

const char * reconstruct_docs = R"pbdoc(
    Wrapper for full reconstruction in C++.
    Used in the :mod:`~tomo.tomography.tomography` module.

    Parameters
    ----------
    xp: ndarray
        2D array containing the coordinate of each particle
        at each time frame. Coordinates should given be as integers of
        the phase space coordinate system. shape: (nparts, nprofiles).
    waterfall: ndarray
        2D array containing measured profiles as waterfall.
        Shape: (nprofiles, nbins).
    niter: int
        Number of iterations in the reconstruction process.
    nbins: int
        Number of bins in a profile.
    npart: int
        Number of tracked particles.
    nprof: int
        Number of profiles.
    verbose: boolean
        Flag to indicate that the tomography routine should broadcast its
        status to stdout. The output is identical to the output
        from the Fortran version.

    Returns
    -------
    weights: ndarray
        1D array containing the weight of each particle.
    discr: ndarray
        1D array containing discrepancy at each
        iteration of the reconstruction.
    recreated: ndarray
        2D array containing the projected profiles as waterfall.
        Shape: (nprofiles, nbins)
)pbdoc";

const char * reconstruct_old_docs = R"pbdoc(
    Wrapper for full reconstruction in C++.
    Used in the :mod:`~tomo.tomography.tomography` module.

    Well tested, but do not return reconstructed waterfall.
    Kept for reference.

    Parameters
    ----------
    weights: ndarray
        1D array containing the weight of each particle initiated to zeroes.
    xp: ndarray
        2D array containing the coordinate of each particle
        at each time frame. Coordinates should given be as integers of
        the phase space coordinate system. shape: (nparts, nprofiles).
    flat_profiles: ndarray
        1D array containing flattened waterfall.
    discr: ndarray
        Array large enough to hold the discrepancy for each iteration of the
        reconstruction process + 1.
    niter: int
        Number of iterations in the reconstruction process.
    nbins: int
        Number of bins in a profile.
    npart: int
        Number of tracked particles.
    nprof: int
        Number of profiles.
    verbose: boolean
        Flag to indicate that the tomography routine should broadcast its
        status to stdout. The output is identical to the output
        from the Fortran version.

    Returns
    -------
    weights: ndarray
        1D array containing the final weight of each particle.
    discr: ndarray
        1D array containing discrepancy at each
        iteration of the reconstruction.
)pbdoc";

const char * make_phase_space_docs = R"pbdoc(
    # Creates a [nbins, nbins] array populated with the weights of each test
    # particle
)pbdoc";


#endif //TOMO_DOCS_H
