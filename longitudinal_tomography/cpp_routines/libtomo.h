/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file libtomo.h
 *
 * Pybind11 wrappers for tomography C++ routines
 */

#ifndef TOMOGRAPHYV3_LIBTOMO_H
#define TOMOGRAPHYV3_LIBTOMO_H

#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> d_array;
typedef py::array_t<float, py::array::c_style | py::array::forcecast> f_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> i_array;

template <typename real_Tarr, typename real_t>
py::tuple wrapper_kick_and_drift_scalar(
        const real_Tarr &input_xp,
        const real_Tarr &input_yp,
        const real_Tarr &input_denergy,
        const real_Tarr &input_dphi,
        const real_Tarr &input_rf1v,
        const real_Tarr &input_rf2v,
        const real_Tarr &input_phi0,
        const real_Tarr &input_deltaE0,
        const real_Tarr &input_drift_coef,
        const real_t phi12,
        const real_t hratio,
        const int dturns,
        const int rec_prof,
        const int deltaturn,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
);

template py::tuple wrapper_kick_and_drift_scalar(
        const d_array &input_xp,
        const d_array &input_yp,
        const d_array &input_denergy,
        const d_array &input_dphi,
        const d_array &input_rf1v,
        const d_array &input_rf2v,
        const d_array &input_phi0,
        const d_array &input_deltaE0,
        const d_array &input_drift_coef,
        const double phi12,
        const double hratio,
        const int dturns,
        const int rec_prof,
        const int deltaturn,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
);

template py::tuple wrapper_kick_and_drift_scalar(
        const f_array &input_xp,
        const f_array &input_yp,
        const f_array &input_denergy,
        const f_array &input_dphi,
        const f_array &input_rf1v,
        const f_array &input_rf2v,
        const f_array &input_phi0,
        const f_array &input_deltaE0,
        const f_array &input_drift_coef,
        const float phi12,
        const float hratio,
        const int dturns,
        const int rec_prof,
        const int deltaturn,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
);

template <typename real_Tarr, typename real_t>
py::tuple wrapper_kick_and_drift_array(
        const real_Tarr &input_xp,
        const real_Tarr &input_yp,
        const real_Tarr &input_denergy,
        const real_Tarr &input_dphi,
        const real_Tarr &input_rf1v,
        const real_Tarr &input_rf2v,
        const real_Tarr &input_phi0,
        const real_Tarr &input_deltaE0,
        const real_Tarr &input_drift_coef,
        const real_Tarr &input_phi12,
        const real_t hratio,
        const int dturns,
        const int rec_prof,
        const int deltaturn,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
);

template py::tuple wrapper_kick_and_drift_array(
        const d_array &input_xp,
        const d_array &input_yp,
        const d_array &input_denergy,
        const d_array &input_dphi,
        const d_array &input_rf1v,
        const d_array &input_rf2v,
        const d_array &input_phi0,
        const d_array &input_deltaE0,
        const d_array &input_drift_coef,
        const d_array &phi12,
        const double hratio,
        const int dturns,
        const int rec_prof,
        const int deltaturn,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
);

template py::tuple wrapper_kick_and_drift_array(
        const f_array &input_xp,
        const f_array &input_yp,
        const f_array &input_denergy,
        const f_array &input_dphi,
        const f_array &input_rf1v,
        const f_array &input_rf2v,
        const f_array &input_phi0,
        const f_array &input_deltaE0,
        const f_array &input_drift_coef,
        const f_array &phi12,
        const float hratio,
        const int dturns,
        const int rec_prof,
        const int deltaturn,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
);

template <typename real_Tarr, typename real_t>
py::tuple wrapper_reconstruct(
        const i_array &input_xp,
        const real_Tarr &waterfall,
        const int n_iter,
        const int n_bins,
        const int n_particles,
        const int n_profiles,
        const bool verbose,
        const std::optional<const py::object> callback
);

template <typename real_Tarr, typename real_t>
py::array_t<real_t> wrapper_make_phase_space(
        const i_array &input_xp,
        const i_array &input_yp,
        const real_Tarr &input_weight,
        const int n_bins
);

class libtomo {

};


#endif //TOMOGRAPHYV3_LIBTOMO_H
