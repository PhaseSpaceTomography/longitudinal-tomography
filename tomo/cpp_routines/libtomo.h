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

py::tuple wrapper_kick_and_drift_scalar(
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
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
);

py::tuple wrapper_kick_and_drift_array(
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
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
);

class libtomo {

};


#endif //TOMOGRAPHYV3_LIBTOMO_H
