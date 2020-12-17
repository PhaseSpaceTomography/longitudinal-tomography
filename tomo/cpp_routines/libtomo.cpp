//
// Created by anton on 12/10/20.
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "libtomo.h"
#include "kick_and_drift.h"
#include "reconstruct.h"

// ----------------
// Python interface
// ----------------

namespace py = pybind11;


void wrapper_kick_up(py::array_t<double, py::array::c_style | py::array::forcecast> input_dphi,
                     py::array_t<double, py::array::c_style | py::array::forcecast> input_denergy,
                     const double rf1v,
                     const double rf2v,
                     const double phi0,
                     const double phi12,
                     const double hratio,
                     const int nr_particles,
                     const double acc_kick
) {

    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    double *const denergy = static_cast<double *>(denergy_buffer.ptr);
    double *const dphi = static_cast<double *>(dphi_buffer.ptr);

    kick_up(dphi, denergy, rf1v, rf2v, phi0, phi12, hratio, nr_particles, acc_kick);
}


void wrapper_kick_down(py::array_t<double, py::array::c_style | py::array::forcecast> input_dphi,
                       py::array_t<double, py::array::c_style | py::array::forcecast> input_denergy,
                       const double rf1v,
                       const double rf2v,
                       const double phi0,
                       const double phi12,
                       const double hratio,
                       const int nr_particles,
                       const double acc_kick
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    double *const denergy = static_cast<double *>(denergy_buffer.ptr);
    double *const dphi = static_cast<double *>(dphi_buffer.ptr);

    kick_down(dphi, denergy, rf1v, rf2v, phi0, phi12, hratio, nr_particles, acc_kick);
}


void wrapper_drift_up(py::array_t<double, py::array::c_style | py::array::forcecast> input_dphi,
                      py::array_t<double, py::array::c_style | py::array::forcecast> input_denergy,
                      const double drift_coef,
                      const int nr_particles
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    double *const denergy = static_cast<double *>(denergy_buffer.ptr);
    double *const dphi = static_cast<double *>(dphi_buffer.ptr);

    drift_up(dphi, denergy, drift_coef, nr_particles);
}


void wrapper_drift_down(py::array_t<double, py::array::c_style | py::array::forcecast> input_dphi,
                        py::array_t<double, py::array::c_style | py::array::forcecast> input_denergy,
                        const double drift_coef,
                        const int nr_particles
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    double *const denergy = static_cast<double *>(denergy_buffer.ptr);
    double *const dphi = static_cast<double *>(dphi_buffer.ptr);

    drift_down(dphi, denergy, drift_coef, nr_particles);
}


// wrap C++ function with NumPy array IO
void wrapper_kick_and_drift(
        py::array_t<double, py::array::c_style | py::array::forcecast> input_xp,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_yp,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_denergy,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_dphi,
        const py::array_t<double, py::array::c_style | py::array::forcecast> input_rf1v,
        const py::array_t<double, py::array::c_style | py::array::forcecast> input_rf2v,
        const py::array_t<double, py::array::c_style | py::array::forcecast> input_phi0,
        const py::array_t<double, py::array::c_style | py::array::forcecast> input_deltaE0,
        const py::array_t<double, py::array::c_style | py::array::forcecast> input_omega_rev0,
        const py::array_t<double, py::array::c_style | py::array::forcecast> input_drift_coef,
        const double phi12,
        const double hratio,
        const int dturns,
        const int rec_prof,
        const int nturns,
        const int nparts,
        const bool ftn_out
) {
    py::buffer_info xp_buffer = input_xp.request();
    py::buffer_info yp_buffer = input_yp.request();
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();
    py::buffer_info rf1v_buffer = input_rf1v.request();
    py::buffer_info rf2v_buffer = input_rf2v.request();
    py::buffer_info phi0_buffer = input_phi0.request();
    py::buffer_info deltaE0_buffer = input_deltaE0.request();
    py::buffer_info omega_rev_0_buffer = input_omega_rev0.request();
    py::buffer_info drift_coef_buffer = input_drift_coef.request();

    double *xp = static_cast<double *>(xp_buffer.ptr);
    double *yp = static_cast<double *>(yp_buffer.ptr);

    const int n_profiles = xp_buffer.shape[0];
    double **const xp_d = new double *[n_profiles];
    for (int i = 0; i < n_profiles; i++)
        xp_d[i] = &xp[i * nparts];

    double **const yp_d = new double *[n_profiles];
    for (int i = 0; i < n_profiles; i++)
        yp_d[i] = &yp[i * nparts];

    double *const denergy = static_cast<double *>(denergy_buffer.ptr);
    double *const dphi = static_cast<double *>(dphi_buffer.ptr);
    const double *const rf1v = static_cast<double *>(rf1v_buffer.ptr);
    const double *const rf2v = static_cast<double *>(rf2v_buffer.ptr);
    const double *const phi0 = static_cast<double *>(phi0_buffer.ptr);
    const double *const deltaE0 = static_cast<double *>(deltaE0_buffer.ptr);
    const double *const omega_rev0 = static_cast<double *>(omega_rev_0_buffer.ptr);
    const double *const drift_coef = static_cast<double *>(drift_coef_buffer.ptr);


//    auto xp = input_xp.mutable_unchecked<2>();
//    auto yp = input_yp.mutable_unchecked<2>();
//    auto denergy = input_denergy.mutable_unchecked<1>();
//    auto dphi = input_dphi.mutable_unchecked<1>();
//    auto rf1v = input_rf1v.mutable_unchecked<1>();
//    auto rf2v = input_rf2v.mutable_unchecked<1>();
//    auto phi0 = input_phi0.mutable_unchecked<1>();
//    auto deltaE0 = input_deltaE0.mutable_unchecked<1>();
//    auto omega_rev0 = input_omega_rev0.mutable_unchecked<1>();
//    auto drift_coef = input_drift_coef.mutable_unchecked<1>();

    kick_and_drift(xp_d, yp_d, denergy, dphi, rf1v, rf2v, phi0, deltaE0, omega_rev0, drift_coef,
                   phi12, hratio, dturns, rec_prof, nturns, nparts, ftn_out);

    delete[] yp_d;
    delete[] xp_d;
}

void wrapper_back_project(
        py::array_t<double, py::array::c_style | py::array::forcecast> input_weights,
        py::array_t<int, py::array::c_style | py::array::forcecast> input_flat_points,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_flat_profiles,
        const int n_particles,
        const int n_profiles
) {
    py::buffer_info buffer_weights = input_weights.request();
    py::buffer_info buffer_flat_points = input_flat_points.request();
    py::buffer_info buffer_flat_profiles = input_flat_profiles.request();

    double *weights = static_cast<double *>(buffer_weights.ptr);
    int *flat_points = static_cast<int *>(buffer_flat_points.ptr);

    int **const flat_points_d = new int *[n_particles];
    for (int i = 0; i < n_particles; i++)
        flat_points_d[i] = &flat_points[i * n_profiles];

    double *const flat_profiles = static_cast<double *>(buffer_flat_profiles.ptr);

    back_project(weights, flat_points_d, flat_profiles, n_particles, n_profiles);

    delete[] flat_points_d;
}


void wrapper_project(
        py::array_t<double, py::array::c_style | py::array::forcecast> input_flat_rec,
        py::array_t<int, py::array::c_style | py::array::forcecast> input_flat_points,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_weights,
        const int n_particles,
        const int n_profiles
) {
    py::buffer_info buffer_flat_rec = input_flat_rec.request();
    py::buffer_info buffer_flat_points = input_flat_points.request();
    py::buffer_info buffer_weights = input_weights.request();

    double *weights = static_cast<double *>(buffer_weights.ptr);
    int *flat_points = static_cast<int *>(buffer_flat_points.ptr);
    double *const flat_rec = static_cast<double *>(buffer_flat_rec.ptr);

    int **const flat_points_d = new int *[n_particles];
    for (int i = 0; i < n_particles; i++)
        flat_points_d[i] = &flat_points[i * n_profiles];

    project(flat_rec, flat_points_d, weights, n_particles, n_profiles);

    delete[] flat_points_d;
}


void wrapper_reconstruct(
        py::array_t<double, py::array::c_style | py::array::forcecast> input_weights,
        py::array_t<int, py::array::c_style | py::array::forcecast> input_xp,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_flat_profiles,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_flat_rec,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_discr,
        const int n_iter,
        const int n_bins,
        const int n_particles,
        const int n_profiles,
        const bool verbose
) {
    py::buffer_info buffer_weights = input_weights.request();
    py::buffer_info buffer_xp = input_xp.request();
    py::buffer_info buffer_flat_profiles = input_flat_profiles.request();
    py::buffer_info buffer_flat_rec = input_flat_rec.request();
    py::buffer_info buffer_discr = input_discr.request();

    double *const weights = static_cast<double *>(buffer_weights.ptr);
    const int *const xp = static_cast<int *>(buffer_xp.ptr);
    double *const flat_profiles = static_cast<double *>(buffer_flat_profiles.ptr);
    double *const flat_rec = static_cast<double *>(buffer_flat_rec.ptr);
    double *const discr = static_cast<double *>(buffer_discr.ptr);

    const int **const xp_d = new const int *[n_particles];
    for (int i = 0; i < n_particles; i++)
        xp_d[i] = &xp[i * n_profiles];

    reconstruct(weights, xp_d, flat_profiles, flat_rec, discr, n_iter, n_bins, n_particles, n_profiles, verbose);

    delete[] xp_d;
}


void wrapper_reconstruct_old(
        py::array_t<double, py::array::c_style | py::array::forcecast> input_weights,
        py::array_t<int, py::array::c_style | py::array::forcecast> input_xp,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_flat_profiles,
        py::array_t<double, py::array::c_style | py::array::forcecast> input_discr,
        const int n_iter,
        const int n_bins,
        const int n_particles,
        const int n_profiles,
        const bool verbose
) {
    py::buffer_info buffer_weights = input_weights.request();
    py::buffer_info buffer_xp = input_xp.request();
    py::buffer_info buffer_flat_profiles = input_flat_profiles.request();
    py::buffer_info buffer_discr = input_discr.request();

    double *const weights = static_cast<double *>(buffer_weights.ptr);
    const int *const xp = static_cast<int *>(buffer_xp.ptr);
    double *const flat_profiles = static_cast<double *>(buffer_flat_profiles.ptr);
    double *const discr = static_cast<double *>(buffer_discr.ptr);

    const int **const xp_d = new const int *[n_particles];
    for (int i = 0; i < n_particles; i++)
        xp_d[i] = &xp[i * n_profiles];

    old_reconstruct(weights, xp_d, flat_profiles, discr, n_iter, n_bins, n_particles, n_profiles, verbose);

    delete[] xp_d;
}



// wrap as Python module
PYBIND11_MODULE(libtomo, m) {
    m.doc() = "pybind11 tomo plugin";
    m.def("kick_up", &wrapper_kick_up, "Tomography kick up");
    m.def("kick_down", &wrapper_kick_down, "Tomography kick down");
    m.def("drift_up", &wrapper_drift_up, "Tomography drift up");
    m.def("drift_down", &wrapper_drift_down, "Tomography drift down");
    m.def("kick_and_drift", &wrapper_kick_and_drift, "Tomography tracking routine");
    m.def("project", &wrapper_project, "Tomography project");
    m.def("back_project", &wrapper_back_project, "Tomography back project");
    m.def("reconstruct", &wrapper_reconstruct, "Tomography reconstruct");
    m.def("reconstruct_old", &wrapper_reconstruct_old, "Tomography old reconstruct");
}
