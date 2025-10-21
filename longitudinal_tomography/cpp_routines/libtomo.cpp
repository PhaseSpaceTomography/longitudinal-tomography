/**
 * @author Anton Lu (anton.lu@cern.ch)
 * @file libtomo.cpp
 *
 * Pybind11 wrappers for tomography C++ routines
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <algorithm>
#include <omp.h>

#include "docs.h"
#include "libtomo.h"
#include "kick_and_drift.h"
#include "reconstruct.h"
#include "data_treatment.h"

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
using namespace pybind11::literals;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> d_array;
typedef py::array_t<float, py::array::c_style | py::array::forcecast> f_array;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> i_array;


void wrapper_set_num_threads(const int num_threads) {
    try {
        omp_set_num_threads(num_threads);
    } catch (...) {
        throw std::runtime_error("Could not set OMP number of threads.");
    }
}


void wrapper_kick_up(const d_array &input_dphi,
                     const d_array &input_denergy,
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

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);

    kick_up(dphi, denergy, rf1v, rf2v, phi0, phi12, hratio, nr_particles, acc_kick);
}


void wrapper_kick_down(const d_array &input_dphi,
                       const d_array &input_denergy,
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

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);

    kick_down(dphi, denergy, rf1v, rf2v, phi0, phi12, hratio, nr_particles, acc_kick);
}


d_array wrapper_kick(const py::object &machine,
                     const d_array &denergy,
                     const d_array &dphi,
                     const d_array &arr_rfv1,
                     const d_array &arr_rfv2,
                     const int n_particles,
                     const int turn,
                     bool up) {
    d_array arr_phi0 = d_array(machine.attr("phi0"));
    const double phi12 = py::float_(machine.attr("phi12"));
    const double h_ratio = py::float_(machine.attr("h_ratio"));
    d_array arr_deltaE0 = d_array(machine.attr("deltaE0"));

    auto phi0 = arr_phi0.mutable_unchecked<1>();
    auto deltaE0 = arr_deltaE0.mutable_unchecked<1>();
    auto rfv1 = arr_rfv1.unchecked<1>();
    auto rfv2 = arr_rfv2.unchecked<1>();

    if (up)
        wrapper_kick_up(dphi, denergy, rfv1(turn), rfv2(turn),
                        phi0(turn), phi12, h_ratio, n_particles,
                        deltaE0(turn));
    else
        wrapper_kick_down(dphi, denergy, rfv1(turn), rfv2(turn),
                          phi0(turn), phi12, h_ratio, n_particles,
                          deltaE0(turn));

    return denergy;
}


void wrapper_drift_up(const d_array &input_dphi,
                      const d_array &input_denergy,
                      const double drift_coef,
                      const int n_particles
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);

    drift_up(dphi, denergy, drift_coef, n_particles);
}


void wrapper_drift_down(const d_array &input_dphi,
                        const d_array &input_denergy,
                        const double drift_coef,
                        const int nr_particles
) {
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();

    auto *const denergy = static_cast<double *>(denergy_buffer.ptr);
    auto *const dphi = static_cast<double *>(dphi_buffer.ptr);

    drift_down(dphi, denergy, drift_coef, nr_particles);
}


d_array wrapper_drift(
        const d_array &denergy,
        const d_array &dphi,
        const d_array &input_drift_coef,
        const int n_particles,
        const int turn,
        bool up) {
    auto drift_coef = input_drift_coef.unchecked<1>();

    if (up)
        wrapper_drift_up(dphi, denergy, drift_coef(turn),
                         n_particles);
    else
        wrapper_drift_down(dphi, denergy, drift_coef(turn),
                           n_particles);

    return dphi;
}


// wrap C++ function with NumPy array IO
py::tuple wrapper_kick_and_drift_machine(
        const d_array &input_xp,
        const d_array &input_yp,
        const d_array &input_denergy,
        const d_array &input_dphi,
        const d_array &input_rf1v,
        const d_array &input_rf2v,
        const py::object &machine,
        const int rec_prof,
        const int deltaturn,
        const int nturns,
        const int nparts,
        const bool ftn_out,
        const std::optional<const py::object> callback
) {
    d_array input_phi0 = d_array(machine.attr("phi0"));
    d_array input_deltaE0 = d_array(machine.attr("deltaE0"));
    d_array input_drift_coef = d_array(machine.attr("drift_coef"));
    const double phi12 = py::float_(machine.attr("phi12"));
    const double hratio = py::float_(machine.attr("h_ratio"));
    const int dturns = py::int_(machine.attr("dturns"));

    wrapper_kick_and_drift_scalar(input_xp, input_yp, input_denergy, input_dphi, input_rf1v, input_rf2v,
                                  input_phi0, input_deltaE0, input_drift_coef, phi12, hratio, dturns,
                                  rec_prof, deltaturn, nturns, nparts, ftn_out, callback);

    return py::make_tuple(input_xp, input_yp);
}

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
) {
    real_t *ptr_phi12 = new real_t[nturns];
    std::fill_n(ptr_phi12, nturns, phi12);

    py::capsule capsule(ptr_phi12, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });
    real_Tarr arr_phi12({nturns}, ptr_phi12, capsule);

    wrapper_kick_and_drift_array(input_xp, input_yp, input_denergy, input_dphi, input_rf1v, input_rf2v, input_phi0,
                                 input_deltaE0,
                                 input_drift_coef, arr_phi12, hratio, dturns, rec_prof, deltaturn, nturns, nparts, ftn_out,
                                 callback);

    return py::make_tuple(input_xp, input_yp);
}

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
) {
    py::buffer_info xp_buffer = input_xp.request();
    py::buffer_info yp_buffer = input_yp.request();
    py::buffer_info denergy_buffer = input_denergy.request();
    py::buffer_info dphi_buffer = input_dphi.request();
    py::buffer_info rf1v_buffer = input_rf1v.request();
    py::buffer_info rf2v_buffer = input_rf2v.request();

    py::buffer_info phi0_buffer = input_phi0.request();
    py::buffer_info deltaE0_buffer = input_deltaE0.request();
    py::buffer_info phi12_buffer = input_phi12.request();
    py::buffer_info drift_coef_buffer = input_drift_coef.request();

    auto *xp = static_cast<real_t *>(xp_buffer.ptr);
    auto *yp = static_cast<real_t *>(yp_buffer.ptr);

    const int n_profiles = xp_buffer.shape[0];
    auto **const xp_d = new real_t *[n_profiles];
    for (int i = 0; i < n_profiles; i++)
        xp_d[i] = &xp[i * nparts];

    auto **const yp_d = new real_t *[n_profiles];
    for (int i = 0; i < n_profiles; i++)
        yp_d[i] = &yp[i * nparts];

    auto cleanup = [xp_d, yp_d]() {
        delete[] xp_d;
        delete[] yp_d;
    };

    real_t *const denergy = static_cast<real_t *>(denergy_buffer.ptr);
    real_t *const dphi = static_cast<real_t *>(dphi_buffer.ptr);
    const real_t *const rf1v = static_cast<real_t *>(rf1v_buffer.ptr);
    const real_t *const rf2v = static_cast<real_t *>(rf2v_buffer.ptr);
    const real_t *const phi0 = static_cast<real_t *>(phi0_buffer.ptr);
    const real_t *const deltaE0 = static_cast<real_t *>(deltaE0_buffer.ptr);
    const real_t *const phi12 = static_cast<real_t *>(phi12_buffer.ptr);
    const real_t *const drift_coef = static_cast<real_t *>(drift_coef_buffer.ptr);

    std::function<void(int, int)> cb;
    if (callback.has_value()) {
        cb = [&callback](const int progress, const int total) {
            callback.value()(progress, total);
        };
    } else
        cb = [](const int progress, const int total) { (void) progress, (void) total; };

    try {
        kick_and_drift<real_t>(xp_d, yp_d, denergy, dphi, rf1v, rf2v, phi0, deltaE0, drift_coef,
                       phi12, hratio, dturns, rec_prof, deltaturn, nturns, nparts, ftn_out, cb);
    } catch (const std::exception &e) {
        cleanup();
        throw;
    }

    cleanup();

    return py::make_tuple(input_xp, input_yp);
}

d_array wrapper_back_project(
        const d_array &input_weights,
        const i_array &input_flat_points,
        const d_array &input_flat_profiles,
        const int n_particles,
        const int n_profiles
) {
    py::buffer_info buffer_weights = input_weights.request();
    py::buffer_info buffer_flat_points = input_flat_points.request();
    py::buffer_info buffer_flat_profiles = input_flat_profiles.request();

    auto *weights = static_cast<double *>(buffer_weights.ptr);
    auto *flat_points = static_cast<int *>(buffer_flat_points.ptr);

    auto *const flat_profiles = static_cast<double *>(buffer_flat_profiles.ptr);

    back_project(weights, flat_points, flat_profiles, n_particles, n_profiles);

    return input_weights;
}


d_array wrapper_project(
        const d_array &input_flat_rec,
        const i_array &input_flat_points,
        const d_array &input_weights,
        const int n_particles,
        const int n_profiles,
        const int n_bins
) {
    py::buffer_info buffer_flat_rec = input_flat_rec.request();
    py::buffer_info buffer_flat_points = input_flat_points.request();
    py::buffer_info buffer_weights = input_weights.request();

    auto *weights = static_cast<double *>(buffer_weights.ptr);
    auto *flat_points = static_cast<int *>(buffer_flat_points.ptr);
    auto *const flat_rec = static_cast<double *>(buffer_flat_rec.ptr);

    project(flat_rec, flat_points, weights, n_particles, n_profiles);

    buffer_flat_rec.shape = std::vector<py::ssize_t>{n_profiles, n_bins};

    return input_flat_rec;
}

template <typename real_Tarr, typename real_t>
real_Tarr wrapper_count_particles_in_bin(
        const real_Tarr &input_parts,
        const i_array &input_xp,
        const int n_profiles,
        const int n_particles,
        const int n_bins
) {
    py::buffer_info buffer_parts = input_parts.request();
    py::buffer_info buffer_xp = input_xp.request();

    auto *parts = static_cast<real_t *>(buffer_parts.ptr);
    auto *xp = static_cast<int *>(buffer_xp.ptr);

    count_particles_in_bin(parts, xp, n_profiles, n_particles, n_bins);

    return input_parts;
}

template <typename real_Tarr, typename real_t>
real_Tarr wrapper_count_particles_in_bin_multi(
        const real_Tarr &input_parts,
        const i_array &input_xpRound0,
        const i_array &input_centers,
        const int n_profiles,
        const int n_particles,
        const int n_bins,
        const int n_centers
) {
    py::buffer_info buffer_parts = input_parts.request();
    py::buffer_info buffer_xp = input_xpRound0.request();
    py::buffer_info buffer_cents = input_centers.request();

    auto *parts = static_cast<real_t *>(buffer_parts.ptr);
    auto *xp = static_cast<int *>(buffer_xp.ptr);
    auto *cents = static_cast<int *>(buffer_cents.ptr);

    count_particles_in_bin_multi(parts, xp, cents, n_profiles, n_particles, n_bins, n_centers);

    return input_parts;
}

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
) {
    py::buffer_info buffer_xp = input_xp.request();
    py::buffer_info buffer_waterfall = waterfall.request();

    auto *weights = new real_t[n_particles]();
    auto *discr = new real_t[n_iter + 1]();
    auto *flat_profs = static_cast<real_t *>(buffer_waterfall.ptr);
    auto *recreated = new real_t[n_profiles * n_bins]();

    const int *const xp = static_cast<int *>(buffer_xp.ptr);

    std::function<void(int, int)> cb;
    if (callback.has_value()) {
        cb = [&callback](const int progress, const int total) {
            callback.value()(progress, total);
        };
    } else
        cb = [](const int progress, const int total) { (void) progress, (void) total; };

    try {
        reconstruct<real_t>(weights, xp, flat_profs, recreated, discr, n_iter, n_bins, n_particles, n_profiles, verbose, cb);
    } catch (const std::exception &e) {
        delete[] weights;
        delete[] discr;
        delete[] recreated;

        throw;
    }

    py::capsule capsule_weights(weights, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });
    py::capsule capsule_discr(discr, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });
    py::capsule capsule_recreated(recreated, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });

    py::array_t<real_t> arr_weights = py::array_t<real_t>({n_particles}, weights, capsule_weights);
    py::array_t<real_t> arr_discr = py::array_t<real_t>({n_iter + 1}, discr, capsule_discr);
    py::array_t<real_t> arr_recreated = py::array_t<real_t>({n_profiles, n_bins}, recreated, capsule_recreated);

    return py::make_tuple(arr_weights, arr_discr, arr_recreated);
}

template <typename real_Tarr, typename real_t>
py::tuple wrapper_reconstruct_multi(
        const i_array &input_xp,
        const real_Tarr &waterfall,
        const i_array &inp_cutleft,
        const i_array &inp_cutright,
        const i_array &inp_centers,
        const int n_iter,
        const int n_bins,
        const int n_particles,
        const int n_profiles,
        const int n_centers,
        const bool verbose,
        const std::optional<const py::object> callback
) {

    py::buffer_info buffer_xp = input_xp.request();
    py::buffer_info buffer_waterfall = waterfall.request();
    py::buffer_info buffer_cutleft = inp_cutleft.request();
    py::buffer_info buffer_cutright = inp_cutright.request();
    py::buffer_info buffer_centers = inp_centers.request();

    auto *weights = new real_t[n_particles * n_centers]();
    auto *discr = new real_t[n_iter + 1]();
    auto *discr_split = new real_t[n_centers * (n_iter + 1)];
    auto *recreated = new real_t[n_profiles * n_bins]();
    auto *flat_profs = static_cast<real_t *>(buffer_waterfall.ptr);

    const int *const xp = static_cast<int *>(buffer_xp.ptr);
    const int *const cutleft = static_cast<int *>(buffer_cutleft.ptr);
    const int *const cutright = static_cast<int *>(buffer_cutright.ptr);
    const int *const centers = static_cast<int *>(buffer_centers.ptr);

    std::function<void(int, int)> cb;
    if (callback.has_value()) {
        cb = [&callback](const int progress, const int total) {
            callback.value()(progress, total);
        };
    } else
        cb = [](const int progress, const int total) { (void) progress, (void) total; };

    try {
        reconstruct_multi(weights, xp, centers, cutleft, cutright, flat_profs,
                          recreated, discr, discr_split, n_iter, n_bins, n_particles,
                           n_profiles, n_centers, verbose, cb);
    } catch (const std::exception &e) {
        delete[] weights;
        delete[] discr;
        delete[] discr_split;
        delete[] recreated;

        throw;
    }

    py::capsule capsule_weights(weights, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });
    py::capsule capsule_discr(discr, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });
    py::capsule capsule_discr_split(discr_split, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });
    py::capsule capsule_recreated(recreated, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });

    py::array_t<real_t> arr_weights = py::array_t<real_t>({n_particles*n_centers}, weights, capsule_weights);
    py::array_t<real_t> arr_discr = py::array_t<real_t>({n_iter + 1}, discr, capsule_discr);
    py::array_t<real_t> arr_discr_split = py::array_t<real_t>({n_centers * (n_iter + 1)}, discr_split, capsule_discr_split);
    py::array_t<real_t> arr_recreated = py::array_t<real_t>({n_profiles, n_bins}, recreated, capsule_recreated);

    return py::make_tuple(arr_weights, arr_discr, arr_discr_split, arr_recreated);
}

template <typename real_Tarr, typename real_t>
py::array_t<real_t> wrapper_make_phase_space(
        const i_array &input_xp,
        const i_array &input_yp,
        const real_Tarr &input_weight,
        const int n_bins
) {
    py::buffer_info buffer_xp = input_xp.request();
    py::buffer_info buffer_yp = input_yp.request();
    py::buffer_info buffer_weight = input_weight.request();

    const int n_particles = buffer_xp.shape[0];

    const auto *xp = static_cast<int *>(buffer_xp.ptr);
    const auto *yp = static_cast<int *>(buffer_yp.ptr);
    const auto *weights = static_cast<real_t *>(buffer_weight.ptr);

    real_t *phase_space = make_phase_space(xp, yp, weights, n_particles, n_bins);
    py::capsule capsule(phase_space, [](void *p) { delete[] reinterpret_cast<real_t *>(p); });

    return py::array_t<real_t>({n_bins, n_bins}, phase_space, capsule);
}


// wrap as Python module
PYBIND11_MODULE(libtomo, m
) {
m.doc() = "pybind11 tomo plugin";

m.def("set_num_threads", &wrapper_set_num_threads, set_num_threads_docs,
      "num_threads"_a);

m.def("kick", &wrapper_kick, kick_docs,
"machine"_a, "denergy"_a, "dphi"_a,
"rfv1"_a, "rfv2"_a, "npart"_a, "turn"_a, "up"_a = true);

m.def("drift", &wrapper_drift, drift_docs,
"denergy"_a, "dphi"_a, "drift_coef"_a, "npart"_a, "turn"_a,
"up"_a = true);

m.def("kick_up", &wrapper_kick_up, "Tomography kick up",
"dphi"_a, "denergy"_a, "rfv1"_a, "rfv2"_a,
"phi0"_a, "phi12"_a, "h_ratio"_a, "n_particles"_a, "acc_kick"_a);

m.def("kick_down", &wrapper_kick_down, "Tomography kick down",
"dphi"_a, "denergy"_a, "rfv1"_a, "rfv2"_a,
"phi0"_a, "phi12"_a, "h_ratio"_a, "n_particles"_a, "acc_kick"_a);

m.def("drift_up", &wrapper_drift_up, "Tomography drift up",
"dphi"_a, "denergy"_a, "drift_coef"_a, "n_particles"_a);

m.def("drift_down", &wrapper_drift_down, "Tomography drift down",
"dphi"_a, "denergy"_a, "drift_coef"_a, "n_particles"_a);

m.def("kick_and_drift", &wrapper_kick_and_drift_machine, kick_and_drift_docs,
"xp"_a, "yp"_a, "denergy"_a, "dphi"_a, "rfv1"_a, "rfv2"_a, "machine"_a,
"rec_prof"_a, "deltaturn"_a, "nturns"_a, "nparts"_a, "ftn_out"_a = false, "callback"_a = py::none()
);

m.def("kick_and_drift", &wrapper_kick_and_drift_scalar<d_array, double>, kick_and_drift_docs,
"xp"_a, "yp"_a, "denergy"_a, "dphi"_a, "rfv1"_a, "rfv2"_a, "phi0"_a,
"deltaE0"_a, "drift_coef"_a, "phi12"_a, "h_ratio"_a, "dturns"_a,
"rec_prof"_a, "deltaturn"_a, "nturns"_a, "nparts"_a, "ftn_out"_a = false, "callback"_a = py::none()
);

m.def("kick_and_drift", &wrapper_kick_and_drift_scalar<f_array, float>, kick_and_drift_docs,
"xp"_a, "yp"_a, "denergy"_a, "dphi"_a, "rfv1"_a, "rfv2"_a, "phi0"_a,
"deltaE0"_a, "drift_coef"_a, "phi12"_a, "h_ratio"_a, "dturns"_a,
"rec_prof"_a, "deltaturn"_a, "nturns"_a, "nparts"_a, "ftn_out"_a = false, "callback"_a = py::none()
);

m.def("kick_and_drift", wrapper_kick_and_drift_array<d_array, double>, kick_and_drift_docs,
"xp"_a, "yp"_a, "denergy"_a, "dphi"_a, "rfv1"_a, "rfv2"_a, "phi0"_a,
"deltaE0"_a, "drift_coef"_a, "phi12"_a, "h_ratio"_a, "dturns"_a,
"rec_prof"_a, "deltaturn"_a, "nturns"_a, "nparts"_a, "ftn_out"_a = false, "callback"_a = py::none()
);

m.def("kick_and_drift", wrapper_kick_and_drift_array<f_array, float>, kick_and_drift_docs,
"xp"_a, "yp"_a, "denergy"_a, "dphi"_a, "rfv1"_a, "rfv2"_a, "phi0"_a,
"deltaE0"_a, "drift_coef"_a, "phi12"_a, "h_ratio"_a, "dturns"_a,
"rec_prof"_a, "deltaturn"_a, "nturns"_a, "nparts"_a, "ftn_out"_a = false, "callback"_a = py::none()
);

m.def("project", &wrapper_project, project_docs,
"flat_rec"_a, "flat_points"_a, "weights"_a,
"n_particles"_a, "n_profiles"_a, "n_bins"_a);

m.def("back_project", &wrapper_back_project, back_project_docs,
"weights"_a, "flat_points"_a, "flat_profiles"_a,
"n_particles"_a, "n_profiles"_a);

m.def("count_particles_in_bins", &wrapper_count_particles_in_bin<d_array, double>, count_particles_in_bin_docs,
    "input_parts"_a, "input_xp"_a, "n_particles"_a, "n_profiles"_a, "n_bins"_a);

m.def("count_particles_in_bins", &wrapper_count_particles_in_bin<f_array, float>, count_particles_in_bin_docs,
    "input_parts"_a, "input_xp"_a, "n_particles"_a, "n_profiles"_a, "n_bins"_a);

m.def("count_particles_in_bins_multi", &wrapper_count_particles_in_bin_multi<d_array, double>, count_particles_in_bin_docs,
    "input_parts"_a, "input_xp"_a, "input_centers"_a, "n_particles"_a, "n_profiles"_a, "n_bins"_a, "n_centers"_a);

m.def("count_particles_in_bins_multi", &wrapper_count_particles_in_bin_multi<f_array, float>, count_particles_in_bin_docs,
    "input_parts"_a, "input_xp"_a, "input_centers"_a, "n_particles"_a, "n_profiles"_a, "n_bins"_a, "n_centers"_a);

m.def("reconstruct", &wrapper_reconstruct<d_array, double>, reconstruct_docs,
"xp"_a, "waterfall"_a, "n_iter"_a, "n_bins"_a, "n_particles"_a,
"n_profiles"_a, "verbose"_a = false, "callback"_a = py::none()
);

m.def("reconstruct", &wrapper_reconstruct<f_array, float>, reconstruct_docs,
"xp"_a, "waterfall"_a, "n_iter"_a, "n_bins"_a, "n_particles"_a,
"n_profiles"_a, "verbose"_a = false, "callback"_a = py::none()
);

m.def("reconstruct_multi", &wrapper_reconstruct_multi<d_array, double>, reconstruct_docs,
"xp"_a, "waterfall"_a, "inp_cutleft"_a, "inpt_cutright"_a, "inp_centers"_a, "n_iter"_a,
"n_bins"_a, "n_particles"_a, "n_profiles"_a, "n_centers"_a, "verbose"_a = false, "callback"_a = py::none()
);

m.def("reconstruct_multi", &wrapper_reconstruct_multi<f_array, float>, reconstruct_docs,
"xp"_a, "waterfall"_a, "inp_cutleft"_a, "inpt_cutright"_a, "inp_centers"_a, "n_iter"_a,
"n_bins"_a, "n_particles"_a, "n_profiles"_a, "n_centers"_a, "verbose"_a = false, "callback"_a = py::none()
);

m.def("make_phase_space", &wrapper_make_phase_space<d_array, double>, make_phase_space_docs,
"xp"_a, "yp"_a, "weights"_a, "n_bins"_a);

m.def("make_phase_space", &wrapper_make_phase_space<f_array, float>, make_phase_space_docs,
"xp"_a, "yp"_a, "weights"_a, "n_bins"_a);

}
