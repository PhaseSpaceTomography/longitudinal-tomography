import numpy as np
from tomo_action.system_handling import SysHandling


class Analyze:

    @staticmethod
    def compare_profiles(o_profile_dir, py_image, f_image,
                         plot, tag='Reconstructed profile'):
        if o_profile_dir[-1] != "/":
            o_profile_dir += "/"

        o_prof_str = SysHandling.find_file_in_dir(o_profile_dir,
                                                  r"^profile....data")
        original_profile = np.genfromtxt(o_profile_dir + o_prof_str[0])

        plot.subplot(223)
        plot.plot(original_profile)
        plot.plot(np.sum(py_image.T, axis=0))
        plot.plot(np.sum(f_image.T, axis=0))
        plot.gca().legend(('original', 'recreated python',
                           'recreated fortran'))
        plot.title(tag)

    @staticmethod
    def compare_phase_space(py_picture, f_picture):
        return np.sum(np.sqrt((f_picture - py_picture)**2))

    @staticmethod
    # Compares differences relative to original profile
    # this difference is calculated in the fortran and python tomo programs
    def show_difference_to_original(f_diff, py_diff, plot):
        print("Fortran difference | Python difference")
        print(np.concatenate((f_diff, py_diff), axis=1))
        diff_py_over_f = py_diff / f_diff
        print(f"Last difference of reconstruction:"
              f"python: {py_diff[-1]}, fortran: {f_diff[-1]}\n"
              f"difference of differences: "
              f"{diff_py_over_f[-1] * 100 - 100}%")

        plot.text(0.0, 0.9, "Differences from original profile")
        plot.text(0.0, 0.75, "  Fortran: {:e}".format(f_diff[0, -1]))
        plot.text(0.0, 0.65, "  Python:  {:e}".format(py_diff[0, -1]))

    @staticmethod
    def show_difference_to_fortran(pf_diff, plot):
        plot.text(0.0, 0.50, f"Difference Fortran - Python:")
        plot.text(0.0, 0.40, "  {:e}".format(pf_diff))

    @staticmethod
    def show_images(py_picture, ftr_picture, plot):
        plot.subplot(221)
        plot.imshow(py_picture.T, cmap='hot', interpolation='nearest', origin='lower')
        plot.title("Python output")
        plot.subplot(222)
        plot.imshow(ftr_picture.T, cmap='hot', interpolation='nearest', origin='lower')
        plot.title("Fortran output")
        return plot
