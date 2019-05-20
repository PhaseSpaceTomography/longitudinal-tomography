import numpy as np
from system_handling import SysHandling


class Analyze:

    @staticmethod
    def compare_profiles(dirr, plot, tag='Reconstructed profile'):
        if dirr[-1] != "/":
            dirr += "/"
        o_prof_str = SysHandling.find_file_in_dir(dirr, r"^profile....data")
        o_prof = np.genfromtxt(dirr + o_prof_str[0])

        py_rec_file = SysHandling.find_file_in_dir(dirr, r"^py_picture")
        py_rec_prof = np.genfromtxt(dirr + py_rec_file[0])
        im_shape = int(np.sqrt(len(py_rec_prof)))
        py_rec_prof = py_rec_prof.reshape((im_shape, im_shape))

        f_rec_file = SysHandling.find_file_in_dir(dirr, r"^image....data")
        f_rec_prof = np.genfromtxt(dirr + f_rec_file[0])
        im_shape = int(np.sqrt(len(f_rec_prof)))
        f_rec_prof = f_rec_prof.reshape((im_shape, im_shape))

        plot.subplot(223)
        plot.plot(o_prof)
        plot.plot(np.sum(py_rec_prof.T, axis=0))
        plot.plot(np.sum(f_rec_prof.T, axis=0))
        plot.gca().legend(('original', 'recreated python',
                           'recreated fortran'))
        plot.title(tag)

    @staticmethod
    def analyze_difference(dirr, plot):
        if dirr[-1] != "/":
            dirr += "/"
        f_diff, py_diff = SysHandling.find_difference_files(dirr)

        plot.subplot(224)
        plot.axis('off')
        plot.text(0.0, 0.9, "Differences in last reconstruction: ")
        plot.text(0.0, 0.75, f"Fortran: {f_diff[-1]}")
        plot.text(0.0, 0.65, f"Python:  {py_diff[-1]}")

        print("Fortran difference | Python difference")
        print(np.concatenate((f_diff, py_diff), axis=1))
        diff_py_over_f = py_diff / f_diff
        print(f"Last difference of reconstruction:"
              f"python: {py_diff[-1]}, fortran: {f_diff[-1]}\n"
              f"difference of differences: "
              f"{diff_py_over_f[-1] * 100 - 100}%")

    @staticmethod
    def show_images(py_picture, ftr_picture, plot):
        plot.subplot(221)
        plot.imshow(py_picture.T, cmap='hot', interpolation='nearest')
        plot.title("Python output")
        plot.subplot(222)
        plot.imshow(ftr_picture.T, cmap='hot', interpolation='nearest')
        plot.title("Fortran output")
        return plot
