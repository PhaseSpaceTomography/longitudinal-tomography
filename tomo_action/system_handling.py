import os
import subprocess
import logging
import numpy as np
import re
import sys


class SysHandling:

    def __init__(self, input_list, working_dir, resources_dir,
                 py_main_path):
        self.input_list = input_list
        self.working_dir = working_dir
        self.resources_dir = resources_dir
        self.py_main_path = py_main_path

    def run_programs(self, index, fortran=True, python=True):
        print("current input: " + self.input_list[index])
        # Retrieving input file from given folder.
        os.system("rm input_v2.dat")
        command = (f"cp {self.resources_dir}/{self.input_list[index]}/"
                   f"{self.input_list[index]}.dat input_v2.dat")
        os.system(command)
        # Running python script
        if python:
            print("Running python")
            # TODO: Catch exceptions
            subprocess.call([sys.executable, self.py_main_path])

        # Running fortran script
        if fortran:
            print("running fortran")
            input_file = open("./input_v2.dat")
            _ = subprocess.call(["./tomo_vo.intelmp"], stdin=input_file)
            # Retrieving Fortran output from /tmp/
            fortran_files = SysHandling.find_files("/tmp/")
            for file in fortran_files:
                os.system("mv /tmp/" + file + " " + self.working_dir)
                logging.info("collected " + file + " from /tmp/")

    # Saving output in relevant folder.
    def move_to_resource_dir(self, index):
        print("Output saved to: " + self.resources_dir + "/"
              + self.input_list[index])
        try:
            out_files = os.listdir(self.working_dir)
            for file in out_files:
                os.system("mv " + self.working_dir + "/" + file + " "
                          + self.resources_dir + "/" + self.input_list[index])
        except os.error:
            print("ERROR: FILES NOT MOVED")

    @staticmethod
    def find_files(dirr, search_str=".data"):
        if dirr[-1] != "/":
            dirr += "/"
        file_list = os.listdir(dirr)
        return [s for s in file_list if search_str in s]

    @staticmethod
    def clear_dir(dirr):
        if dirr[-1] != "/":
            dirr += "/"
        file_list = os.listdir(dirr)
        for file in file_list:
            os.remove(dirr + file)
        logging.info("Removed " + str(len(file_list)) + " from " + dirr)

    @staticmethod
    def get_pics_from_file(dirr):
        if dirr[-1] != "/":
            dirr += "/"
        images = SysHandling.find_images(dirr)
        py_picture = np.genfromtxt(dirr + str(images['python']))
        sqrt_arraysize = int(np.sqrt(len(py_picture)))
        py_picture = py_picture.reshape((sqrt_arraysize, sqrt_arraysize))

        ftr_picture = np.genfromtxt(dirr + str(images['fortran']))
        ftr_picture = ftr_picture.reshape((sqrt_arraysize, sqrt_arraysize))

        return py_picture, ftr_picture

    @staticmethod
    def find_images(dirr):
        if dirr[-1] != "/":
            dirr += "/"

        file_list = os.listdir(dirr)
        fortran_image = [s for s in file_list if "image" in s]
        python_image = [s for s in file_list if "py_picture" in s]

        SysHandling._assert_images_found(python_image, fortran_image)

        return {'fortran': fortran_image[0], 'python': python_image[0]}

    @staticmethod
    def _assert_images_found(p_im, f_im):
        if len(p_im) > 1 or len(f_im) > 1:
            raise AssertionError("Moore than two images were found.")
        if len(p_im) == 0 and len(f_im) == 0:
            raise AssertionError("No images found")
        if len(p_im) == 0:
            raise AssertionError("Python image not found")
        if len(f_im) == 0:
            raise AssertionError("Fortran image not found")

    @staticmethod
    def find_file_in_dir(dirr, regex):
        if dirr[-1] != "/":
            dirr += "/"
        file_list = os.listdir(dirr)
        r = re.compile(regex)
        return list(filter(r.match, file_list))

    @staticmethod
    def find_difference_files(dirr):
        if dirr[-1] != "/":
            dirr += "/"
        f_diff_str = SysHandling.find_file_in_dir(dirr, r"\Ad[0-9][0-9][0-9]")
        p_diff_str = SysHandling.find_file_in_dir(dirr, r"\Apy_d[0-9]")
        f_diff = np.genfromtxt(dirr + f_diff_str[0])
        f_diff = np.delete(f_diff, (0), axis=1)
        p_diff = np.genfromtxt(dirr + p_diff_str[0])
        p_diff = np.delete(p_diff, (0), axis=1)
        print("Found difference files: "
              + f_diff_str[0] + " and " + p_diff_str[0] + ".")
        return f_diff, p_diff

    @staticmethod
    def dir_exists(dir_name, path="./"):
        if not os.path.isdir(path + dir_name):
            os.mkdir(path + dir_name)