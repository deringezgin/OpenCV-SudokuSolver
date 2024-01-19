"""File that contains essential functions that are used to read / write / view images."""

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import shutil
import glob
import os


def read_image(path: str, mode: int = 0) -> np.ndarray:
    """Function that reads an image and returns the BGR version by default. To return the Grayscale version mode=1."""
    img = cv.imread(path)
    assert img is not None, "Couldn't find image. Make sure that file path is correct and the file exists."  # Checking if image is not None
    if mode == 0:
        return img
    elif mode == 1:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def read_images(folder_name: str, mode: int = 0) -> list[np.ndarray]:
    """Function that reads the images from a specific folder and returns them in a list"""
    img_files = glob.glob(folder_name + "/*.jpg")
    imgs = [read_image(path, mode) for path in img_files]
    return imgs


def write_images(folder_name: str, img: np.ndarray, i: int) -> None:
    """Function to write images to a specific folder in the project directory"""
    # Creating the output folder path.
    output_folder_path = os.path.join(os.getcwd(), folder_name)

    # Checking if the folder already exists and if this is the first time running or not, if yes, clean it.
    if i == 0 and os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)  # Remove the entire folder and its contents

    # Checking if the folder not exists. If yes, creates the folder.
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)  # Create the empty folder

    # Writing the image to the folder
    output_image_path = os.path.join(output_folder_path, f"frame_{i + 1}.jpg")
    cv.imwrite(output_image_path, img)


def view_image(i, convert_method: int = cv.COLOR_BGR2RGB) -> None:
    """Function that shows the input image using matplotlib. If input is a list, plots all the images."""
    if type(i) is not list:
        plt.imshow(cv.cvtColor(i, convert_method))
        plt.show()
    else:
        for g in i:
            plt.imshow(cv.cvtColor(g, convert_method))
            plt.show()
