"""Model Creation Script"""

# Starting point for model training --> https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
# I also added an extra training with the existing cell values

from preprocess import read_images
from process import *
import cv2 as cv
import numpy as np
import sys


# noinspection DuplicatedCode
def create_model_from_input(boards: list) -> None:
    """Function that creates a model with the pitrain.png and existing sudoku values"""
    samples = np.empty((0, 100))
    responses = []
    keys = [i for i in range(48, 58)]

    # Iterating through the sudoku cells and training the model
    for cell in (cell for board in boards for row in board for cell in row):
        height, width = cell.shape[:2]  # Taking the properties of the cell

        # Applying filtering and threshold.
        cell = cv.bilateralFilter(cell, d=13, sigmaColor=20, sigmaSpace=20)
        gray = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), np.uint8)
        gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        thresh = cv.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

        # Detecting contours.
        contours = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
        max_contour = None
        max_area = 0

        # Looping through contours
        for cnt in contours:
            [x, y, w, h] = cv.boundingRect(cnt)

            # Check if the contour is in the 5% from the borders
            # Checking:
            # 1. Contour is 5% away from the borders,
            # 2. Contour area is larger than 60,
            # 3. Contour's perimeter is larger than 20.
            border_percentage = 5
            if (x > width * border_percentage / 100 and
                    y > height * border_percentage / 100 and
                    x + w < width * (100 - border_percentage) / 100 and
                    y + h < height * (100 - border_percentage) / 100 and
                    cv.contourArea(cnt) > 60 and
                    cv.arcLength(cnt, True) > 20):

                # Checking if the contour is the largest area contour.
                contour_area = cv.contourArea(cnt)
                if contour_area > max_area:
                    max_area = contour_area
                    max_contour = cnt

        # If a large contour is found,
        if max_contour is not None:
            [x, y, w, h] = cv.boundingRect(max_contour)

            # Calculating the distance between the center of the contour and the center of the cell
            contour_center = (x + w // 2, y + h // 2)
            distance = np.linalg.norm(np.array(contour_center) - np.array((width // 2, height // 2)))

            # If the distance is less than 20, this means that contour is close to the center of the cell.
            if distance < 20:
                cv.rectangle(cell, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv.resize(roi, (10, 10))

                # Showing the image with the found contour and requesting the numerical input from the user
                cv.imshow("norm", roi)
                key = cv.waitKey(0)
                if key == 27:
                    continue
                elif key in keys:  # If pressed key is valid append it to the responses and samples
                    responses.append(int(chr(key)))
                    print(int(chr(key)))
                    sample = roismall.reshape((1, 100))
                    samples = np.append(samples, sample, 0)
            else:
                print("None")
        cv.waitKey(10)

    # Doing the same process with the pitrain.png
    im = cv.imread("pitrain.png")
    im = im.copy()

    # Preprocessing
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # Detecting and filtering contours. Taking input from the user and creating the model.
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) > 50:
            [x, y, w, h] = cv.boundingRect(cnt)

            if h > 28:
                cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv.resize(roi, (10, 10))
                cv.imshow("norm", roi)
                key = cv.waitKey(0)

                if key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    print(int(chr(key)))
                    sample = roismall.reshape((1, 100))
                    samples = np.append(samples, sample, 0)

    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))
    print("Training Complete")

    # Saving the model
    np.savetxt("generalsamples.data", samples)
    np.savetxt("generalresponses.data", responses)


# Model creation
images = read_images("raw_data", mode=1)  # Saving the images in grayscale
images_c = read_images("raw_data", mode=0)  # Saving the images in color mode
fixed_imgs = correct_images(images, images_c)  # Get the perspective corrected version of the raw images
splitted_blocks = extract_blocks(fixed_imgs)  # Extracting the ind. sudoku block from the images
create_model_from_input(splitted_blocks)  # Creating a model using the existing pitrain.png and block values
