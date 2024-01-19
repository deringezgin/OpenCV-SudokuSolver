"""File that contains processing functions used for perspective correction of the initial image, extracting individual cells
and finding numerical values."""

# Starting point for the perspective correction is --> https://data-flair.training/blogs/opencv-sudoku-solver/
# Starting point for the cell evaluation is --> https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
# I made additions and corrections in both of the codes in order to have more accurate predictions

import copy
import cv2 as cv
import numpy as np
import imutils
from preprocess import write_images


def correct_images(imgs: list, imgs_c: list) -> list[np.ndarray]:
    """Function that returns the perspective corrected images and saves them to a new folder called new_images"""
    corrected = []
    for i in range(len(imgs)):
        img, imgc = imgs[i], imgs_c[i]  # Saving the gray and colored images

        # Applying bilateral filter to the grey img. Used bilateral in order to preserve the edges while reducing noise.
        # d is diameter, sigmaColor is the std.dev of the color space. Larger -> More different colors will be included.
        # sigmaSpace is how far the influence of a pixel reaches.
        bfilter = cv.bilateralFilter(img, d=13, sigmaColor=20, sigmaSpace=20)

        # Applying canny edge detection to the filtered image
        # Values under t.1 is considered as not edges. Between 1 & 2 is considered as weak edges. Over 2 is considered as strong edges.
        edged = cv.Canny(bfilter, threshold1=30, threshold2=180)

        # Finding contours in the edge detected image
        # Finding the contours using cv.findContours. mode is the retrieval mode for contour. In this case it's the full hierarchy of nested contours.
        # Method is how the contours are approximated. Simple is faster and more optimized because it only stores the endpoints.
        key_points = cv.findContours(edged, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)

        # Being sure that we have a list of contours. In some versions of OpenCV, cv.findContours can return another type.
        contours = imutils.grab_contours(key_points)

        # Sorting contours by area and selecting the top 50 contours.
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:50]

        location = None
        # Finding the rectangular contour
        for contour in contours:
            # Approx. the contour with a polygon using the Douglas-Peucker algorithm.
            # Epsilon is maximum distance from the contour to its approximation. Closed signifies if the approximated polygon is closed or not.
            approx = cv.approxPolyDP(contour, epsilon=20, closed=True)
            if len(approx) == 4:  # Checking if the approx. has 4 vertices.
                location = approx  # If true, approx. is the location and break the loop.
                break

        # If no rectangular contour is found, print the message and continue
        if location is None:
            print("Couldn't find a rectangular contour.")
        else:
            # If found, apply perspective transformation, append the image to the list
            new_img = get_perspective(imgc, location)
            corrected.append(new_img)

    return corrected


def get_perspective(img: np.ndarray, loc: np.ndarray, height: int = 720, width: int = 720) -> np.ndarray:
    """Function to correct the location array and apply perspective transformation to the image to fit the board in the frame"""
    # Being sure that corners are matched correctly.
    # In some cases the order of the loc. can be wrong and the image can be transformed in the wrong direction
    loc_2d = loc.reshape(-1, 2)  # Converting the location array from 3D to 2D

    # Sorting the locations. Location with the lowest x & y would be the bottom left, highest x & y would be top right.
    sorted_locations = loc_2d[np.argsort(loc_2d[:, 0])]  # Sorting by x
    left, right = sorted_locations[:2], sorted_locations[2:]  # Left has leftmost points, right has rightmost points
    left, right = left[np.argsort(left[:, 1])], right[np.argsort(right[:, 1])]  # Sorting to find points in top / bottom

    # Creating numpy arrays storing the current and target locations.
    pts1 = np.float32([left[0], left[1], right[0], right[1]])
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

    # Applying perspective transform
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    result = cv.warpPerspective(img, matrix, (width, height))
    return result


def extract_blocks(imgs: list) -> list[list[list[np.ndarray]]]:
    """Given a list of fixed sudoku grids, splits the grid into ind. cells and returns the grids in a new list"""
    spilitted_images = []
    for img in imgs:  # Looping through the images
        rows = np.vsplit(img, indices_or_sections=9)  # Splitting the image to rows
        table = []
        for row in rows:
            block = np.hsplit(row, indices_or_sections=9)  # Splitting the rows into ind. cells.
            table.append(block)  # Appending cells to the table
        spilitted_images.append(table)

    return spilitted_images


# noinspection DuplicatedCode
def process_cells(boards: list) -> list[list[list[int]]]:
    """Function to process the blocks, detect the contours and numbers"""
    # Loading the model files
    samples = np.loadtxt('generalsamples.data', np.float32)
    responses = np.loadtxt('generalresponses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    # Initialize KNearest model
    model = cv.ml.KNearest_create()

    # Training the model
    model.train(samples, cv.ml.ROW_SAMPLE, responses)

    combines_board_values = []
    for board in boards:
        board_values = []
        for row in board:
            row_values = []
            for cell in row:
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
                cell_value = 0

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

                    # Increase the space from every side by 5 percent
                    x -= int(w * 5 / 100)
                    y -= int(h * 5 / 100)
                    w += int(w * 5 / 100) * 2
                    h += int(h * 5 / 100) * 2

                    # Ensure the coordinates and dimensions are within valid ranges
                    x = max(x, 0)
                    y = max(y, 0)
                    w = min(w, width)
                    h = min(h, height)

                    # Calculating the distance between the center of the contour and the center of the cell
                    contour_center = (x + w // 2, y + h // 2)
                    distance = np.linalg.norm(np.array(contour_center) - np.array((width // 2, height // 2)))
                    # If the distance is less than 20, this means that contour is close to the center of the cell and this is a valid contour
                    if distance < 20:
                        # Determining a region of interest and predicting the numerical value with the model
                        roi = thresh[y:y + h, x:x + w]
                        roismall = cv.resize(roi, (10, 10))
                        roismall = roismall.reshape((1, 100))
                        roismall = np.float32(roismall)
                        retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                        cell_value = int((results[0][0]))
                row_values.append(cell_value)
            board_values.append(row_values)
        combines_board_values.append(board_values)

    return combines_board_values


def finalize_board(initial_values: list, final_values: list, cells: list, validation: list) -> list[np.ndarray]:
    """Function that writes the solved values on the board and returns the final board"""
    # Writing the values on the cells, passing if the program couldn't solve the board
    for i in range(len(initial_values)):
        if not validation[i]:
            continue
        for j in range(len(initial_values[i])):
            for k in range(len(initial_values[i][j])):
                if initial_values[i][j][k] == 0:
                    cell = cells[i][j][k]
                    cv.putText(cell, str(final_values[i][j][k]), (20, 65), 0, 2, (0, 0, 255), 5)  # Filling the cell
                    cells[i][j][k] = cell

    # Merging the cells, creating boards and appending them
    merged_boards = []
    for i in range(len(initial_values)):
        merged_rows = []
        for j in range(len(initial_values[i])):
            merged_rows.append(np.concatenate(cells[i][j], axis=1))
        merged_boards.append(np.concatenate(merged_rows, axis=0))

    # Labeling the boards that couldn't be solved
    for i in range(len(initial_values)):
        if not validation[i]:
            cv.rectangle(merged_boards[i], (0, 300), (720, 500), (0, 0, 0), -1)
            cv.putText(merged_boards[i], str("Couldn't Solve the Board"), (55, 360), 0, 1.5, (0, 0, 255), 5)
            cv.putText(merged_boards[i], str("Try Again Please!"), (75, 460), 0, 2, (0, 0, 255), 5)

    # Writing the final results to an output file
    for i in range(len(merged_boards)):
        write_images("final_results", merged_boards[i], i)

    return merged_boards


def create_copy(l: list) -> list:
    """Function that creates a copy of a list of any dimensions"""
    return copy.deepcopy(l)
