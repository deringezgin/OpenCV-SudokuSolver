"""File that contains functions for user validation of the extracted board and numbers"""

import cv2 as cv
import numpy as np
from process import get_perspective

mouse_coordinates = None


# noinspection PyTypeChecker
def check_images(org_imgs: list, fixed_imgs: list) -> list:
    """Function to check the perspective corrected versions of the images and make corrections if needed"""
    global mouse_coordinates

    print("Check the perspective corrected version of the board\n"
          "If you found any errors in perspective correction, click on the corners in the original image on the left\n"
          "Check the output again. When you're done, press c")

    # Iterating through the original board images and the perspective corrected versions.
    for i in range(len(org_imgs)):
        org_img, fixed_img = org_imgs[i], fixed_imgs[i]
        mouse_locs = []
        if len(org_img.shape) > 2: org_img = cv.cvtColor(org_img, cv.COLOR_BGR2GRAY)
        if len(fixed_img.shape) > 2: fixed_img = cv.cvtColor(fixed_img, cv.COLOR_BGR2GRAY)

        # Checking the image sizes
        height1, width1 = org_img.shape
        height2, width2 = fixed_img.shape

        # Ensure the same height for fixed_img
        fixed_img = cv.resize(fixed_img, (int(width2 * height1 / height2), height1))
        result_image = np.concatenate((org_img, fixed_img), axis=1)

        while True:
            cv.namedWindow("Check Perspective")
            cv.imshow("Check Perspective", result_image)
            cv.setMouseCallback("Check Perspective", click_event_for_board)

            result_image = cv.cvtColor(result_image, cv.COLOR_GRAY2BGR)
            if mouse_coordinates is not None and len(mouse_locs) != 4:  # If mouse is clicked on a valid region
                mouse_locs.append(mouse_coordinates)
                cv.circle(result_image, mouse_coordinates, 10, (255, 255, 255), -1)
                cv.imshow("Check Perspective", result_image)
                mouse_coordinates = None
            result_image = cv.cvtColor(result_image, cv.COLOR_BGR2GRAY)

            if len(mouse_locs) == 4:

                mouse_locs_3d = np.array([[list(loc) for loc in mouse_locs]])

                new_image = get_perspective(org_img, mouse_locs_3d)
                height3, width3 = new_image.shape
                fixed_img = cv.resize(new_image, (int(width3 * height1 / height3), height1))
                checked_image = np.concatenate((org_img, fixed_img), axis=1)

                cv.imshow("Check Perspective", checked_image)

                print("This is the new form of the image. If you are content with the result press, 'c', "
                      "otherwise press 'n' to create a new version.")

                key = cv.waitKey(0)
                if key == ord('c'):
                    fixed_img = cv.resize(fixed_img, (720, 720))
                    fixed_imgs[i] = fixed_img
                    break
                elif key == ord('n'):
                    mouse_locs.clear()
                    result_image = np.concatenate((org_img, fixed_img), axis=1)
            key = cv.waitKey(1)
            if key == ord('c'):
                break

    for i in range(len(fixed_imgs)):
        img = fixed_imgs[i]
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            fixed_imgs[i] = img
    return fixed_imgs


# noinspection PyUnusedLocal
def click_event_for_board(event, x, y, flags, params) -> None:
    """Returns where the mouse is clicked"""
    global mouse_coordinates
    if event == cv.EVENT_LBUTTONDOWN:
        mouse_coordinates = (x, y)


# noinspection PyTupleAssignmentBalance,PyUnresolvedReferences
def check_values(board_values: list, imgs: list) -> list:
    """Function that allows user to check the detected values"""
    global mouse_coordinates

    print("Check values that are detected in the board.\n"
          "If you see any wrong detected values in the digital board on the left, click on the wrong cell.\n"
          "Write the correct int value of the cell.\n"
          "Press c to pass the next board if you don't see any wrong answers.")

    # Iterating through the board image and corresponding values
    for board_img, board_value in zip(imgs, board_values):
        while True:
            combined_dig_board = create_dig_board(board_value)  # Creates a digital board with the values
            if len(board_img.shape) == 3:
                board_img = cv.cvtColor(board_img, cv.COLOR_BGR2GRAY)

            # Merging the original image and created digital board
            org_board_and_dig_board = np.concatenate((board_img, combined_dig_board), axis=1)

            cv.namedWindow("Check Values")
            cv.imshow("Check Values", org_board_and_dig_board)
            cv.setMouseCallback("Check Values", click_event_for_values)
            if mouse_coordinates is not None and mouse_coordinates[0] > 0:  # If mouse is clicked on a valid region
                # Extract coordinates and find the relevant value by doing floor division

                x, y = mouse_coordinates
                x, y = x // 80, y // 80
                value = board_value[y][x]
                while True:  # Taking a replacement value which is numeric and in valid range
                    new_value = input(
                        f"Please enter the value you'd like to replace with the value {value} in the Row: {y + 1}, and Column: {x + 1} --> ")

                    # Check if the entered value is numeric and between 0 and 10
                    if new_value.isdigit() and 0 <= int(new_value) <= 10:
                        break
                    else:
                        print("Please enter a numeric value between 0 and 10.")

                board_value[y][x] = new_value  # Replacing the value
                mouse_coordinates = None

            key = cv.waitKey(1)
            if key == ord('c'):  # If "c" is pressed, pass to the new board image / values pair
                break
    return board_values


# noinspection PyUnusedLocal
def click_event_for_values(event, x, y, flags, params) -> None:
    """Returns where the mouse is clicked"""
    global mouse_coordinates
    if event == cv.EVENT_LBUTTONDOWN:
        mouse_coordinates = (x - 720, y)


def create_dig_cell(value: int) -> np.ndarray:
    """Creates a digital cell with an int value. Returns the digital cell itself."""
    # Creates an empty cell as a white canvas.
    cell = np.zeros((80, 80), np.uint8)
    cell[:] = 255
    cv.rectangle(cell, (0, 0), (79, 79), (0, 0, 0), 2)  # Creating a border
    cv.putText(cell, str(value), (20, 60), 0, 2, (0, 0, 0), 5)  # Writing the value on the cell
    return cell


def create_dig_board(board_values: list) -> np.ndarray:
    """Creates a digital board with an input list. Creates individual digital cells and merges them."""
    dig_board = []
    for row in board_values:  # Iterating through the board_values and creating digital cells
        dig_row = []
        for value in row:
            dig_row.append(create_dig_cell(value))
        dig_board.append(dig_row)

    dig_row_img = []
    for row in dig_board:  # Merging the digital cells
        combined_dig_board = np.concatenate(row, axis=1)  # Combining columns
        dig_row_img.append(combined_dig_board)

    combined_dig_board = np.concatenate(dig_row_img, axis=0)  # Combining rows

    return combined_dig_board
