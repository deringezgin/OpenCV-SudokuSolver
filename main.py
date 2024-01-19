from preprocess import read_images, view_image
from process import *
from user_validation import check_values, check_images
from solve_sudoku import solve

# Reading the images
images = read_images("raw_data", mode=1)  # Saving the images in grayscale
images_c = read_images("raw_data", mode=0)  # Saving the images in color mode

# Processing the images
fixed_imgs = correct_images(images, images_c)  # Get the perspective corrected version of the raw images
checked_images = check_images(images, fixed_imgs)  # Allowing user to view the detected boards and make any changes
splitted_blocks = extract_blocks(fixed_imgs)  # Extracting the ind. sudoku block from the images
int_values = process_cells(splitted_blocks)  # Detecting the int values of the cells and creating a new array with them
checked_values = check_values(int_values, fixed_imgs)  # Allowing user to view the detected values and make any changes

# Solving the board
checked_values_copy = create_copy(checked_values)  # Saving the board values before solving
is_board_solved = [solve(board) for board in checked_values]  # Solving the board
final_boards = finalize_board(checked_values_copy, checked_values, splitted_blocks, is_board_solved)  # Updating boards
view_image(final_boards)  # Viewing the final versions of the boards
