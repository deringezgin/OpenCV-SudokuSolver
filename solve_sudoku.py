"""File contains functions that are used to solve the Sudoku Board that is extracted. solve(board) uses recursion"""


# Starting point of the Sudoku Solver is taken from this website.
# https://www.techwithtim.net/tutorials/python-programming/sudoku-solver-backtracking


def solve(board: list) -> bool:
    """Solving a sudoku board with two helper functions; is_valid and find_empty_cell"""
    find = find_empty_cell(board)  # Finding an empty cell
    if not find:  # If any empty cell is found, process is complete, return True
        return True
    else:
        row, col = find  # Extract the row, column values from the found cell position

    for i in range(1, 10):  # Trying to place values from 1 to 9 in the empty spot
        if is_valid(board, i, (row, col)):  # If it's a valid position, place the number
            board[row][col] = i

            if solve(board):  # Calling solve(board) recursively to solve the rest of the board
                return True  # If the recursive function returns true, this function returns true too

            board[row][col] = 0  # If not returned True, this value combination doesn't work, pass to the next value
    return False  # If didn't return true until this point, returns False, indicating that couldn't solve the board


def is_valid(board: list, num: int, pos: tuple) -> bool:
    """Checking if putting the num in the relevant pos in the board will result in a valid board combination"""
    row, col = pos  # Separating pos into row, col in order to make the code more understandable

    # RULE 1: A number can be present in a row only for once
    for i in range(len(board[0])):
        # Checking if the num is already present in that row but in a different column
        if board[row][i] == num and col != i:
            return False

    # RULE 2: A number can be present in a column only for once
    for i in range(len(board)):
        # Checking if the num is already present in that column but in a different row
        if board[i][col] == num and row != i:
            return False

    # RULE 3: When we divide the 9x9 board into 9 3x3 regions, all numbers can exist once in a region
    # Finding the region address of the value with floor division.
    box_x = col // 3
    box_y = row // 3
    for i in range(box_y * 3, (box_y + 1) * 3):  # Iterating through the relevant region
        for j in range(box_x * 3, (box_x + 1) * 3):
            # Checking if the num is already present in that region in a different location
            if board[i][j] == num and (i, j) != pos:
                return False
    return True


def find_empty_cell(board: list) -> tuple | None:
    """Function that finds the first empty cell in the Sudoku board and returns the row / column values for it"""
    for i in range(len(board)):  # Iterating through the board
        for j in range(len(board[0])):
            # If it's an empty cell, return the location for it
            if board[i][j] == 0:
                return i, j
    return None  # If no empty cell is found, returns None
