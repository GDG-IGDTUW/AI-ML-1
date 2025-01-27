import tkinter as tk
from tkinter import messagebox
from time import time

# TODO: Implement the ability to load puzzles from a file.
# TODO: Implement the ability to save solutions to a file.

def is_valid(board, row, col, num):
    """Checks if placing num in board[row][col] is valid."""
    for c in range(6):
        if board[row][c] == num:
            return False
    for r in range(6):
        if board[r][col] == num:
            return False
    start_row = (row // 2) * 2
    start_col = (col // 3) * 3
    for r in range(start_row, start_row + 2):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False
    return True

def solve_sudoku(board):
    """Solves the Sudoku puzzle using backtracking."""
    # TODO: Show the solving process visually step-by-step.
    for row in range(6):
        for col in range(6):
            if board[row][col] == 0:
                for num in range(1, 7):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def get_board():
    """Retrieves the current board from the input fields."""
    board = []
    for r in range(6):
        row = []
        for c in range(6):
            val = entries[r][c].get()
            # TODO: Add validation to ensure input is a valid integer between 1 and 6 for 6x6 grid.
            try: #error handling for it to run a block of code
                # Convert the value to an integer
                int_val = int(val)
                
                # Check if it's between 1 and 6
                if 1 <= int_val <= 6:
                    row.append(int_val)
                else:
                    raise ValueError(f"Value out of range: {val}")
            except ValueError: # in case the block isn't executed successfully 
                print(f"Invalid input at row {r + 1}, column {c + 1}: '{val}'. Must be an integer between 1 and 6.")
                return None  # return None if invalid input is found
        board.append(row)
    return board

def solve():
    """Solves the Sudoku puzzle and updates the GUI."""
    board = get_board()
    
    #start timer
    start_time = time.time()

    if board is None:
        messagebox.showinfo("Unsolvable", "Please provide a valid puzzle board")
        return
    if solve_sudoku(board):
        for r in range(6):
            for c in range(6):
                if board[r][c] != 0:
                    entries[r][c].delete(0, tk.END)
                    entries[r][c].insert(0, str(board[r][c]))
    else:
        messagebox.showinfo("Unsolvable", "The puzzle cannot be solved")
    # TODO: Display the time taken to solve the puzzle.
    end_time = time.time()
    time_taken = end_time - start_time
    tk.messagebox.showinfo("Time taken", f"Time taken to solve the puzzle: {time_taken:.2f} seconds")

def classify_puzzle():
    """Classify the puzzle as Easy, Medium, or Hard based on complexity."""
    # TODO: Implement puzzle classification based on number of blanks or complexity.
    board = get_board()
    blank_count = sum(row.count(0) for row in board)

    if blank_count < 10:
        difficulty = "Easy"
    elif 10 <= blank_count <= 20:
        difficulty = "Medium"
    else:
        difficulty = "Hard"

    messagebox.showinfo("Puzzle Classification", f"The puzzle is classified as: {difficulty}")


def clear_grid():
    """Clear the Sudoku grid."""
    # TODO: Add a button to clear the grid and restart.
def clear_grid():
    """Clear the Sudoku grid."""
    for r in range(6):
        for c in range(6):
            entries[r][c].delete(0, tk.END)

clear_button = tk.Button(root, text="Clear Grid", font=('Arial', 20), command=clear_grid)
clear_button.grid(row=6, column=3, columnspan=3)
    

def show_metrics():
    """Display metrics such as recursion depth or steps taken."""
    # TODO: Implement functionality to track and show recursion depth or steps taken.

def provide_hint():
    """Provide a hint for the next move without solving completely."""
    # TODO: Implement hint functionality.

# Create the main window
root = tk.Tk()
root.title("6x6 Sudoku Solver")

# Create a 6x6 grid of entry fields
entries = []
for r in range(6):
    row_entries = []
    for c in range(6):
        entry = tk.Entry(root, width=2, font=('Arial', 24), borderwidth=2, relief="solid")
        entry.grid(row=r, column=c, padx=5, pady=5)
        row_entries.append(entry)
    entries.append(row_entries)

# Create buttons for the functionalities
solve_button = tk.Button(root, text="Solve", font=('Arial', 16), command=solve)
solve_button.grid(row=6, column=0, columnspan=3)

# TODO : Add button to clear grid

# TODO: Add buttons for loading puzzles from a file and saving solutions.

root.mainloop()
