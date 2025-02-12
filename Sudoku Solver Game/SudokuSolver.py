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
    """
    Provides intelligent hints for the next move based on different solving strategies.
    Orders hints from basic to advanced techniques.
    """
    def get_current_board():
        """Get the current state of the board from GUI"""
        board = []
        for r in range(6):
            row = []
            for c in range(6):
                val = entries[r][c].get()
                try:
                    row.append(int(val) if val else 0)
                except ValueError:
                    messagebox.showerror("Error", "Invalid board values. Please ensure all entries are numbers.")
                    return None
            board.append(row)
        return board

    def find_empty_cells(board):
        """Find all empty cells in the board"""
        empty_cells = []
        for r in range(6):
            for c in range(6):
                if board[r][c] == 0:
                    empty_cells.append((r, c))
        return empty_cells

    def get_possible_values(board, row, col):
        """Get all possible values for a given cell"""
        possible = []
        for num in range(1, 7):
            if is_valid(board, row, col, num):
                possible.append(num)
        return possible

    def find_single_candidate(board):
        """Find cells with only one possible value"""
        empty_cells = find_empty_cells(board)
        for row, col in empty_cells:
            possible = get_possible_values(board, row, col)
            if len(possible) == 1:
                return (row, col, possible[0], "This cell can only be {}.")

    def find_hidden_single(board):
        """Find cells where a number can only go in one place in a row/column/box"""
        # Check rows
        for row in range(6):
            for num in range(1, 7):
                possible_cols = []
                for col in range(6):
                    if board[row][col] == 0 and is_valid(board, row, col, num):
                        possible_cols.append(col)
                if len(possible_cols) == 1:
                    return (row, possible_cols[0], num, 
                           "In this row, {} can only go in this cell.")

        # Check columns
        for col in range(6):
            for num in range(1, 7):
                possible_rows = []
                for row in range(6):
                    if board[row][col] == 0 and is_valid(board, row, col, num):
                        possible_rows.append(row)
                if len(possible_rows) == 1:
                    return (possible_rows[0], col, num, 
                           "In this column, {} can only go in this cell.")

        # Check boxes
        for box_row in range(2):
            for box_col in range(3):
                for num in range(1, 7):
                    possible_positions = []
                    for r in range(box_row*2, (box_row+1)*2):
                        for c in range(box_col*3, (box_col+1)*3):
                            if board[r][c] == 0 and is_valid(board, r, c, num):
                                possible_positions.append((r, c))
                    if len(possible_positions) == 1:
                        row, col = possible_positions[0]
                        return (row, col, num, 
                               "In this 2x3 box, {} can only go in this cell.")

    def highlight_cell(row, col, color='#ffeb3b'):
        """Highlight the cell being hinted at"""
        original_color = entries[row][col].cget('bg')
        entries[row][col].config(bg=color)
        
        # Reset color after 2 seconds
        root.after(2000, lambda: entries[row][col].config(bg=original_color))

    def show_hint_message(message, value):
        """Show hint message in a mini popup"""
        hint_window = tk.Toplevel(root)
        hint_window.title("Hint")
        hint_window.geometry("300x100")
        
        # Position the window near the main window
        x = root.winfo_x() + root.winfo_width()//2 - 150
        y = root.winfo_y() + root.winfo_height()//2 - 50
        hint_window.geometry(f"+{x}+{y}")
        
        ttk.Label(hint_window, text=message.format(value), 
                 wraplength=250, justify='center').pack(pady=20)
        
        # Auto-close after 3 seconds
        root.after(3000, hint_window.destroy)

    # Main hint logic
    board = get_current_board()
    if not board:
        return

    # Try different hint strategies in order of difficulty
    hint = None
    
    # Strategy 1: Look for cells with only one possible value
    hint = find_single_candidate(board)
    
    # Strategy 2: Look for hidden singles if no obvious candidates
    if not hint:
        hint = find_hidden_single(board)
    
    if hint:
        row, col, value, message = hint
        highlight_cell(row, col)
        show_hint_message(message, value)
    else:
        # If no specific hint found, show general helper message
        empty_cells = find_empty_cells(board)
        if empty_cells:
            row, col = empty_cells[0]
            possible = get_possible_values(board, row, col)
            highlight_cell(row, col, '#e3f2fd')
            show_hint_message("Try this cell. Possible values are: {}", 
                            ", ".join(map(str, possible)))
        else:
            messagebox.showinfo("Hint", "The puzzle is already complete!")

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
