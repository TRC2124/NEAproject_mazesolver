import tkinter as tk
from generating import MazeGenerator
from solving import MazeSolver

def main_menu():
    root = tk.Tk() #create root window
    root.title("Main Menu") #window title

    title_label = tk.Label(root, text="Main Menu", font=("Arial", 16)) #Create a large label
    title_label.pack(pady=20) #20 pixels below top

    button_frame = tk.Frame(root) #create frame container for buttons
    button_frame.pack(pady=20) #place 20 pixel below title
    
    solve_btn = tk.Button(
        button_frame, #parent container
        text="Solve Maze", #text label
        command=lambda: open_solver(root), #click handler, calls open_solver
        width=15, #dimensions
        height=3
    )
    solve_btn.grid(row=0, column=0, padx=10) #left side with 10px padding

    generate_btn = tk.Button( #same as before
        button_frame, 
        text="Generate Maze", 
        command=lambda: open_generator(root), #calls open_generator this time 
        width=15, 
        height=3
    )
    generate_btn.grid(row=0, column=1, padx=10) #place beside solve button

    root.mainloop() #Start tkinter event loop

def open_generator(current_root): 
    current_root.destroy() #destroy menu window
    generator = MazeGenerator() #crate mazegen instance
    generator.back_button.config(command=lambda: close_generator(generator)) #configure for back button to call close generator
    generator.mainloop() #start event loop

def open_solver(current_root):
    current_root.destroy() #destroy menu window
    solver = MazeSolver() #crate solving instance
    solver.back_button.config(command=lambda: close_solver(solver)) #configure for back button to call close solver
    solver.mainloop() #start event loop

def close_generator(generator): #button handler
    generator.destroy() #close window
    main_menu() #create main menu

def close_solver(solver):
    solver.destroy() #close window
    main_menu() #create main menu

if __name__ == "__main__": #execution entry point
    main_menu() #start by calling main_menu