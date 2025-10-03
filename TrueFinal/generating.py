import random #used for shuffling edges to create a random maze
import numpy as np #used to increase the efficiency of lists in python
from PIL import Image, ImageTk #Used to invert the colour of the image of the maze
import tkinter as tk #for GUI features
from tkinter import messagebox #for errors
import time
        
class MazeGenerator(tk.Tk):
    def __init__(self):
        super().__init__() #Initalise TKinter
        self.title("Maze Generator") #Window title

        self.width_label = tk.Label(self, text="Width:") #create the fields for the user to enter maze size
        self.width_label.grid(row=0, column=0)
        self.width_entry = tk.Entry(self)
        self.width_entry.grid(row=0, column=1)

        self.height_label = tk.Label(self, text="Height:")
        self.height_label.grid(row=1, column=0)
        self.height_entry = tk.Entry(self)
        self.height_entry.grid(row=1, column=1)

        self.generate_button = tk.Button(self, text="Generate Maze", command=self.generate_maze) #create button to start maze solving, links to generate_maze
        self.generate_button.grid(row=2, column=0, columnspan=2)

        self.image_label = tk.Label(self)
        self.image_label.grid(row=3, column=0, columnspan=2)
        self.image_label.image = None #keep a reference to prevent garbage collection
        
        self.back_button = tk.Button(self, text="Back", command=self.back) #dummy back button for now
        self.back_button.grid(row=4, column=0, columnspan=2)
        
    def back(self):
        self.destroy()
     
    def find(self, parent, i):#When 2 cells are to be  to see if they should be joined, their root is found using this function to determine their set.
        #parent: list representing the parent of each element, i: element whose set is to be found.
        if parent[i] != i: #if i is not the root of the set
            parent[i] = self.find(parent, parent[i])#recursively find the root of the set and set the parent of i to be the root directly
        return parent[i]#return the root of the set

    def union(self, parent, rank, x, y): #When 2 cells are decided to be merged, union is performed using rank by union
        #used to merge sets when connecting cells
        #parent: list representing the parent of each element, rank: list representing the rank of each set
        #x and y: the elements to be joined
        xroot = self.find(parent, x) #find the root of the set containing x using the find function
        yroot = self.find(parent, y) #find the root of the set containing y using the find function

        #attach the smaller rank tree under the root of the higher rank tree
        #helps to keep the trees flatter
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        #if ranks are the same, make root of x the root of y and increment its rank by 1
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
            
    def kruskal_maze(self, width, height): #function for generating the maze
        cells = width * height #calculate the total amount of cells in the maze, used to initalise parent and rank
        parent = list(range(cells))#represents all cells in the maze. When the program starts, all cells are their own parents [0, 1, 2, 3, 4, ...]
        rank = [0] * cells #all cells have a rank of zero when the program starts [0, 0, 0, 0, ...]

        edges = [] #create an empty list to be filled with all empty edges
        for y in range(height): #iterate through rows
            for x in range(width): #iterate through colums
                if x < width - 1: #check if there is a cell to the right
                    edges.append((y * width + x, y * width + x + 1)) #append the coordiate of the edge, example [(0, 1), (0, 50), (1, 2), (1, 51), ...]
                if y < height - 1: #check if there is a cell to the bottom
                    edges.append((y * width + x, (y + 1) * width + x))  #append the coordinate of the edge

        random.shuffle(edges)  #radndomize the edges, so that the final maze is random

        maze = np.ones((2 * height + 1, 2 * width + 1), dtype=np.uint8)  # Initialize with walls (1s)
        maze[1::2, 1::2] = 0  # Set cells as paths (0s)

        for u, v in edges: #for loop, every item in the list of all edges (This loop is is ued to join cells)
            if self.find(parent, u) != self.find(parent, v): #If the cells are connected elsewhere we move on to the next edge, if they are in differnet sets we proceed
                self.union(parent, rank, u, v) #Call the union function. The trees are merged.
                uy, ux = u // width, u % width #calculate the row and column indicies of the 2 cells connected by the edge
                vy, vx = v // width, v % width #divide index of cell by width of grid to get row, taking remainder to get column
                my, mx = (uy + vy) + 1, (ux + vx) + 1 #taking the average of the 2 cells to get coordinate of edge in self.maze, add 1 to x and y to account for the surrounding walls.
                maze[my, mx] = 0  # Carve path (set to 0)

        return maze #send the completed bitmap back

    def save_maze_text(self, maze, filename): #convert maze to a text file
        with open(filename, "w") as f: #open file in write mode
            for row in maze: #iterate over each row
                f.write("".join(str(x) for x in row) + "\n") #convert each line to a single string, write to file and move to new line

    def save_maze_image(self, maze, filename, cell_size=10): #new much easier method to convert bitmap into png
        image = Image.fromarray((1 - maze) * 255).convert("RGB") #invert maze array, scale and convert to RGB values and make it a photo using pillow
        image = image.resize((maze.shape[1] * cell_size, maze.shape[0] * cell_size), Image.NEAREST) #
        image.save(filename) #save the image

    def generate_maze(self): #handling of clicking the generate button
        st = time.time()
        try:
            width = int(self.width_entry.get()) #get the entered dimentions
            height = int(self.height_entry.get())

            if width <= 4 or height <= 4: #implement maze size restictions (min 5 max 2000)
                raise ValueError("Dimensions must be at least 5.") #warn user
            if width > 2000 or height > 2000:
                raise ValueError(f"Maximum maze size is 2000x2000.")

        except ValueError as e: #any other error
            messagebox.showerror("Error", str(e)) #warn user
            return

        maze = self.kruskal_maze(width, height) #generate the maze
        self.save_maze_text(maze, "maze.txt") #save maze bitmap
        self.save_maze_image(maze, "maze.png", cell_size=2) #save maze image

        #display the maze image
        img = Image.open("maze.png") #open photo file
        img = img.resize((400, 400))  #resize to 400x400
        photo = ImageTk.PhotoImage(img) #convert into tkinter image

        self.image_label.config(image=photo) #show image
        self.image_label.image = photo  # Keep a reference
        et = time.time()
        elt = et - st
        print(elt)

        messagebox.showinfo("Success", "Maze generated and saved as maze.txt and maze.png") #success message


if __name__ == "__main__":
    app = MazeGenerator()
    app.mainloop()