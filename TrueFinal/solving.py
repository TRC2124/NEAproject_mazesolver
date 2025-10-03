import tkinter as tk #import tkinter
from tkinter import filedialog, messagebox, Toplevel #import features
import numpy as np #bitmap functions
from PIL import Image, ImageDraw, ImageTk, UnidentifiedImageError #pillow for image processing, unident for easy error handling
import time #for testing only
import heapq #for A* 
from collections import deque #for BFS

class MazeSolver(tk.Tk): #inherit tkinter
    def __init__(self):
        super().__init__() #init tkinter
        self.title("Maze Solver") #window title

        self.maze_image = None #image to show
        self.original_maze_image = None #for reverting when needed
        self.maze_bitmap = None #for storing the bitmap
        self.displayed_image = None #store the pillow image

        self.visualize_var = tk.BooleanVar(value=True) #initalze checkboxes, values preset
        self.without_bitmap_var = tk.BooleanVar(value=False)

        self.start_coord_var = tk.StringVar() #the start and end points for non bitmap solving
        self.end_coord_var = tk.StringVar()

        self.create_UI() #initalise UI

    def create_UI(self): #create the UI
        self.import_image_button = tk.Button(self, text="Import Maze Image", command=self.import_image)# import image button
        self.import_image_button.grid(row=0, column=0) #location

        self.import_bitmap_button = tk.Button(self, text="Import Maze Bitmap", command=self.import_bitmap) #import bitmap button
        self.import_bitmap_button.grid(row=1, column=0) #location

        self.visualize_checkbutton = tk.Checkbutton(self, text="Visualize", variable=self.visualize_var) #checkbox toggle for visualise
        self.visualize_checkbutton.grid(row=2, column=1) #location

        self.without_bitmap_checkbutton = tk.Checkbutton(self, text="Without Bitmap", variable=self.without_bitmap_var, command=self.toggle_bitmap_mode)
        self.without_bitmap_checkbutton.grid(row=3, column=1) #toggle bitmap mode, calls toggle functions to disable import bitmap button and visualize checkbox

        self.start_coord_label = tk.Label(self, text="Start (x, y):") #set up textboxes for entering coords
        self.start_coord_label.grid(row=4, column=0, sticky="e") #location, stick to right side
        self.start_coord_entry = tk.Entry(self, textvariable=self.start_coord_var) #link the input to the variable
        self.start_coord_entry.grid(row=4, column=1) #location

        self.end_coord_label = tk.Label(self, text="End (x, y):") #same as the last 4 lines
        self.end_coord_label.grid(row=5, column=0, sticky="e")
        self.end_coord_entry = tk.Entry(self, textvariable=self.end_coord_var)
        self.end_coord_entry.grid(row=5, column=1)

        self.solve_button = tk.Button(self, text="Solve Maze", command=self.solve_maze) #button to start solving, calls solve maze function
        self.solve_button.grid(row=6, column=0)  #location

        self.clear_button = tk.Button(self, text="Clear", command=self.clear_maze) # Clear button
        self.clear_button.grid(row=6, column=1) #location

        self.image_label = tk.Label(self) #The photo of the maze
        self.image_label.grid(row=7, column=0, columnspan=2)  #location, full width
        
        self.save_button = tk.Button(self, text="Save Solved Maze", command=self.save_image, state=tk.DISABLED) #save button, disabled initially
        self.save_button.grid(row=8, column=1) #location, next to back button

        self.back_button = tk.Button(self, text="Back", command=self.back)#back button
        self.back_button.grid(row=8, column=0)#location, bottom of UI

        self.toggle_bitmap_mode() #call to set inital state

    def back(self): #palceholder for now
        self.destroy()

    def toggle_bitmap_mode(self): #Controls the UI when the without bitmap checkbox is changed
        use_bitmap = not self.without_bitmap_var.get() #inverse to define if user is using a bitmap
        self.import_bitmap_button.config(state=tk.NORMAL if use_bitmap else tk.DISABLED) #disable and gray out the button, the user does not have a bitmap
        self.visualize_checkbutton.config(state=tk.NORMAL if use_bitmap else tk.DISABLED) #disable and grey out the checkbox, visualising is disabled for non-bitmap solving
        self.visualize_var.set(use_bitmap) #make sure the prorgam does not use bitmap solving mode when there is no bitmap
        self.start_coord_label.grid_remove() if use_bitmap else self.start_coord_label.grid() #show or hide the labels and textboxes depending on the use-bitmap variable
        self.start_coord_entry.grid_remove() if use_bitmap else self.start_coord_entry.grid()
        self.end_coord_label.grid_remove() if use_bitmap else self.end_coord_label.grid()
        self.end_coord_entry.grid_remove() if use_bitmap else self.end_coord_entry.grid()
        if not use_bitmap: #Force visualize to be false if it is checked before this function is called
            self.visualize_var.set(False)
            
    def save_image(self):
        if self.displayed_image: # Check if there is an image to save
            filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                                                    title="Save Solved Image") #set file type and initalze filedialog
            if filename: #When set
                try:
                    self.displayed_image.save(filename) #save at location
                    messagebox.showinfo("Success", "Image saved successfully!") #message
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image: {e}") #handle error
        else:
            messagebox.showinfo("Info", "No image to save.") #no image to save

    def import_image(self): #import the photo into the program
        filename = filedialog.askopenfilename(title="Select Maze Image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]) #Use filedialog to get the desired image, file types defined
        if filename: #When file is chosen
            try:
                img = Image.open(filename).convert('RGB') #use pillow to convert the picture to RGB mode
                self.maze_image = img.convert('L') if img.mode != 'L' else img #then convert into grayscale, if already in grayscale for whatever reason, keep it as is.
                self.original_maze_image = self.maze_image.copy() # Store original image for when using the clear button
                self.display_image(self.maze_image) #call functions to resize and display image
            except FileNotFoundError: #Various error handlers to warn the user of incompatiable images
                messagebox.showerror("Error", "File not found.") #Specified file cannot be found
            except UnidentifiedImageError:
                messagebox.showerror("Error", "Unable to open image.") #specified file cannot be opened
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}") #any other error

    def import_bitmap(self): #when the import bitmap button is pressed, we use filedialog to select the file and then save the bitmap to a numpr array
        filename = filedialog.askopenfilename(title="Select Maze Bitmap", filetypes=[("Text files", "*.txt")]) #same method as before, only open txt now
        if filename: #When chosen
            try:
                with open(filename, "r") as f: #open using read mode
                    self.maze_bitmap = np.array([[int(c) for c in line.strip()] for line in f]) #read the contents and save as numpy array, strip removes whitespaces from each line
            except Exception as e: #For any error, warn user
                messagebox.showerror("Error", f"Failed to load bitmap: {e}")

    def clear_maze(self): #when the clear maze button is pressed, reload to the original image
        self.maze_image = self.original_maze_image.copy() if self.original_maze_image else None #reset to original or none if not present
        self.maze_bitmap = None #Clear current loaded bitmap
        self.displayed_image = None #remove loaded imge
        self.start_coord_var.set("") #remove loaded coords
        self.end_coord_var.set("")
        self.image_label.config(image=None) # Clear displayed image
        if self.maze_image:
            self.display_image(self.maze_image) # Redisplay original image if available
        self.save_button.config(state=tk.DISABLED) # Disable save button when maze is cleared

    def _bfs(self, maze, start, end): #protected method for BFS bitmap no vis
        queue = [(start, [start])] #first in first out queue. starting coordinates and path taken
        visited = set() #empty set, keep track of visited cells

        while queue: #as long as the queue is not empty
            (x, y), path = queue.pop(0) #dequeue and pop first element
            if (x, y) == end: #if current coords are the end coords
                return path #return the found path

            visited.add((x, y)) #if not end, mark current as visited

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: #up down left right from current cell
                nx, ny = x + dx, y + dy #add the difference to get actial coord of neighbour cell
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0 and (nx, ny) not in visited: #check if cell is in bounds, path and not visited
                    queue.append(((nx, ny), path + [(nx, ny)])) #if all conditions are met, add to the end of the queue
        return None #eveyrthing is searched and no path is found

    def _a_star(self, maze, start, end): ##proteched method for A* no vis
        def heuristic(a, b): #for calculating absolute distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) #easier than pythagoras

        queue = [(0, start, [start])] #tuple containing (f_score, current cell coords, path so far). f-score(g_score+heuristic) is 0 at start
        visited = set() #keep track of visited cells
        g_scores = {start: 0} #dictionary of g-scores (score from start coords), starts form 0

        while queue: #as long as the queue is not empty
            f_score, (x, y), path = heapq.heappop(queue) #pop lowest value from queue

            if (x, y) == end: #if current coord is the end
                return path #we have found the path, return it

            visited.add((x, y)) #mark current cell is visited

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: #Same with bfs, up down left right
                nx, ny = x + dx, y + dy #calculate difference
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0 and (nx, ny) not in visited: #in bounds, path, not visited
                    new_g_score = g_scores.get((x, y), 0) + 1 #get g-score. if current coord does not have record, set 0 then add 1 to all (1 step)
                    if (nx, ny) not in g_scores or new_g_score < g_scores[(nx, ny)]: #check existing g-score(if present) see if a lower score was found
                        g_scores[(nx, ny)] = new_g_score #if this is best current path, update the g-score
                        f_score = new_g_score + heuristic((nx, ny), end) #calculate new f_score (g_score + heuristic)
                        heapq.heappush(queue, (f_score, (nx, ny), path + [(nx, ny)])) #push to priority queue
        return None #if no path found, return none

    def draw_path(self, image, path, visited=None): #visualisation of the path
        draw = ImageDraw.Draw(image) #cnovert to imagedraw
        line_width = 3 #fixed width
        red_color = (255, 0, 0) #define rgb values of lines and boxes
        lightblue_color = (173, 216, 230)
        
        if self.maze_bitmap is not None: #check if bitmap is present
            cell_size_x = image.width // self.maze_bitmap.shape[1] #calculate width of cells based on image width and number of columns
            cell_size_y = image.height // self.maze_bitmap.shape[0] #calculate height if cells based on image height and number of rows
        else: #no bitmap
            cell_size_x = 1 #needed for solving without bitmap, assume 1:1 mapping (using all pixels)
            cell_size_y = 1

        if visited and self.maze_bitmap is not None: #showing light blue boxes(visited cells), requires visited set and the bitmap
            for row, col in visited: #iterate through each row and column in visited
                x1, y1 = col * cell_size_x, row * cell_size_y #calculate top left corner pixel coords at col, row
                x2, y2 = (col + 1) * cell_size_x, (row + 1) * cell_size_y #calculate bottom right pixel of coords at col, row
                draw.rectangle((x1, y1, x2, y2), fill=lightblue_color) #draw the rectangle and add colour

        for i in range(len(path) - 1): #iterate through path coords and draw lines to connect them. -1 goes to second to last point
            row1, col1 = path[i] #get current point
            row2, col2 = path[i + 1] #get next point to be connected
            if self.maze_bitmap is not None: # if there is a bitmap
                draw.line((col1 * cell_size_x + cell_size_x // 2, row1 * cell_size_y + cell_size_y // 2,
                           col2 * cell_size_x + cell_size_x // 2, row2 * cell_size_y + cell_size_y // 2),
                          fill=red_color, width=line_width) #calculate coords of centre of the cell and link them, fill with red and calculated width
            else:
                draw.line((col1, row1, col2, row2), fill=red_color, width=line_width) #cannot calculate centre of cell as solving pixel by pixel


    def display_image(self, image): #take the PIL image, resize with aspect ratio, convert to tkinter image, the display it
        new_height = 800 #fixed height limit
        aspect_ratio = image.width / image.height #get the width to height ratio
        new_width = int(new_height * aspect_ratio) #calculat the new width based on ratio

        resized_image = image.resize((new_width, new_height)) #resize to the new width and height
        photo = ImageTk.PhotoImage(resized_image) #convetr to tkinter image

        self.displayed_image = image
        self.image_label.config(image=photo) #display the image
        self.image_label.image = photo #keep image alive to prevent deletion

    def solve_maze(self): #core of the program. This is called when solve maze button is clicked on the UI
        #This if ensures that each time solving is started, we use a clean maze image to avoid drawing on top of each other
        if self.original_maze_image: #check if there is an original image
            self.maze_image = self.original_maze_image.copy() #if so, copy the original and save as maze.image, discarding everything done to it
            self.display_image(self.maze_image) #display the image

        if self.without_bitmap_var.get(): #Check of without bitmap checkbox is checked
            #we are in pixel by pixel solving mode now
            if self.maze_image is None: ##if there is no image imported
                messagebox.showerror("Error", "Please import maze image first.") #prompt user to import image
                return #stops executing
            start_coord_str = self.start_coord_var.get() #get starting and ending coords
            end_coord_str = self.end_coord_var.get()
            try: #convert user input to coords and validation
                start_x, start_y = map(int, start_coord_str.replace(" ", "").split(',')) #save start coords by removing spaces and seperating numbers by the comma
                end_x, end_y = map(int, end_coord_str.replace(" ", "").split(',')) #same with the ending coords
                start = (start_y, start_x) #now we have the starting and ending coords
                end = (end_y, end_x)
            except ValueError: #If any error is caught
                messagebox.showerror("Error", "Invalid coordinate format. Please use 'x, y' format (e.g., '15, 15').") #prompt user to correct values and provides an example
                return #stops executing

            def choose_algorithm_image(algorithm): #nested function for when the user clicks on a button on the choose algorithm window (created below this function)
                algo_window.destroy() #close the choose algorithm window
                st = time.time()
                if algorithm == "bfs": #if user choose BFS
                    pathfinding_function = self.find_path_image_bfs #we will use BFS
                elif algorithm == "a*": #if user chooses A*
                    pathfinding_function = self.find_path_image_astar #we will use A*

                path = pathfinding_function(self.maze_image, start, end) #Call whatever function was chosen above and save to path
                if path: #if we have a path returned
                    image_copy = self.maze_image.copy().convert('RGB') #convert to RGB for drawing (double confirm)
                    self.draw_path(image_copy, path) #call draw path
                    self.display_image(image_copy) #display final image
                    et = time.time()
                    elt = et-st
                    print(elt)
                    self.save_button.config(state=tk.NORMAL) # Enable save button after solving
                else: #if the solving algorithm determines there is no solution
                    messagebox.showinfo("Info", "No solution found, check entered coordinates.") #inform the user and recommended steps to resolve the problem

            #code for creating the choose algorithm window. Shows when the validations are passed
            algo_window = Toplevel(self) #first, the window must be on the top level
            algo_window.title("Choose Algorithm") #set window title
            bfs_button = tk.Button(algo_window, text="BFS", command=lambda: choose_algorithm_image("bfs")) #create BFS button and when pressed, use labmda function to pass bfs to the function above
            bfs_button.pack(pady=5) #show the button with padding
            a_star_button = tk.Button(algo_window, text="A*", command=lambda: choose_algorithm_image("a*")) #same as above, but passes a* instead
            a_star_button.pack(pady=5) #show the button

        else: #if without bitmap checkbox is unchecked
            #We are now in bitmap solving mode
            if self.maze_bitmap is None or self.maze_image is None: #if bitmap or image is not imported
                messagebox.showerror("Error", "Please import both image and bitmap first.") #notify user and how to proceed
                return #stop executing

            if self.visualize_var.get(): #150x150 limit if visualizing
                if self.maze_bitmap.shape[0] > 302 or self.maze_bitmap.shape[1] > 302:
                    messagebox.showerror("Error", "Maze bitmap dimensions exceed 150x150 limit for visualized solving.")
                    return  #stop executing
            else: #not visualizing, 1000x1000 limit
                if self.maze_bitmap.shape[0] > 2002 or self.maze_bitmap.shape[1] > 2002:#account for pixel count
                    messagebox.showerror("Error", "Maze bitmap dimensions exceed 1000x1000 limit.")
                    return  #stop executing

            start = (1, 1) #in bitmap solving mode, only path is from top left to bottom right, Define starting coords
            end = (self.maze_bitmap.shape[0] - 2, self.maze_bitmap.shape[1] - 2) #bottom right coords by using .shape

            def choose_algorithm_bitmap(algorithm): #Similar to pixel by pixel mode for defining which solving algorithm to use
                st = time.time()
                algo_window.destroy() #destroy choosing window
                if algorithm == "bfs": #if use choose bfs
                    pathfinding_function = self._bfs #set solving function
                elif algorithm == "a_star": #if A* chosen
                    pathfinding_function = self._a_star #set solving function

                if self.visualize_var.get(): #If the checkbox is checked for visualise maze solving
                    self.solve_maze_with_visualization(start, end, pathfinding_function) #call mentioned function(below) along with the desired solving algorithm
                else: #no visualizing solving. Parts in this else clause are the same as pixel by pixel solving, refer to that part
                    path = pathfinding_function(self.maze_bitmap, start, end)
                    if path:
                        image_copy = self.maze_image.copy().convert('RGB')
                        self.draw_path(image_copy, path)
                        self.display_image(image_copy)
                        et = time.time()
                        elt = et-st
                        print(elt)
                        self.save_button.config(state=tk.NORMAL) # Enable save button after solving
                    else:
                        messagebox.showinfo("Info", "No solution found.")

            #window for choosing algorithm, same as pixel by pixel mode
            algo_window = Toplevel(self)
            algo_window.title("Choose Algorithm")
            bfs_button = tk.Button(algo_window, text="BFS", command=lambda: choose_algorithm_bitmap("bfs"))
            bfs_button.pack(pady=5)
            a_star_button = tk.Button(algo_window, text="A*", command=lambda: choose_algorithm_bitmap("a_star"))
            a_star_button.pack(pady=5)


    def solve_maze_with_visualization(self, start, end, pathfinding_function): #Maze solving with visualisation, A* and BFS combined into a single fucntion
        def heuristic(a, b): #for A*
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) #absolute distance betwen 2 points

        visited = set() #track visited cells when visualsisating

        if pathfinding_function == self._bfs: #if using BFS
            queue = [(start, [start])] #first in first out queue
        else: #if A*
            queue = [(0, start, [start])] #priority queue with inital f-score of 0

        if pathfinding_function == self._a_star: #if using A*
            g_scores = {start: 0} #initalise a dictionary
        else:
            g_scores = {} #empty dictionary (not used for BFS)

        while queue: #While there are still paths to explore
            if pathfinding_function == self._bfs: #if using BFS
                (x, y), path = queue.pop(0) #pop for BFS
            else: #A*
                f_score, (x, y), path = heapq.heappop(queue) #dequeue lowest f-score value for A*

            if (x, y) == end: #have we found path?
                temp_image = self.maze_image.copy().convert('RGB') #if so, convert image to RGB(insurance)
                self.draw_path(temp_image, path, visited=visited) #Call draw path function with the visited set to draw the blue pixels
                self.display_image(temp_image) #display image
                self.save_button.config(state=tk.NORMAL) #enable save button
                return #WE Are DONE!

            #We have not found the path yet
            visited.add((x, y)) #add current cell to visited set

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]: #up down, right left cells from current cell, iterate through each
                nx, ny = x + dx, y + dy #calculate the actual location of the cells

                if 0 <= nx < self.maze_bitmap.shape[0] and 0 <= ny < self.maze_bitmap.shape[1] and \
                   self.maze_bitmap[nx][ny] == 0 and (nx, ny) not in visited: #check if neighour is a valid cell (within bounds and not visited)

                    if pathfinding_function == self._bfs: #if using BFS
                        queue.append(((nx, ny), path + [(nx, ny)])) #add new cell to the queue
                    else: #if using A*
                        new_g_score = g_scores.get((x, y), 0) + 1 #add 1 to the current g_score (1 move) sets to 0 then adds 1 if does not exist
                        if (nx, ny) not in g_scores or new_g_score < g_scores[(nx, ny)]: #check existing g_score(if present) see if a lower score was found
                            g_scores[(nx, ny)] = new_g_score #if this is the best current path, update the g_Score
                            f_score = new_g_score + heuristic((nx, ny), end) #calulate new f_score (g_score + heuristic)
                            heapq.heappush(queue, (f_score, (nx, ny), path + [(nx, ny)])) #push to priority queue
                            
            #For each new item added to visited, we update the image
            temp_image = self.maze_image.copy().convert('RGB') #conver to RGB
            self.draw_path(temp_image, path=path, visited=visited) #Draw current unfinished path with the visited set to visualise
            self.display_image(temp_image) #update the displayed image
            
            self.update() #force tkinter to update image

    def find_path_image_bfs(self, img, start, end): #BFS solving without bitmap, get the image, start and end coords
        width, height = img.size #set width and weight of image
        start_pixel = (start[1], start[0]) #define the start and end pixel 
        end_pixel = (end[1], end[0])

        queue = deque([(start_pixel, [start_pixel])]) #dequeue queue, start pixel to start
        visited = set([start_pixel]) #visited set, start with start pixel
        parent = {} #parent dictionay

        while queue: #while there are still paths to explore
            current_pixel, path = queue.popleft() #take next pixel in list
            if current_pixel == end_pixel: #Have we found final path?
                return [(p[1], p[0]) for p in path] #if yes, reconstruct path to y, x format and return it
            #contunie solving
            x, y = current_pixel #unpack current coords
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] #define neighbours

            for neighbor_pixel in neighbors: #iterate through each neighbour
                nx, ny = neighbor_pixel #unpack into coords
                if 0 <= nx < width and 0 <= ny < height and img.getpixel(neighbor_pixel) != 0 and neighbor_pixel not in visited:  #check if neighbour is valid
                    queue.append((neighbor_pixel, path + [neighbor_pixel])) #if valid, enqueue neighbour with updated path
                    visited.add(neighbor_pixel) #add neighbour as visited
                    parent[neighbor_pixel] = current_pixel #set parent of neighbour as current
        return None #if no path is found

    def find_path_image_astar(self, img, start, end): #A* solving without bitmap
        def heuristic(a, b): #for calculating absolute distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1]) 

        width, height = img.size #get image dimensions
        start_pixel = (start[1], start[0]) #unpack to get coords
        end_pixel = (end[1], end[0])

        queue = [] #list for priority set with heapq
        heapq.heappush(queue, (0, start_pixel, [start_pixel])) #push starting pixel with f-score of 0
        visited = set([start_pixel]) #initalise visited set with start pixel to start
        g_score = {start_pixel: 0} #initalise g_Score dictionary with 0 as start

        while queue: #while there are pixels to explore
            f_score, current_pixel, path = heapq.heappop(queue) #pop pixel with lowest f-score from queue

            if current_pixel == end_pixel: #Have we found a path?
                return [(p[1], p[0]) for p in path] #convert to y, x and send back
            #continue solving
            x, y = current_pixel #unpack current coordinates
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)] #define neighbours

            for neighbor_pixel in neighbors: #iterate through each neighbour
                nx, ny = neighbor_pixel #unpack coordinates
                if 0 <= nx < width and 0 <= ny < height and img.getpixel(neighbor_pixel) != 0 and neighbor_pixel not in visited: #check if neighbour is valid
                    tentative_g_score = g_score[current_pixel] + 1 #add 1 to g_score as 1 step has been taken
                    if neighbor_pixel not in g_score or tentative_g_score < g_score[neighbor_pixel]: #check if a better path has been found before
                        g_score[neighbor_pixel] = tentative_g_score #if not, update g_score dictionary
                        f_score = tentative_g_score + heuristic(neighbor_pixel, end_pixel) #Calculate new f_score
                        heapq.heappush(queue, (f_score, neighbor_pixel, path + [neighbor_pixel])) #push neighbour to queue
                        visited.add(neighbor_pixel) #mark as visited
        return None


if __name__ == "__main__": #for implementation with the main menu
    app = MazeSolver()
    app.mainloop()