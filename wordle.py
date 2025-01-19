import tkinter as tk
from tkinter import messagebox

class Wordle:
    def __init__(self, root):
        self.root = root
        self.root.title('Wordle')
        self.root.configure(bg='#121213')

        #Constants
        self.WORD_LENGTH = 5
        self.MAX_ATTEMPTS = 6
        self.current_row = 0
        self.current_col = 0
        self.current_guess = []

        #Colors
        self.CORRECT = '#538D4E'
        self.PRESENT = '#B59F3B'
        self.ABSENT = '#3A3A3C'
        self.DEFAULT = '#121213'
        self.BORDER = '#3A3A3C'

        self.setup_ui()

    def setup_ui(self):
        #Main Frame Grid
        self.grid_frame = tk.Frame(self.root, bg=self.DEFAULT)
        self.grid_frame.pack(pady=20)

        #Letter Entry Frames
        self.boxes = []
        for i in range(self.MAX_ATTEMPTS):
            row = []
            for j in range(self.WORD_LENGTH):
                box = tk.Label(
                    self.grid_frame,
                    width=2,
                    height=1,
                    text="",
                    font=("Arial", 24, "bold"),
                    bg=self.DEFAULT,
                    fg="white",
                    relief="solid",
                    borderwidth=2
                )
                box.grid(row=i, column=j, padx=5, pady=5)
                row.append(box)
            self.boxes.append(row)
        
        #Keyboard Frame
        self.create_keyboard()

        #Key Bindings
        self.root.bind("<Key>", self.key_press)
        self.root.bind("<Return>", self.submit_guess)
        self.root.bind("<BackSpace>", self.backspace)
    
    def create_keyboard(self):
        keyboard_layout = [
            "QWERTYUIOP",
            "ASDFGHJKL",
            "ZXCVBNM"
        ]

        self.key_buttons = {}
        keyboard_frame = tk.Frame(self.root, bg=self.DEFAULT)
        keyboard_frame.pack(pady=20)

        for i, row in enumerate(keyboard_layout):
            row_frame = tk.Frame(keyboard_frame, bg=self.DEFAULT)
            row_frame.pack(pady=2)

            for letter in row:
                btn = tk.Button(
                    row_frame,
                    text=letter,
                    width=4,
                    height=2,
                    font=("Arial", 10, "bold"),
                    bg="#818384",
                    fg="white",
                    command=lambda l=letter: self.key_press({"char": l})
                )
                btn.pack(side=tk.LEFT, padx=2)
                self.key_buttons[letter] = btn

    def key_press(self, event):
        if not isinstance(event, dict):
            char = event.char.upper()
        else:
            char = event["char"].upper()
        
        if char.isalpha() and len(self.current_guess) < self.WORD_LENGTH:
            self.current_guess.append(char)
            self.boxes[self.current_row][self.current_col].config(text=char)
            self.current_col += 1
        
    def submit_guess(self, event):
        if len(self.current_guess) == self.WORD_LENGTH:
            self.simulate_feedback()

            self.current_row += 1
            self.current_col = 0
            self.current_guess = []

            if self.current_row >= self.MAX_ATTEMPTS:
                messagebox.showinfo("Game Over", "You have run out of attempts!")

    def backspace(self, event):
        if self.current_col > 0:
            self.current_col -= 1
            self.current_guess.pop()
            self.boxes[self.current_row][self.current_col].config(text="")
    
    def simulate_feedback(self):
        import random
        colors = [self.CORRECT, self.PRESENT, self.ABSENT]
        for i in range(self.WORD_LENGTH):
            color = random.choice(colors)
            self.boxes[self.current_row][i].config(bg=color)
            letter = self.current_guess[i]
            if letter in self.key_buttons:
                self.key_buttons[letter].config(bg=color)
        
if __name__ == "__main__":
    root = tk.Tk()
    app = Wordle(root)
    root.mainloop()