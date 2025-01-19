import tkinter as tk
from tkinter import messagebox
import random

class WordleGame:
    def __init__(self):
        with open("words.txt") as f:
            self.valid_words = set(word.strip().upper() for word in f.readlines())
        
        self.target_word = self.choose_random_word()
        self.num_guesses = 0
        print(self.target_word)
    
    def choose_random_word(self):
        return random.choice(list(self.valid_words))
    
    def check_guess(self, guess):
        guess = guess.upper()
        if guess not in self.valid_words:
            return None
        
        self.num_guesses += 1

        result = []
        target_letters = list(self.target_word)

        #Check for correct letters(Green)
        for i, (guess_letter, target_letter) in enumerate(zip(guess, self.target_word)):
            if guess_letter == target_letter:
                result.append(("correct", guess_letter))
                target_letters[i] = None
            else:
                result.append((None, guess_letter))
        
        #Check for present letters(Yellow)
        for i, (status, guess_letter) in enumerate(result):
            if status is None:
                if guess_letter in target_letters:
                    result[i] = ("present", guess_letter)
                    target_letters[target_letters.index(guess_letter)] = None
                else:
                    result[i] = ("absent", guess_letter)
        
        return result

class WordleUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Wordle')
        self.root.configure(bg='#121213')

        self.game = WordleGame()

        #Constants
        self.WORD_LENGTH = 5
        self.MAX_ATTEMPTS = 6
        self.current_row = 0
        self.current_col = 0
        self.current_guess = []
        self.game_over = False

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

        #New Game Button
        self.new_game_btn = tk.Button(
            self.root,
            text="New Game",
            command=self.new_game,
            font=("Arial", 13),
            bg="#818384",
            fg="white"
        )
        self.new_game_btn.pack(pady=20)

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
        if self.game_over:
            return

        if len(self.current_guess) == self.WORD_LENGTH:
            guess = ''.join(self.current_guess)
            result = self.game.check_guess(guess)

            if result is None:
                messagebox.showerror("Invalid Guess", "Please enter a valid word.")
                return

            self.update_display(result)

            if all(status == 'correct' for status, _ in result):
                messagebox.showinfo("Congratulations!", 
                    f"You won in {self.game.num_guesses} {'guess' if self.game.num_guesses == 1 else 'guesses'}!\n")
                self.game_over = True
            elif self.current_row >= self.MAX_ATTEMPTS - 1:
                messagebox.showwarning("Game Over", f"The word was {self.game.target_word}")
                self.game_over = True
            else:
                self.current_row += 1
                self.current_col = 0
                self.current_guess = []
    
    def update_display(self, result):
        for i, (status, letter) in enumerate(result):
            color = {
                'correct': self.CORRECT,
                'present': self.PRESENT,
                'absent': self.ABSENT
            }[status]

            self.boxes[self.current_row][i].config(bg=color)

            current_btn_color = self.key_buttons[letter].cget("bg")
            if current_btn_color != self.CORRECT:
                if status == 'correct' or (status == 'present' and current_btn_color != self.PRESENT):
                    self.key_buttons[letter].config(bg=color)
                elif status == 'absent' and current_btn_color not in (self.CORRECT, self.PRESENT):
                    self.key_buttons[letter].config(bg=color)


    def backspace(self, event):
        if self.current_col > 0:
            self.current_col -= 1
            self.current_guess.pop()
            self.boxes[self.current_row][self.current_col].config(text="")
    
    def new_game(self):
        self.game = WordleGame()
        self.current_row = 0
        self.current_col = 0
        self.current_guess = []
        self.game_over = False

        for row in self.boxes:
            for box in row:
                box.config(text="", bg=self.DEFAULT)
        
        for letter, btn in self.key_buttons.items():
            btn.config(bg="#818384")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = WordleUI(root)
    root.mainloop()