import curses

def main():
  # Initialize curses screen
  screen = curses.initscr()
  curses.cbreak()  # Get chars without waiting for enter

  # Display prompt for user input
  screen.addstr(1, 1, "Enter your name: ")

  # Get user input using getstr
  user_name = screen.getstr()

  # Print the entered name
  screen.addstr(2, 1, f"Hello, {user_name}!")
  screen.getch()  # Wait for a key press to exit

  # Clean up
  curses.endwin()

if __name__ == "__main__":
  main()
