from puzzle_cube_3d import *

if __name__ == '__main__':
    print("\033[1;38;2;0;192;255mHOW TO USE:\n"
          "1. Camera Movement:\033[0m\n"
          "\t- use WASD to move forward, left, backward, and right, respectively\n"
          "\t- use 'Q' and 'E' to move up and down\n\n"
          "2. \033[1;38;2;0;192;255mPuzzle Cube Appearance:\033[0m\n"
          "\t- use arrow keys, forward slash (/) and RSHIFT or drag with the mouse to rotate the cube\n"
          "\t- use 1-5 to toggle the visibility of each piece\n"
          "\t- use 'SPACE' or 'X' to toggle exploded view\n"
          "\t- use 'O' to toggle cube outlines\n"
          "\t- use 'M' to toggle between perspective and orthographic projection\n\n"
          "3. \033[1;38;2;0;192;255mPuzzle Cube Generation:\033[0m\n"
          "\t- use 'R' to regenerate the puzzle cube\n"
          "\t- use 'P' to print out the cube's layers in the console\n"
          "\t\t- this can be used to load the cube in the future\n"
          "\t- use 'L' to load a previously generated cube\n"
          "\t\t- copy and paste the cube's layers from when you pressed 'P'")

    puzzle = PuzzleCube3D()
    puzzle.render()
