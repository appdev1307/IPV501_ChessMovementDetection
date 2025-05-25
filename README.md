# Chess Movement Detection

This project uses computer vision to detect chess piece movements from a video feed. It can track the board and detect when pieces are moved, recording the moves in chess notation.

## Setup

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Project

1. Run the main script:

   ```
   python chess_detection.py
   ```

2. The program will:

   - Open your webcam (or specified video source)
   - Look for a 7x7 chessboard pattern
   - Track the board and detect piece movements
   - Display the processed board view
   - Print detected moves in chess notation

3. To quit the program, press 'q' while the window is active

## Notes

- The program requires a clear view of the chessboard
- The chessboard should be well-lit and clearly visible
- The program looks for a 7x7 chessboard pattern (8x8 squares)
- Make sure the entire board is visible in the camera frame
