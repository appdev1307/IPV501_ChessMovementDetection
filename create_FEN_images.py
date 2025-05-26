import cv2
import os

def fen_to_board(fen):
    board = []
    for row in fen.split()[0].split('/'):
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend([''] * int(char))
            else:
                board_row.append(char)
        board.append(board_row)
    return board

def split_board_into_squares(image):
    h, w = image.shape[:2]
    square_size = h // 8
    squares = []
    for i in range(8):
        for j in range(8):
            y1, y2 = i * square_size, (i + 1) * square_size
            x1, x2 = j * square_size, (j + 1) * square_size
            square = image[y1:y2, x1:x2]
            squares.append(square)
    return squares

def save_square(square, label, index, folder="templates"):
    os.makedirs(folder, exist_ok=True)
    resized = cv2.resize(square, (50, 50))
    filename = f"{label}_{index}.png"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, resized)
    print(f"Saved: {path}")

def extract_templates_auto(image_path, fen, save_dir="templates"):
    image = cv2.imread(image_path)
    board_labels = fen_to_board(fen)
    squares = split_board_into_squares(image)

    if len(squares) != 64 or len(board_labels) != 8 or any(len(row) != 8 for row in board_labels):
        raise ValueError("Board or FEN dimensions incorrect.")

    for i in range(8):
        for j in range(8):
            label = board_labels[i][j]
            if label:
                index = i * 8 + j
                save_square(squares[index], label, index, save_dir)

if __name__ == "__main__":
    # Example FEN for starting position:
    fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    image_file = "ChessBoard_Template.png"
    extract_templates_auto(image_file, fen_str)
