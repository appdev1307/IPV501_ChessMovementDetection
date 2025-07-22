import cv2
import os
import numpy as np

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

def crop_and_resize_board(image, crop_margin_x=0, crop_margin_y=0, output_size=552):
    h, w = image.shape[:2]

    x1 = crop_margin_x
    x2 = w - crop_margin_x
    y1 = crop_margin_y
    y2 = h - crop_margin_y

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Cropping margins too large for image size.")

    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, (output_size, output_size))
    return resized

def split_board_into_squares(image):
    square_size = 69
    squares = []
    for i in range(8):
        for j in range(8):
            y1 = i * square_size
            y2 = (i + 1) * square_size
            x1 = j * square_size
            x2 = (j + 1) * square_size
            square = image[y1:y2, x1:x2]
            squares.append(square)
    return squares

def save_square(square, label, index, folder="templates"):
    os.makedirs(folder, exist_ok=True)

    # Resize for consistency
    resized = cv2.resize(square, (68, 68))

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply a fixed threshold to create a binary mask
    # Adjust threshold (e.g., 200) based on the piece's contrast with the background
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Create 3-channel mask for color images
    mask_3ch = cv2.merge([mask] * 3)

    # Create white background
    background = 255 * np.ones_like(resized)  # White background (255,255,255)

    # Combine piece and background using mask
    final = cv2.bitwise_and(resized, mask_3ch)
    final = cv2.add(final, cv2.bitwise_and(background, cv2.bitwise_not(mask_3ch)))

    # Save to file
    filename = f"{label}_{index}.png"
    path = os.path.join(folder, filename)
    cv2.imwrite(path, final)
    print(f"Saved: {path}")

def extract_templates_auto(image_path, fen, save_dir="templates", crop_margin_x=0, crop_margin_y=0):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    board_img = crop_and_resize_board(image, crop_margin_x=crop_margin_x, crop_margin_y=crop_margin_y)

    board_labels = fen_to_board(fen)
    squares = split_board_into_squares(board_img)

    if len(squares) != 64 or len(board_labels) != 8 or any(len(row) != 8 for row in board_labels):
        raise ValueError("Board or FEN dimensions incorrect.")

    for i in range(8):
        for j in range(8):
            label = board_labels[i][j]
            if label:
                index = i * 8 + j
                save_square(squares[index], label, index, save_dir)

if __name__ == "__main__":
    # Example FEN for starting chess position
    fen_str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Your board image
    image_file = "ChessBoard_Template.png"
    
    # Adjust margin (crop from all 4 sides evenly)
    crop_margin_x = 10  # Left and right
    crop_margin_y = 10  # Top and bottom
    
    # Extract and save templates
    extract_templates_auto(image_file, fen_str, save_dir="templates", crop_margin_x=crop_margin_x, crop_margin_y=crop_margin_y)
