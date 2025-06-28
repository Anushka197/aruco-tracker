import cv2
import numpy as np

def generate_chessboard(columns=9, rows=6, square_size_mm=20, output_file="chessboard.png"):
    width = columns * square_size_mm
    height = rows * square_size_mm

    # Size in pixels (10 pixels per mm for high-res print)
    scale = 10
    img_width = width * scale
    img_height = height * scale
    square_size_px = square_size_mm * scale

    # Create white image
    board = 255 * np.ones((int(img_height), int(img_width)), dtype=np.uint8)

    # Draw black squares
    for row in range(rows):
        for col in range(columns):
            if (row + col) % 2 == 0:
                x0 = int(col * square_size_px)
                y0 = int(row * square_size_px)
                x1 = int((col + 1) * square_size_px)
                y1 = int((row + 1) * square_size_px)
                board[y0:y1, x0:x1] = 0

    cv2.imwrite(output_file, board)
    print(f"Chessboard saved to: {output_file}")

generate_chessboard(columns=9, rows=6, square_size_mm=20)
