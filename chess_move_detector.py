import cv2
import numpy as np
import os
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
from skimage.metrics import structural_similarity as ssim

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Erosion kernel
EROSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

class ChessVideoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Video Analysis")
        self.running = False
        self.paused = False
        self.process_thread = None
        self.display_thread = None
        self.video_path = ""
        self.result_queue = queue.Queue()

        # GUI Elements
        self.create_gui()

    def create_gui(self):
        # Video selection
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(pady=5)
        tk.Button(self.video_frame, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=5)
        self.video_label = tk.Label(self.video_frame, text="No video selected", font=("Arial", 10))
        self.video_label.pack(side=tk.LEFT, padx=5)

        # Canvas for video display
        self.canvas = tk.Canvas(self.root, width=640, height=360, bg="black")
        self.canvas.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self.root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack(pady=5)

        # FEN and moves output
        self.fen_text = tk.Text(self.root, height=5, width=80, font=("Arial", 10))
        self.fen_text.pack(pady=5)

        # Control buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=10)
        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.pause_button = tk.Button(self.button_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def select_video(self):
        """Open a file dialog to select a video file."""
        self.video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.mkv *.avi")]
        )
        if self.video_path:
            self.video_label.config(text=os.path.basename(self.video_path))
            self.start_button.config(state=tk.NORMAL)
        else:
            self.video_label.config(text="No video selected")
            self.start_button.config(state=tk.DISABLED)

    def get_video_duration(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {self.video_path}")
            raise ValueError("Cannot open video file")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps <= 0:
            logging.error("Invalid FPS value in video")
            raise ValueError("Invalid FPS value")
        duration = frame_count / fps
        logging.info(f"Video duration: {duration:.2f} seconds")
        return duration

    def extract_frames_in_duration(self, start_time, end_time, frame_interval=1.0):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {self.video_path}")
            raise ValueError("Cannot open video file")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logging.error("Invalid FPS value in video")
            raise ValueError("Invalid FPS value")
        duration = self.get_video_duration()
        if start_time < 0 or end_time > duration or start_time >= end_time:
            logging.error(f"Invalid duration: start_time={start_time}s, end_time={end_time}s, video_duration={duration}s")
            raise ValueError(f"Invalid duration")
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_step = max(1, int(frame_interval * fps))
        frames = []
        frame_times = []
        for frame_num in range(start_frame, end_frame + 1, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to extract frame {frame_num}")
                continue
            frame_time = frame_num / fps
            frames.append(frame)
            frame_times.append(frame_time)
        cap.release()
        if not frames:
            logging.error(f"No frames extracted in the specified duration")
            raise ValueError("No frames extracted")
        return frames, frame_times, fps

    def compute_frame_difference(self, prev_frame, curr_frame):
        if prev_frame is None or curr_frame is None:
            return 0.0
        prev_resized = cv2.resize(prev_frame, (320, 180))
        curr_resized = cv2.resize(curr_frame, (320, 180))
        prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(prev_gray, curr_gray, full=True)
        logging.info(f"SSIM score between frames: {score:.4f}")
        return score

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def extract_digital_board(self, image):
        h, w = image.shape[:2]
        crop = image[0:h, 0:w]
        if crop.size == 0:
            raise ValueError("Cropped image is empty")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) >= 4 and cv2.contourArea(c) > 5000:
                points = approx.reshape(-1, 2)
                return crop, points
        return crop, None

    def load_templates(self, template_dir="templates"):
        pieces = ["P", "N", "B", "R", "Q", "K", "pb", "nb", "bb", "rb", "qb", "kb"]
        templates = {}
        if not os.path.exists(template_dir):
            logging.error(f"Template directory not found: {template_dir}")
            raise ValueError(f"Template directory not found")
        for p in pieces:
            path = os.path.join(template_dir, f"{p}.png")
            if not os.path.exists(path):
                logging.error(f"Template for {p} not found at {path}")
                raise ValueError(f"Template for {p} not found")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            resized = cv2.resize(img, (80, 80))
            eroded = cv2.erode(resized, EROSION_KERNEL, iterations=1)
            templates[p] = eroded
        return templates

    def match_piece(self, square_img, img_name, templates, frame_idx, threshold=0.6):
        if square_img.size == 0 or square_img.shape[0] == 0 or square_img.shape[1] == 0:
            logging.warning(f"Empty square image: {img_name}")
            return None
        square_gray = square_img if len(square_img.shape) == 2 else cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        square_resized = cv2.resize(square_gray, (80, 80))
        square_resized = cv2.erode(square_resized, EROSION_KERNEL, iterations=1)
        max_val = 0
        best_match = None
        for piece, template in templates.items():
            try:
                res = cv2.matchTemplate(square_resized, template, cv2.TM_CCOEFF_NORMED)
                _, val, _, _ = cv2.minMaxLoc(res)
                if val > max_val:
                    max_val = val
                    best_match = piece
            except cv2.error as e:
                logging.error(f"matchTemplate error for {piece} in {img_name}: {e}")
                continue
        return best_match if max_val >= threshold else None

    def warp_board(self, crop, points):
        rect = self.order_points(points)
        dst = np.array([[0, 0], [551, 0], [551, 551], [0, 551]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(crop, M, (552, 552))
        if warped.size == 0 or warped.shape[0] == 0 or warped.shape[1] == 0:
            raise ValueError("Warped image is empty")
        return warped, M

    def split_into_squares(self, board_img):
        squares = []
        square_names = []
        height, width = board_img.shape[:2]
        dy, dx = 69, 69
        if height < 8 * dy or width < 8 * dx:
            top_pad = max(0, 8 * dy - height)
            left_pad = max(0, 8 * dx - width)
            board_img = cv2.copyMakeBorder(board_img, 0, top_pad, 0, left_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        for row in range(8):
            for col in range(8):
                y_start, y_end = row * dy, (row + 1) * dy
                x_start, x_end = col * dx, (col + 1) * dx
                square = board_img[y_start:y_end, x_start:x_end]
                if square.size == 0 or square.shape[0] < dy or square.shape[1] < dx:
                    continue
                gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if len(square.shape) == 3 else square
                resized = cv2.resize(gray, (68, 68))
                name = f'square_r{row}_c{col}.png'
                squares.append(resized)
                square_names.append(name)
        return squares, square_names

    def generate_fen(self, squares, square_names, templates, frame_idx):
        board = [['' for _ in range(8)] for _ in range(8)]
        fen_map = {'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
                   'pb': 'p', 'nb': 'n', 'bb': 'b', 'rb': 'r', 'qb': 'q', 'kb': 'k'}
        for i, (square, name) in enumerate(zip(squares, square_names)):
            row, col = i // 8, i % 8
            piece = self.match_piece(square, name, templates, frame_idx)
            board[row][col] = fen_map.get(piece, '') if piece else ''
        fen_rows = []
        for row in board:
            empty = 0
            fen_row = ''
            for square in row:
                if square == '':
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += square
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        fen = '/'.join(fen_rows) + ' w KQkq - 0 1'
        return fen, board

    def detect_movement(self, prev_board, curr_board):
        moves = []
        for row in range(8):
            for col in range(8):
                prev_piece = prev_board[row][col]
                curr_piece = curr_board[row][col]
                if prev_piece != curr_piece:
                    if prev_piece == '' and curr_piece != '':
                        to_square = f"{chr(97 + col)}{8 - row}"
                        for r in range(8):
                            for c in range(8):
                                if prev_board[r][c] == curr_piece and curr_board[r][c] == '':
                                    from_square = f"{chr(97 + c)}{8 - r}"
                                    moves.append(f"{curr_piece}{from_square}-{to_square}")
                                    break
                    elif prev_piece != '' and curr_piece == '':
                        continue
                    elif prev_piece != '' and curr_piece != '':
                        to_square = f"{chr(97 + col)}{8 - row}"
                        moves.append(f"{prev_piece}{to_square}->{curr_piece}{to_square}")
        return moves

    def annotate_frame(self, frame, moves, frame_time, points, M):
        annotated = frame.copy()
        for i in range(4):
            pt1 = tuple(points[i].astype(int))
            pt2 = tuple(points[(i + 1) % 4].astype(int))
            cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
        y_offset = 50
        cv2.putText(annotated, f"Time: {frame_time:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        for i, move in enumerate(moves):
            cv2.putText(annotated, f"Move: {move}", (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated

    def update_canvas(self, frame):
        max_width, max_height = 640, 360
        h, w = frame.shape[:2]
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def start_processing(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file")
            return
        try:
            self.get_video_duration()
            self.running = True
            self.paused = False
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Processing")
            self.fen_text.delete(1.0, tk.END)
            self.result_queue = queue.Queue()  # Reset queue
            self.process_thread = threading.Thread(target=self.process_frames)
            self.display_thread = threading.Thread(target=self.display_frames)
            self.process_thread.start()
            self.display_thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start processing: {e}")
            self.status_label.config(text="Status: Error")

    def toggle_pause(self):
        if self.running:
            self.paused = not self.paused
            self.pause_button.config(text="Resume" if self.paused else "Pause")
            self.status_label.config(text=f"Status: {'Paused' if self.paused else 'Processing'}")

    def stop_processing(self):
        if self.running:
            self.running = False
            self.paused = False
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED, text="Pause")
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped")
            if self.process_thread:
                self.process_thread.join()
            if self.display_thread:
                self.display_thread.join()

    def process_frames(self):
        try:
            start_time, end_time, frame_interval = 300, 450, 1.0
            duration = self.get_video_duration()
            if end_time > duration:
                end_time = duration
            frames, frame_times, fps = self.extract_frames_in_duration(start_time, end_time, frame_interval)
            templates = self.load_templates()
            prev_board = None
            prev_frame = None
            total_frames = len(frames)
            ssim_threshold = 0.95

            for i, (frame, frame_time) in enumerate(zip(frames, frame_times)):
                if not self.running:
                    break
                while self.paused:
                    if not self.running:
                        break
                try:
                    ssim_score = self.compute_frame_difference(prev_frame, frame)
                    process_frame = (prev_frame is None or ssim_score < ssim_threshold)
                    logging.info(f"Frame {i} at {frame_time:.2f}s: SSIM={ssim_score:.4f}, Process={process_frame}")
                    
                    if process_frame:
                        points = np.array([[1280, 240], [1879, 240], [1879, 836], [1280, 836]], dtype="float32")
                        crop = frame
                        board, M = self.warp_board(crop, points)
                        squares, square_names = self.split_into_squares(board)
                        fen, curr_board = self.generate_fen(squares, square_names, templates, i)
                        moves = self.detect_movement(prev_board, curr_board) if prev_board is not None else []
                        annotated_frame = self.annotate_frame(frame, moves, frame_time, points, M)
                        self.result_queue.put((annotated_frame, frame_time, fen, moves, i, total_frames))
                        prev_board = curr_board
                    else:
                        annotated_frame = self.annotate_frame(frame, [], frame_time, points, M)
                        self.result_queue.put((annotated_frame, frame_time, None, [], i, total_frames))
                    prev_frame = frame
                except Exception as e:
                    logging.error(f"Frame error at {frame_time:.2f}s: {e}")
                    self.result_queue.put((frame, frame_time, None, [], i, total_frames))
                    continue
            self.result_queue.put(None)  # Signal end of processing
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing error: {e}"))
            self.running = False

    def display_frames(self):
        try:
            output_path = "annotated_chess_moves.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            first_frame = None
            while first_frame is None and self.running:
                try:
                    item = self.result_queue.get(timeout=1)
                    if item is None:
                        break
                    first_frame = item[0]
                except queue.Empty:
                    continue
            if first_frame is None:
                return
            height, width = first_frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))  # Assume 30 FPS for output
            fen_results = []

            while self.running:
                try:
                    item = self.result_queue.get(timeout=1)
                    if item is None:
                        break
                    annotated_frame, frame_time, fen, moves, frame_idx, total_frames = item
                    self.update_canvas(annotated_frame)
                    out.write(annotated_frame)
                    if fen is not None:
                        fen_results.append((frame_time, fen, moves))
                    self.fen_text.delete(1.0, tk.END)
                    for ft, fen, mv in fen_results:
                        self.fen_text.insert(tk.END, f"Time {ft:.2f}s: {fen}\n")
                        if mv:
                            self.fen_text.insert(tk.END, f"  Moves: {', '.join(mv)}\n")
                    self.progress["value"] = (frame_idx + 1) / total_frames * 100
                    self.status_label.config(text=f"Status: Processing frame {frame_idx + 1}/{total_frames}")
                    self.root.update()
                except queue.Empty:
                    if not self.running:
                        break
                    continue
            out.release()
            if self.running:
                self.root.after(0, lambda: self.status_label.config(text="Status: Processing Complete"))
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.pause_button.config(state=tk.DISABLED))
                self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
                self.running = False
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Display error: {e}"))
            self.running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessVideoGUI(root)
    root.mainloop()
