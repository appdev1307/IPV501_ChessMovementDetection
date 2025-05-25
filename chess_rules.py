import logging

logger = logging.getLogger(__name__)

class ChessMove:
    def __init__(self, piece_type, color, from_square, to_square):
        self.piece_type = piece_type  # 'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'
        self.color = color  # 'white' or 'black'
        self.from_square = from_square  # e.g., 'e2'
        self.to_square = to_square  # e.g., 'e4'
        self.is_valid = True
        self.warning = None
        logger.debug(f"Created new move: {color} {piece_type} from {from_square} to {to_square}")

class ChessRules:
    def __init__(self):
        logger.info("Initializing chess rules")
        self.initial_position = self._create_initial_position()
        self.current_position = self.initial_position.copy()
        self.move_history = []
        logger.info("Chess board initialized with starting position")

    def _create_initial_position(self):
        logger.debug("Creating initial board position")
        # Create initial board position
        position = {}
        # Set up pawns
        for file in 'abcdefgh':
            position[f'{file}2'] = {'piece': 'pawn', 'color': 'white'}
            position[f'{file}7'] = {'piece': 'pawn', 'color': 'black'}
        
        # Set up other pieces
        pieces = {
            'a': 'rook', 'b': 'knight', 'c': 'bishop', 'd': 'queen',
            'e': 'king', 'f': 'bishop', 'g': 'knight', 'h': 'rook'
        }
        
        for file, piece in pieces.items():
            position[f'{file}1'] = {'piece': piece, 'color': 'white'}
            position[f'{file}8'] = {'piece': piece, 'color': 'black'}
        
        logger.debug("Initial position created successfully")
        return position

    def validate_move(self, move):
        """Validate if a move is legal according to chess rules."""
        logger.debug(f"Validating move: {move.color} {move.piece_type} from {move.from_square} to {move.to_square}")
        
        if move.from_square not in self.current_position:
            move.is_valid = False
            move.warning = f"No piece at {move.from_square}"
            logger.warning(f"Invalid move: {move.warning}")
            return False

        piece = self.current_position[move.from_square]
        if piece['color'] != move.color:
            move.is_valid = False
            move.warning = f"Wrong color piece at {move.from_square}"
            logger.warning(f"Invalid move: {move.warning}")
            return False

        if piece['piece'] != move.piece_type:
            move.is_valid = False
            move.warning = f"Wrong piece type at {move.from_square}"
            logger.warning(f"Invalid move: {move.warning}")
            return False

        # Basic move validation based on piece type
        if not self._is_valid_piece_move(move):
            move.is_valid = False
            move.warning = f"Invalid move for {move.piece_type}"
            logger.warning(f"Invalid move: {move.warning}")
            return False

        logger.info(f"Move validated successfully: {move.color} {move.piece_type} from {move.from_square} to {move.to_square}")
        return True

    def _is_valid_piece_move(self, move):
        """Check if the move is valid for the specific piece type."""
        logger.debug(f"Checking piece-specific move validity for {move.piece_type}")
        from_file, from_rank = ord(move.from_square[0]) - ord('a'), int(move.from_square[1])
        to_file, to_rank = ord(move.to_square[0]) - ord('a'), int(move.to_square[1])
        
        file_diff = abs(to_file - from_file)
        rank_diff = abs(to_rank - from_rank)

        is_valid = False
        if move.piece_type == 'pawn':
            # Pawns move forward one square, or two on first move
            if move.color == 'white':
                if from_rank == 2:  # First move
                    is_valid = (file_diff == 0 and to_rank - from_rank in [1, 2])
                else:
                    is_valid = (file_diff == 0 and to_rank - from_rank == 1)
            else:  # black
                if from_rank == 7:  # First move
                    is_valid = (file_diff == 0 and from_rank - to_rank in [1, 2])
                else:
                    is_valid = (file_diff == 0 and from_rank - to_rank == 1)

        elif move.piece_type == 'knight':
            is_valid = (file_diff == 2 and rank_diff == 1) or (file_diff == 1 and rank_diff == 2)

        elif move.piece_type == 'bishop':
            is_valid = file_diff == rank_diff

        elif move.piece_type == 'rook':
            is_valid = file_diff == 0 or rank_diff == 0

        elif move.piece_type == 'queen':
            is_valid = file_diff == rank_diff or file_diff == 0 or rank_diff == 0

        elif move.piece_type == 'king':
            is_valid = file_diff <= 1 and rank_diff <= 1

        logger.debug(f"Piece move validation result: {is_valid}")
        return is_valid

    def make_move(self, move):
        """Make a move and update the board position."""
        logger.info(f"Attempting to make move: {move.color} {move.piece_type} from {move.from_square} to {move.to_square}")
        if self.validate_move(move):
            # Remove piece from original square
            piece = self.current_position.pop(move.from_square)
            # Place piece on new square
            self.current_position[move.to_square] = piece
            self.move_history.append(move)
            logger.info(f"Move made successfully: {move.color} {move.piece_type} from {move.from_square} to {move.to_square}")
            return True
        logger.warning(f"Move not made: {move.warning}")
        return False

    def get_move_history(self):
        """Return the list of moves made in the game."""
        logger.debug(f"Retrieving move history. Total moves: {len(self.move_history)}")
        return self.move_history 