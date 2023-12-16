import chess
import chess.svg
import random
import struct
import torch
import numpy
from lichess_model import EvaluationModel


def board_to_bitboard(board) -> bytes:
    # Create a chess.Board object from the FEN string
    castle = board.has_queenside_castling_rights(chess.BLACK) << 3 | board.has_kingside_castling_rights(chess.WHITE) << 2 | board.has_queenside_castling_rights(chess.BLACK) << 1 | board.has_kingside_castling_rights(chess.WHITE)
    check = board.is_check() << 2 | board.was_into_check() << 1 | board.is_checkmate()
    # Get the bitboard representation for each piece
    bitboard = struct.pack('QQQQQQQQQQQQHBB', 
        board.pieces_mask(chess.KING, chess.WHITE),
        board.pieces_mask(chess.QUEEN, chess.WHITE),
        board.pieces_mask(chess.ROOK, chess.WHITE),
        board.pieces_mask(chess.KNIGHT, chess.WHITE),
        board.pieces_mask(chess.BISHOP, chess.WHITE),
        board.pieces_mask(chess.PAWN, chess.WHITE),
        board.pieces_mask(chess.KING, chess.BLACK),
        board.pieces_mask(chess.QUEEN, chess.BLACK),
        board.pieces_mask(chess.ROOK, chess.BLACK),
        board.pieces_mask(chess.KNIGHT, chess.BLACK),
        board.pieces_mask(chess.BISHOP, chess.BLACK),
        board.pieces_mask(chess.PAWN, chess.BLACK),
        board.fullmove_number,
        (board.turn << 4) | castle,
        (board.has_legal_en_passant() << 3) | check
	)

    return bitboard

def play_random_move(board):
    legal_moves = [move for move in board.legal_moves]
    if legal_moves:
        return random.choice(legal_moves)
    return None


def play_best_eval_move(board, model):
    legal_moves = [move for move in board.legal_moves]
    if legal_moves:
        best_eval = 50.0
        best_move = legal_moves[0]
        for move in legal_moves:
            board_copy = board.copy()
            board_copy.push(move)
            bit_board = board_to_bitboard(board_copy)
            bits = numpy.unpackbits(numpy.frombuffer(bit_board, dtype=numpy.uint8))
            bits = numpy.array([bit.astype(numpy.float32) for bit in bits])
            x = torch.tensor(bits)
            eval_score = model(x).item()
            if eval_score < best_eval:
                best_move = move
                best_eval = eval_score
        return best_move
    return None


def play_chess():
    checkpoint_file_path = 'data/MODEL/lichess-model-7.ckpt'
    model = EvaluationModel.load_from_checkpoint(checkpoint_path=checkpoint_file_path)
    wins = {'AI 1': 0, 'AI 2': 0}
    
    for _ in range(100):  # You can adjust the number of games played
        board = chess.Board()

        while not board.is_game_over():

            # AI 1 plays
            #move_ai1 = play_random_move(board)
            move_ai1 = play_best_eval_move(board, model)
            if move_ai1:
                board.push(move_ai1)

            # Check for checkmate or stalemate
            if board.is_check() or board.is_stalemate():
                print('AI-2 In Check')
                wins['AI 1'] += 1
                break

            # AI 2 plays
            move_ai2 = play_random_move(board)
            if move_ai2:
                board.push(move_ai2)

            # Check for checkmate or stalemate
            if board.is_check() or board.is_stalemate():
                print('AI-1 In Check')
                wins['AI 2'] += 1
                break

        # Determine the winner
        # result = board.result()
        # if result == '1-0':
        #     wins['AI 1'] += 1
        # elif result == '0-1':
        #     wins['AI 2'] += 1

        # print(f"Game Result: {result}")
        print("Wins:", wins)
        print("="*20)

if __name__ == "__main__":
    play_chess()
