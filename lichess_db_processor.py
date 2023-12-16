import argparse
import zstandard as zstd
import requests
import io
import chess.pgn
import re
import struct
import sqlite3
from sqlite3 import Connection
import os
import sys
from datetime import datetime

REGEX_EXTRACT_EVAL = r'[%]eval\s[#]?([+-]?\d*\.?\d+|\d+)'

def create_db(sql_db_path:str) -> Connection:
    """
    Create and initialize a SQLite database for chess evaluations.

    Args:
        sql_db_path (str): Path to the SQLite database.

    Returns:
        Connection: SQLite database connection.
    """
    conn = sqlite3.connect(sql_db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY,
        fen TEXT,
        binary BLOB,
        evaluation NUMERIC
    );''')
    return conn


def enter_pgn_into_db(conn:Connection, data:list) -> None:
    """
    Insert PGN data into the SQLite database.

    Args:
        conn (Connection): SQLite database connection.
        data (list): List of tuples containing (fen, binary, evaluation).
    """
    cursor = conn.cursor()
    # Insert rows into the table
    cursor.executemany("INSERT INTO evaluations (fen, binary, evaluation) VALUES (?, ?, ?)", data)
    conn.commit()


def fen_to_bitboard(fen:str) -> bytes:
    """
    Convert a FEN string to a bitboard representation.

    Args:
        fen (str): FEN string.

    Returns:
        bytes: Bitboard representation.
    """
    # Create a chess.Board object from the FEN string
    board = chess.Board(fen)
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


def pgn_to_algebraic_notation(pgn_file:io.StringIO) -> list:
    """
    Extract algebraic notation and evaluation values from a PGN file.

    Args:
        pgn_file (io.StringIO): StringIO object containing PGN data.

    Returns:
        list: List of tuples containing (fen, binary, evaluation).
    """
    game = chess.pgn.read_game(pgn_file)
    
    board = game.board()
    data = []
    for move in game.mainline():
        e = re.search(REGEX_EXTRACT_EVAL, move.comment)
        if not e:
            board.push(move.move)
            continue
        ev= float(e.group(1))
        fen = board.fen()
        bitboard = fen_to_bitboard(fen)
        board.push(move.move)
        data.append((fen, bitboard, ev))
    return data


def parse_pgn_games(file_path) -> io.StringIO:
    """
    Parse PGN games from a file.

    Args:
        file_path (str): Path to the PGN file.

    Yields:
        io.StringIO: StringIO object containing PGN data.
    """
    games = 0
    with open(file_path) as pgn_file:
        game = []
        for line in pgn_file.readlines():
            if line.startswith('[Event'):
                games += 1
                if games % 1000 == 0:
                    print(f'Games parsed: {games}', end='\r')
                game.clear()
            game.append(line)
            if line.startswith('1. ') and r'%eval' in line:
                yield io.StringIO(''.join(game))


def decompress_zstd_file(compressed_file:bytes, output_file:str) -> None:
    """
    Decompress a Zstandard compressed file.

    Args:
        compressed_file (bytes): Compressed file content.
        output_file (str): Path to save the decompressed file.
    """
    print(f'Decompressing .zst file and saving to {output_file}')
    try:
        # with open(input_file, 'rb') as compressed_file:
        # Create a Zstandard decompressor
        decompressor = zstd.ZstdDecompressor()

        # Open the output file in binary write mode
        with open(output_file, 'wb') as decompressed_file:
            # Decompress the content of the input file and write it to the output file

            sr = decompressor.stream_reader(compressed_file)
            decompressed_content = sr.read()
            decompressed_file.write(decompressed_content)

        print("Decompression successful.")
    except Exception as e:
        print(f"Error during decompression: {e}")


def fetch_lichess_database(year:str, month:str) -> bytes:
    """
    Fetch a compressed Lichess database file.

    Args:
        year (str): Year in 20XX format from 2013 to 2023.
        month (str): Month in numerical format (e.g., 01, 02, ..., 12).

    Returns:
        bytes: Compressed database file content.
    """
    print(f'Downloading file for {month} {year}')
    url = f'https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month}.pgn.zst'
    response = requests.get(url, stream=True)
    response.raise_for_status()
    print('File Downloaded!')
    return response.content

# def decompress_zstd_stream(response:Response, output_file):
#     decompressor = zstd.ZstdDecompressor()
#     try:
#         with open(output_file, 'wb') as decompressed_file:
#             for chunk in response.iter_content():
#                 decompressed_chunk = decompressor.stream_reader(chunk)
#             decompressed_file.write(decompressed_chunk.read())

#         print("Download and decompression successful.")
#     except Exception as e:
#         print(f"Error during download and decompression: {e}")
# Example usage:


def month_to_number(month_name:str) -> str:
    """
    Convert month name or abbreviation to numerical format.

    Args:
        month_name (str): Month name or abbreviation.

    Returns:
        str: Numerical format of the month (e.g., 01, 02, ..., 12).
    """
    try:
        # Parse the input month using datetime
        parsed_month = datetime.strptime(month_name, "%B").strftime("%m")
        return parsed_month
    except ValueError:
        try:
            # Try parsing as an abbreviation
            parsed_month = datetime.strptime(month_name, "%b").strftime("%m")
            return parsed_month
        except ValueError:
            raise ValueError("Invalid month name. Please enter a valid month.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert month name to a 2-character number.')
    parser.add_argument('-m', dest='month', required=True, action='store', type=str, help='Month name or abbreviation (e.g., Jan, February, etc.)')
    parser.add_argument('-y',dest='year', required=True, action='store', type=str, help='Year in 20XX format from 2013 to 2023')

    args = parser.parse_args()

    month = month_to_number(args.month)
    pgn_file_path = os.path.join('data', 'PGN', f'lichess_db_standard_rated_{args.year}-{month}.pgn')
    sqlite_file_path = os.path.join('data', 'SQL', f'lichess_db_standard_rated_{args.year}-{month}.sqlite')

    if os.path.exists(pgn_file_path):
        print(f'Database for {args.month} {args.year}, already exists')
        sys.exit(0)

    content = fetch_lichess_database(args.year, month)
    decompress_zstd_file(content, pgn_file_path)

    conn = create_db(sqlite_file_path)
    print(f'Database created at {sqlite_file_path}')
    print('Parsing PGN file and creating bitboards to insert into database')
    parsed = 0
    boards = 0
    for pgn in parse_pgn_games(pgn_file_path):
        data = pgn_to_algebraic_notation(pgn)
        enter_pgn_into_db(conn, data)
        boards += len(data)
        parsed += 1
    conn.close()
    print(f'Parsed {parsed} PGN games with {boards} board layouts')

