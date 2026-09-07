import sys

from .app import SlidingTilePuzzleApp

if __name__ == '__main__':
    app = SlidingTilePuzzleApp(sys.argv)
    sys.exit(app.exec())
