import os
from tkinter import *
from PIL import ImageTk, Image

# obtain current file path
FILE_PATH = os.path.dirname(os.path.abspath(__file__))

from .carcassonne_game_state import CarcassonneGameState
from .objects.meeple_position import MeeplePosition
from .objects.meeple_type import MeepleType
from .objects.side import Side
from .objects.tile import Tile


class CarcassonneVisualiser:

    meeple_icons = {
        MeepleType.NORMAL: ["blue_meeple.png", "red_meeple.png", "black_meeple.png", "yellow_meeple.png", "green_meeple.png", "pink_meeple.png"],
        MeepleType.ABBOT: ["blue_abbot.png", "red_abbot.png", "black_abbot.png", "yellow_abbot.png", "green_abbot.png", "pink_abbot.png"]
    }
    tile_size = 60
    meeple_size = 15
    big_meeple_size = 25

    meeple_position_offsets = {
        Side.TOP: (tile_size / 2, (meeple_size / 2) + 3),
        Side.RIGHT: (tile_size - (meeple_size / 2) - 3, tile_size / 2),
        Side.BOTTOM: (tile_size / 2, tile_size - (meeple_size / 2) - 3),
        Side.LEFT: ((meeple_size / 2) + 3, tile_size / 2),
        Side.CENTER: (tile_size / 2, tile_size / 2),
        Side.TOP_LEFT: (tile_size / 4, (meeple_size / 2) + 3),
        Side.TOP_RIGHT: ((tile_size / 4) * 3, (meeple_size / 2) + 3),
        Side.BOTTOM_LEFT: (tile_size / 4, tile_size - (meeple_size / 2) - 3),
        Side.BOTTOM_RIGHT: ((tile_size / 4) * 3, tile_size - (meeple_size / 2) - 3)
    }

    big_meeple_position_offsets = {
        Side.TOP: (tile_size / 2, (big_meeple_size / 2) + 3),
        Side.RIGHT: (tile_size - (big_meeple_size / 2) - 3, tile_size / 2),
        Side.BOTTOM: (tile_size / 2, tile_size - (big_meeple_size / 2) - 3),
        Side.LEFT: ((big_meeple_size / 2) + 3, tile_size / 2),
        Side.CENTER: (tile_size / 2, tile_size / 2),
        Side.TOP_LEFT: (tile_size / 4, (big_meeple_size / 2) + 3),
        Side.TOP_RIGHT: ((tile_size / 4) * 3, (big_meeple_size / 2) + 3),
        Side.BOTTOM_LEFT: (tile_size / 4, tile_size - (big_meeple_size / 2) - 3),
        Side.BOTTOM_RIGHT: ((tile_size / 4) * 3, tile_size - (big_meeple_size / 2) - 3)
    }

    def __init__(self):
        pass
        self.initialized = False

    def _initialize(self):
        if self.initialized:
            return

        self.initialized = True
        root = Tk()
        self.canvas = Canvas(root, width=2300, height=1300, bg='white')
        self.canvas.pack(fill='both', expand=True)
        self.images_path = os.path.join(FILE_PATH, 'resources', 'images')
        self.meeple_image_refs = {}
        self.tile_image_refs = {}

    def draw_game_state(self, game_state: CarcassonneGameState):
        self._initialize()
        self.canvas.delete('all')
        for row_index, row in enumerate(game_state.board):
            for column_index, tile in enumerate(row):
                tile: Tile
                if tile is not None:
                    self.__draw_tile(column_index, row_index, tile)

        for player, placed_meeples in enumerate(game_state.placed_meeples):
            meeple_position: MeeplePosition
            for meeple_position in placed_meeples:
                self.__draw_meeple(player, meeple_position)

        self.canvas.update()

    def __draw_meeple(self, player_index: int, meeple_position: MeeplePosition):
        image = self.__get_meeple_image(player=player_index, meeple_type=meeple_position.meeple_type)

        if meeple_position.meeple_type == MeepleType.BIG:
            x = meeple_position.coordinate_with_side.coordinate.column * self.tile_size + self.big_meeple_position_offsets[meeple_position.coordinate_with_side.side][0]
            y = meeple_position.coordinate_with_side.coordinate.row * self.tile_size + self.big_meeple_position_offsets[meeple_position.coordinate_with_side.side][1]
        else:
            x = meeple_position.coordinate_with_side.coordinate.column * self.tile_size + self.meeple_position_offsets[meeple_position.coordinate_with_side.side][0]
            y = meeple_position.coordinate_with_side.coordinate.row * self.tile_size + self.meeple_position_offsets[meeple_position.coordinate_with_side.side][1]

        self.canvas.create_image(
            x,
            y,
            anchor=CENTER,
            image=image
        )

    def __flattenAlpha(self, img):
        alpha = img.split()[-1]  # Pull off the alpha layer
        ab = alpha.tobytes()  # Original 8-bit alpha

        checked = []  # Create a new array to store the cleaned up alpha layer bytes

        # Walk through all pixels and set them either to 0 for transparent or 255 for opaque fancy pants
        transparent = 50  # change to suit your tolerance for what is and is not transparent

        p = 0
        for pixel in range(0, len(ab)):
            if ab[pixel] < transparent:
                checked.append(0)  # Transparent
            else:
                checked.append(255)  # Opaque
            p += 1

        mask = Image.frombytes('L', img.size, bytes(checked))

        img.putalpha(mask)

        return img

    def __draw_tile(self, column_index, row_index, tile):
        image_filename = tile.image
        reference = f"{image_filename}_{str(tile.turns)}"
        if reference in self.tile_image_refs:
            photo_image = self.tile_image_refs[reference]
        else:
            abs_file_path = os.path.join(self.images_path, image_filename)
            image = Image.open(abs_file_path).resize((self.tile_size, self.tile_size), Image.ANTIALIAS).rotate(
                -90 * tile.turns)
            image = self.__flattenAlpha(image)
            height = image.height
            width = image.width
            crop_width = max(0, width - height) / 2
            crop_height = max(0, height - width) / 2
            image.crop((crop_width, crop_height, crop_width, crop_height))
            photo_image = ImageTk.PhotoImage(image)
        self.tile_image_refs[f"{image_filename}_{str(tile.turns)}"] = photo_image
        self.canvas.create_image(column_index * self.tile_size, row_index * self.tile_size, anchor=NW, image=photo_image)

    def __get_meeple_image(self, player: int, meeple_type: int):
        reference = f"{str(player)}_{str(meeple_type)}"

        if reference in self.meeple_image_refs:
            return self.meeple_image_refs[reference]

        icon_type = MeepleType.NORMAL
        if meeple_type == MeepleType.ABBOT:
            icon_type = meeple_type

        image_filename = self.meeple_icons[icon_type][player]
        abs_file_path = os.path.join(self.images_path, image_filename)

        photo_image = None
        if meeple_type == MeepleType.NORMAL or meeple_type == MeepleType.ABBOT:
            image = Image.open(abs_file_path).resize((self.meeple_size, self.meeple_size), Image.ANTIALIAS)
            image = self.__flattenAlpha(image)
            photo_image = ImageTk.PhotoImage(image)
        elif meeple_type == MeepleType.BIG:
            image = Image.open(abs_file_path).resize((self.big_meeple_size, self.big_meeple_size), Image.ANTIALIAS)
            image = self.__flattenAlpha(image)
            photo_image = ImageTk.PhotoImage(image)
        elif meeple_type == MeepleType.FARMER:
            image = Image.open(abs_file_path).resize((self.meeple_size, self.meeple_size), Image.ANTIALIAS).rotate(-90)
            image = self.__flattenAlpha(image)
            photo_image = ImageTk.PhotoImage(image)
        elif meeple_type == MeepleType.BIG_FARMER:
            image = Image.open(abs_file_path).resize((self.big_meeple_size, self.big_meeple_size), Image.ANTIALIAS).rotate(-90)
            image = self.__flattenAlpha(image)
            photo_image = ImageTk.PhotoImage(image)
        else:
            print(f"ERROR LOADING IMAGE {abs_file_path}!")
            exit(1)

        self.meeple_image_refs[f"{str(player)}_{str(meeple_type)}"] = photo_image
        return photo_image
