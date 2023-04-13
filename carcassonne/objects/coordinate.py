class Coordinate:

    def __init__(self, row: int, column: int):
        self.row = row
        self.column = column

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column

    def __hash__(self):
        return hash((self.row, self.column))

    def __str__(self):
        return f"({self.row}, {self.column})"

    def __repr__(self) -> str:
        return str(self)

    def to_json(self):
        return [self.row, self.column]

    @staticmethod
    def from_json(data: list):
        return Coordinate(data[0], data[1])
