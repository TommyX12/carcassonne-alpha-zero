from ..objects.connection import Connection
from ..objects.farmer_connection import FarmerConnection
from ..objects.farmer_side import FarmerSide
from ..objects.side import Side

class SideModificationUtil:

    @classmethod
    def turn_side(cls, side: int, times: int) -> int:
        if side == Side.CENTER:
            return side

        return (side + times * 2) % 8

    @classmethod
    def opposite_side(cls, side: int):
        return cls.turn_side(side, 2)

    @classmethod
    def turn_sides(cls, sides: [Side], times: int):
        return list(map(lambda side: cls.turn_side(side, times), sides))

    @classmethod
    def turn_farmer_side(cls, farmer_side: int, times: int) -> int:
        return (farmer_side + times * 2) % 8

    @classmethod
    def turn_farmer_sides(cls, farmer_sides: [FarmerSide], times: int) -> [FarmerSide]:
        return list(map(lambda farmer_side: cls.turn_farmer_side(farmer_side, times), farmer_sides))

    @classmethod
    def opposite_farmer_side(cls, farmer_side: int) -> int:
        if farmer_side == FarmerSide.TLL:
            return FarmerSide.TRR
        elif farmer_side == FarmerSide.TLT:
            return FarmerSide.BLB
        elif farmer_side == FarmerSide.TRT:
            return FarmerSide.BRB
        elif farmer_side == FarmerSide.TRR:
            return FarmerSide.TLL
        elif farmer_side == FarmerSide.BRR:
            return FarmerSide.BLL
        elif farmer_side == FarmerSide.BRB:
            return FarmerSide.TRT
        elif farmer_side == FarmerSide.BLB:
            return FarmerSide.TLT
        else:  # farmer_side == FarmerSide.BLL:
            return FarmerSide.BRR

    @classmethod
    def turn_farmer_connection(cls, farmer_connection: FarmerConnection, times: int):
        return FarmerConnection(
            farmer_positions=cls.turn_sides(farmer_connection.farmer_positions, times),
            tile_connections=cls.turn_farmer_sides(farmer_connection.tile_connections, times),
            city_sides=cls.turn_sides(farmer_connection.city_sides, times)
        )

    @classmethod
    def turn_connection(cls, connection: Connection, times: int) -> Connection:
        return Connection(cls.turn_side(connection.a, times), cls.turn_side(connection.b, times))
