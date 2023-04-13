from enum import Enum

from ..objects.side import Side


class FarmerSide(object):
    TLL = 0
    TLT = 1
    TRT = 2
    TRR = 3
    BRR = 4
    BRB = 5
    BLB = 6
    BLL = 7

    @staticmethod
    def get_side(side: int) -> Side:
        s = FARMER_SIDE_TO_STR[side]
        if s[2] == "l":
            return Side.LEFT
        if s[2] == "r":
            return Side.RIGHT
        if s[2] == "b":
            return Side.BOTTOM
        if s[2] == "t":
            return Side.TOP


FARMER_SIDE_TO_STR = {
    FarmerSide.TLL: "tll",
    FarmerSide.TLT: "tlt",
    FarmerSide.TRT: "trt",
    FarmerSide.TRR: "trr",
    FarmerSide.BRR: "brr",
    FarmerSide.BRB: "brb",
    FarmerSide.BLB: "blb",
    FarmerSide.BLL: "bll",
}