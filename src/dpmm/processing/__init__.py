from dpmm.processing.binners import PrivTreeBinner, UniformBinner

BINNERS = [UniformBinner, PrivTreeBinner]
BINNER_DICT = {BINNER.name: BINNER for BINNER in BINNERS}
