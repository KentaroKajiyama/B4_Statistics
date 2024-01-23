from pathlib import Path

def main():
    parent = Path(__file__).resolve().parent
    exec(open(parent.joinpath("postdist_squared_norm_d2_Shimizu_algo.py")).read())
    exec(open(parent.joinpath("postdist_squared_norm_d2_Shimizu_algo_median.py")).read())