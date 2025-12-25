import sys

from utils import settings2csv

if __name__ == '__main__':
    sys.path.append("./")

    r_path = "./out"
    out_path = "./"

    df = settings2csv(r_path=r_path,
                      out_path=out_path)

