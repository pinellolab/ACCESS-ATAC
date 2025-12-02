import argparse
import logging
import warnings
import gzip

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="This script adds barcode to fastq file",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required parameters
    parser.add_argument("--reads_fastq", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    total = 0
    bad = 0
    with gzip.open(args.reads_fastq, "r") as f:
        while True:
            header = f.readline()
            if not header:
                break  # EOF
            seq = f.readline()
            plus = f.readline()
            qual = f.readline()

            if not qual:
                print("Warning: truncated FASTQ record near the end of file")
                break

            total += 1
            seq = seq.strip()
            qual = qual.strip()

            if len(seq) != len(qual):
                bad += 1

    with open(f"{args.out_dir}/{args.out_name}_fastq_quality.txt", "w") as out_f:
        out_f.write("Total_Reads\tMismatched_Seq_Qual_Length\n")
        out_f.write(f"{total}\t{bad}\n")

    logging.info("Done!")

if __name__ == "__main__":
    main()