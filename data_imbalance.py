import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Balance dataset by class distribution")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    parser.add_argument(
        "--dominant_class",
        type=int,
        default=0,
        help="Dominant class label (default: 0)"
    )
    parser.add_argument(
        "--dominant_ratio",
        type=float,
        default=0.90,
        help="Ratio of dominant class (default: 0.90)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()

args = parse_arguments()
df = pd.read_csv(args.input_path)

def make_class_dominant(
    df,
    dominant_class,
    dominant_ratio,
    seed=42
):

    if not 0 < dominant_ratio < 1:
        raise ValueError("dominant_ratio must be between 0 and 1")

    other_class = 1 if dominant_class == 0 else 0

    n_dominant = len(df[df["target"] == dominant_class])
    n_other = len(df[df["target"] == other_class])

    total_size = min(n_dominant / dominant_ratio, n_dominant + n_other)
    n_dominant_sample = int(total_size * dominant_ratio)
    n_other_sample = int(total_size - n_dominant_sample)

    dominant_sample = df[df["target"] == dominant_class].sample(
        n=n_dominant_sample, random_state=seed
    )
    other_sample = df[df["target"] == other_class].sample(
        n=n_other_sample, random_state=seed
    )

    result = pd.concat([dominant_sample, other_sample])
    
    print("Data distribution: ", result['target'].value_counts())
    return result


# sample usage: result_df = make_class_dominant(df, 1, 0.84)

result_df = make_class_dominant(
    df,
    dominant_class=args.dominant_class,
    dominant_ratio=args.dominant_ratio,
    seed=args.seed
)

result_df.to_csv(args.output_path, index=False)

print("Saved to:", args.output_path)