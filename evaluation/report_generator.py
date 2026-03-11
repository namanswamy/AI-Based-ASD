import pandas as pd

from evaluation.model_comparison import compare


def generate():

    df = compare()

    report = df.to_markdown()

    with open(
        "outputs/reports/model_report.md",
        "w"
    ) as f:

        f.write(report)

    print("Report generated")


if __name__ == "__main__":

    generate()