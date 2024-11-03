import os
import bz2
from collections import defaultdict
import pandas as pd

# Filesystem paths
DATA_DIR = "/Users/einar/git/hafsteinn/together_rag/data/wikipedia_meta/"
OUTPUT_DIR = "/Users/einar/git/hafsteinn/together_rag/data/wikipedia_meta/aggregated/"

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Languages to keep, split into three batches
WIKI_CODES_BATCHES = [
    ["is.wikipedia", "no.wikipedia", "nn.wikipedia", "de.wikipedia"],
    ["en.wikipedia", "fo.wikipedia"],
    ["nl.wikipedia", "se.wikipedia", "dk.wikipedia", "lu.wikipedia"],
]


def process_pageviews(bz2_file_path, wiki_codes_to_keep):
    """
    Process a single .bz2 pageviews file and aggregate data for specified wiki codes.

    Parameters:
        bz2_file_path (str): Path to the .bz2 file.
        wiki_codes_to_keep (list): List of wiki codes to process.

    Returns:
        defaultdict: Aggregated views and page_ids per (wiki_code, article_title).
    """
    monthly_views = defaultdict(lambda: {"views": 0, "page_ids": set()})

    with bz2.open(bz2_file_path, "rt") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 6:
                continue  # Skip invalid lines

            wiki_code = parts[0].strip()
            if wiki_code not in wiki_codes_to_keep:
                continue  # Skip unwanted wiki codes

            article_title = parts[1].strip()
            page_id = parts[2].strip()
            try:
                daily_total = int(parts[-2].strip())
            except ValueError:
                continue  # Skip lines with invalid view counts

            key = (wiki_code, article_title)
            monthly_views[key]["views"] += daily_total
            monthly_views[key]["page_ids"].add(page_id)

    return monthly_views


def monthly_stats_to_df(monthly_stats):
    """
    Convert aggregated monthly_stats to a pandas DataFrame.

    Parameters:
        monthly_stats (dict): Aggregated data.

    Returns:
        pd.DataFrame: DataFrame containing aggregated data.
    """
    data = []
    for (wiki_code, article_title), stats in monthly_stats.items():
        data.append(
            {
                "wiki_code": wiki_code,
                "article_title": article_title,
                "views": stats["views"],
                "page_ids": list(stats["page_ids"]),
            }
        )

    df = pd.DataFrame(data)
    return df


def append_to_parquet(df, language, output_dir):
    """
    Append DataFrame to a Parquet file specific to the language.

    Parameters:
        df (pd.DataFrame): DataFrame to append.
        language (str): Wiki code/language.
        output_dir (str): Directory to save Parquet files.
    """
    file_path = os.path.join(output_dir, f"{language}_monthly_views.parquet")
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_parquet(file_path)
    else:
        df.to_parquet(file_path)


def process_batch(batch, bz2_files):
    """
    Process a batch of wiki codes for all .bz2 files.

    Parameters:
        batch (list): List of wiki codes to process in this batch.
        bz2_files (list): List of .bz2 file paths to process.
    """
    for bz2_file_path in bz2_files:
        print(f"Processing file: {bz2_file_path} for batch: {batch}")
        monthly_stats = process_pageviews(bz2_file_path, batch)
        df = monthly_stats_to_df(monthly_stats)

        for language in batch:
            language_df = df[df["wiki_code"] == language]
            if not language_df.empty:
                append_to_parquet(language_df, language, OUTPUT_DIR)
                print(
                    f"Appended data for {language} to {language}_monthly_views.parquet"
                )


def main():
    """
    Main function to process all .bz2 pageview files and aggregate data in batches.
    """
    bz2_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith("-user.bz2")
    ]

    for batch_num, batch in enumerate(WIKI_CODES_BATCHES, 1):
        print(f"Processing batch {batch_num} of {len(WIKI_CODES_BATCHES)}")
        process_batch(batch, bz2_files)

    print("Processing complete.")


if __name__ == "__main__":
    main()
