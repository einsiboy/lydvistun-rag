"""
For running the app locally to test things out.
"""

from vectordb import create_indices
from vectordb.load_test_data import main as load_test_data


def main():
    create_indices()
    load_test_data()


if __name__ == "__main__":
    main()
