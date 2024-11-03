import os
import subprocess
import time
import pandas as pd
from sqlalchemy import create_engine, text
import pymysql


def run_sql_file(file_path, schema=None):
    container_name = "wikipedia_db"

    # Wait for the container to be ready
    max_retries = 30
    retry_interval = 5
    for _ in range(max_retries):
        result = subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "mysqladmin",
                "ping",
                "-h",
                "localhost",
                "-u",
                "root",
                "-prootpassword",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            break
        print("Waiting for MySQL to be ready...")
        time.sleep(retry_interval)
    else:
        raise Exception("MySQL container did not become ready in time")

    # Copy the SQL file to the container
    subprocess.run(["docker", "cp", file_path, f"{container_name}:/tmp/"], check=True)

    # Run the SQL file in the container
    command = f"mysql -u root -prootpassword"
    if schema:
        command += f" {schema}"
    command += f" < /tmp/{os.path.basename(file_path)}"

    subprocess.run(
        ["docker", "exec", container_name, "bash", "-c", command], check=True
    )


def get_dataframe(engine, table_name):
    query = text(f"SELECT * FROM {table_name}")
    return pd.read_sql(query, engine)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "..", "..", "data", "wikipedia")
    schemas_file = os.path.join(here, "schemas.sql")

    try:
        # Run schemas.sql first
        print("Initializing schemas...")
        run_sql_file(schemas_file)

        # Iterate over language directories
        for lang_dir in os.listdir(data_dir):
            lang_path = os.path.join(data_dir, lang_dir)
            if os.path.isdir(lang_path):
                schema = f"{lang_dir}wiki"
                sql_files = [f for f in os.listdir(lang_path) if f.endswith(".sql")]

                for sql_file in sql_files:
                    file_path = os.path.join(lang_path, sql_file)
                    print(f"Importing {lang_dir} data from {sql_file}...")
                    run_sql_file(file_path, schema)

        print("Data import and processing complete.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running a subprocess command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
