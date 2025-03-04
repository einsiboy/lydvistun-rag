This module (wikipedia) contains code for processing Wikipedia data dumps.

## Data

The tree structure is as follows:

```
data/wikipedia/
    is/
        data_dump_files_for_is
    no/
        data_dump_files_for_no
    en/
        data_dump_files_for_en
    ...
```

## Wikipedia sql files

Wikipedia has sql files for each of the wikis.
We can use these to populate a database to get information about the pages, users, etc.

### Spin up the database with docker compose

```
# from root of repo
docker compose up -d
```

### Run the python script to create schemas and import data

```
# from root of repo
python wikipedia/import_wikipedia_sql.py
```

