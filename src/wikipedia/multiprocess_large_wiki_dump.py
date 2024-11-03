import bz2
import os
import multiprocessing
import duckdb
import xml.sax
import io
import wikitextparser as wtp

DATA_DIR = "/Users/einar/git/hafsteinn/together_rag/data"

# Paths to your dump and index files
# DUMP_FILE = f"{DATA_DIR}/wikipedia/no/nowiki-20241001-pages-articles-multistream.xml.bz2"
# INDEX_FILE = f"{DATA_DIR}/wikipedia/no/nowiki-20241001-pages-articles-multistream-index.txt.bz2"

DUMP_FILE = f"{DATA_DIR}/wikipedia/en/enwiki-20241001-pages-articles-multistream.xml.bz2"
INDEX_FILE = f"{DATA_DIR}/wikipedia/en/enwiki-20241001-pages-articles-multistream-index.txt.bz2"

language = DUMP_FILE.split("/")[-2]
DATABASE_FILE = f"{DATA_DIR}/wikipedia_articles_{language}.duckdb"

# Number of worker processes (adjust as needed)
NUM_WORKERS = 10

# Sentinel value to indicate the writer should stop
SENTINEL = "DONE"

def get_offsets(index_file):
    """
    Extract unique offsets from the index file.
    """
    offsets = set()
    with bz2.open(index_file, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) >= 1:
                try:
                    offset = int(parts[0])
                    offsets.add(offset)
                except ValueError:
                    continue  # Skip lines with invalid format
    return sorted(offsets)

class WikiXmlHandler(xml.sax.ContentHandler):
    """
    SAX handler to parse XML content and send rows to the queue.
    """
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.current_tag = ""
        self.in_revision = False
        self.in_page = False
        self.page = {}
        self.buffer = ""

    def startElement(self, name, attrs):
        self.current_tag = name
        if name == "page":
            self.in_page = True
            self.page = {}
        elif name == "revision":
            self.in_revision = True
        elif name == "redirect":
            self.page["redirect"] = attrs.getValue("title")

    def endElement(self, name):
        if name == "page":
            self.in_page = False
            # Process the page if it's not a redirect and is in the main namespace
            if (
                "ns" in self.page
                and self.page["ns"] == "0"
                and "redirect" not in self.page
            ):
                self.process_page(self.page)
            self.page = {}
        elif name == "revision":
            self.in_revision = False
        elif self.current_tag in ["title", "ns", "id", "timestamp", "text"]:
            self.page[self.current_tag] = self.buffer.strip()
        self.buffer = ""
        self.current_tag = ""

    def characters(self, content):
        if self.current_tag in ["title", "ns", "id", "timestamp", "text"]:
            self.buffer += content

    def process_page(self, page):
        """
        Extract metadata and processed text, then send it to the queue.
        """
        try:
            title = page.get("title", "")
            text = page.get("text", "")
            page_id = int(page.get("id", -1))
            timestamp = page.get("timestamp", "")

            if text and not text.lower().startswith("#redirect"):
                parsed = wtp.parse(text)
                plain_text = parsed.plain_text()
                word_count = len(plain_text.split())
                outlinks = parsed.wikilinks
                outlink_count = len(outlinks)
                categories = [
                    link.title.strip()
                    for link in outlinks
                    if link.title.startswith("Category:")
                ]
                templates = parsed.templates

                is_disambiguation = any(
                    template.name.strip().lower() == "disambiguation"
                    for template in templates
                )
                if is_disambiguation:
                    return  # Skip disambiguation pages

                external_links = parsed.external_links
                url_title = title.replace(" ", "_")
                url = f"https://{language}.wikipedia.org/wiki/{url_title}"

                # Prepare the data tuple
                data = (
                    page_id,
                    title,
                    url,
                    word_count,
                    outlink_count,
                    len(categories),
                    "|".join(categories),
                    len(templates),
                    len(external_links),
                    timestamp,
                    plain_text,
                )

                # Send the data to the queue
                self.queue.put(data)
        except Exception as e:
            print(f"Error processing page {page.get('id', 'Unknown')}: {e}")

def writer_process(queue, database_file):
    """
    Writer process that consumes data from the queue and writes to DuckDB.
    """
    try:
        conn = duckdb.connect(database_file)
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                page_id INTEGER,
                title TEXT,
                url TEXT,
                word_count INTEGER,
                outlink_count INTEGER,
                category_count INTEGER,
                categories TEXT,
                template_count INTEGER,
                external_link_count INTEGER,
                last_modified TIMESTAMP,
                processed_text TEXT
            )
            """
        )
        conn.commit()

        count = 0
        batch = []
        batch_size = 1000  # Adjust batch size as needed

        while True:
            row = queue.get()
            if row == SENTINEL:
                break
            batch.append(row)
            count += 1
            if count % 25_000 == 0:
                print(f"Processed {count} articles")

            # Insert in batches for efficiency
            if len(batch) >= batch_size:
                cursor.executemany(
                    """
                    INSERT INTO articles (
                        page_id, title, url, word_count, outlink_count,
                        category_count, categories, template_count,
                        external_link_count, last_modified, processed_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                batch = []

        # Insert any remaining data
        if batch:
            cursor.executemany(
                """
                INSERT INTO articles (
                    page_id, title, url, word_count, outlink_count,
                    category_count, categories, template_count,
                    external_link_count, last_modified, processed_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.commit()

        print(f"Writer process finished. Total articles processed: {count}")
        conn.close()
    except Exception as e:
        print(f"Writer process encountered an error: {e}")

def worker_process(offset_queue, data_queue):
    """
    Worker process that reads from the dump, parses XML, and sends data to the queue.
    """
    try:
        while True:
            offset = offset_queue.get()
            if offset == SENTINEL:
                break  # Exit the loop if sentinel is received

            with open(DUMP_FILE, "rb") as dump_file:
                dump_file.seek(offset)
                decompressor = bz2.BZ2Decompressor()
                xml_data = b""
                while True:
                    chunk = dump_file.read(1024 * 1024 * 8)  # 16 MB
                    if not chunk:
                        break
                    try:
                        decompressed = decompressor.decompress(chunk)
                        xml_data += decompressed
                        if decompressor.eof:
                            break
                    except OSError as e:
                        print(f"Decompression error at offset {offset}: {e}")
                        break

                # Wrap the XML fragment with <mediawiki> tags
                xml_content = b"<mediawiki>" + xml_data + b"</mediawiki>"

                handler = WikiXmlHandler(data_queue)
                try:
                    xml.sax.parse(io.BytesIO(xml_content), handler)
                except Exception as e:
                    print(f"XML parsing error at offset {offset}: {e}")
    except Exception as e:
        print(f"Worker process encountered an error: {e}")

def main():
    # Extract unique offsets
    print("Extracting offsets from the index file...")
    offsets = get_offsets(INDEX_FILE)
    print(f"Total unique offsets extracted: {len(offsets)}")
    
    # Create a multiprocessing.Queue for data
    data_queue = multiprocessing.Queue(maxsize=10000)  # Adjust maxsize as needed

    # Create a multiprocessing.Queue for offsets
    print(f"Creating offset queue")
    offset_queue = multiprocessing.Queue()

    # Start the writer process
    print(f"Starting writer process")
    writer = multiprocessing.Process(target=writer_process, args=(data_queue, DATABASE_FILE))
    writer.start()

    # Start worker processes
    print(f"Starting {NUM_WORKERS} worker processes")
    workers = []
    for _ in range(NUM_WORKERS):
        worker = multiprocessing.Process(target=worker_process, args=(offset_queue, data_queue))
        worker.start()
        workers.append(worker)

    # Now fill the offset_queue
    for offset in offsets:
        offset_queue.put(offset)

    # After all offsets are added, put a sentinel for each worker
    for _ in range(NUM_WORKERS):
        offset_queue.put(SENTINEL)

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Send the sentinel to the writer to indicate completion
    data_queue.put(SENTINEL)

    # Wait for the writer to finish
    writer.join()

    print("All processes have completed successfully.")


if __name__ == "__main__":
    main()
