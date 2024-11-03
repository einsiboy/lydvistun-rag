import xml.etree.ElementTree as ET
import wikitextparser as wtp
from collections import Counter
import pandas as pd
from urllib.parse import quote


class WikiXMLHandler:
    def __init__(self, max_pages=None):
        self.current_page = {}
        self.pages_processed = 0
        self.max_pages = max_pages
        self.page_count = Counter()
        self.metadata = []

    def parse(self, xml_file):
        for event, elem in ET.iterparse(xml_file, events=("start", "end")):
            if event == "end" and elem.tag.endswith("page"):
                self.process_page(elem)
                elem.clear()
                if self.max_pages and self.pages_processed >= self.max_pages:
                    break

    def process_page(self, elem):
        ns = elem.find("./{*}ns")
        if ns is not None and ns.text == "0":  # Main articles
            title = elem.findtext("./{*}title")
            text = elem.findtext(".//{*}text")
            page_id = elem.findtext("./{*}id")
            timestamp = elem.findtext(".//{*}timestamp")

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
                external_links = parsed.external_links

                # Generate URL
                url = f"https://is.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

                self.metadata.append(
                    {
                        "page_id": page_id,
                        "title": title,
                        "url": url,
                        "word_count": word_count,
                        "outlink_count": outlink_count,
                        "category_count": len(categories),
                        "categories": "|".join(categories),
                        "template_count": len(templates),
                        "external_link_count": len(external_links),
                        "last_modified": timestamp,
                        "processed_text": plain_text,  # This is the text we'll use for embedding
                        "raw_text": text,  # Original wikitext, in case we need it later
                    }
                )

                self.page_count["content"] += 1
            else:
                self.page_count["redirect"] += 1

            self.pages_processed += 1
            if self.pages_processed % 10000 == 0:
                print(
                    f"Processed {self.pages_processed} pages "
                    f"(Content: {self.page_count['content']}, "
                    f"Redirects: {self.page_count['redirect']})"
                )


def process_wikipedia_dump(file_path, max_pages=None):
    handler = WikiXMLHandler(max_pages)
    handler.parse(file_path)

    print(f"Total pages processed: {handler.pages_processed}")
    print(f"Content pages: {handler.page_count['content']}")
    print(f"Redirect pages: {handler.page_count['redirect']}")

    return pd.DataFrame(handler.metadata)


# Usage
# dump_file = "../../data/wikipedia/iswiki-20240901-pages-articles.xml"
# df = process_wikipedia_dump(dump_file, max_pages=None)

# print(len(df))
# df.head()
