"""Read paragraphs out of an EPUB, in reading (spine) order, with the stdlib only."""
import posixpath
import re
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

BLOCK_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'div'}


class _Paragraphs(HTMLParser):
    """Collect the text of each block element as one paragraph."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.paras, self._buf, self._skip = [], [], 0

    def _flush(self):
        text = ''.join(self._buf).strip()
        if text:
            self.paras.append(text)
        self._buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'head'):
            self._skip += 1
        elif tag in BLOCK_TAGS or tag == 'br':
            self._flush()

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'head'):
            self._skip -= 1
        elif tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()
        return self.paras


def _ns(tag):
    return re.sub(r'\{.*?\}', '', tag)


def spine_documents(path):
    """Yield (href, xhtml string) for each spine item of the EPUB at `path`."""
    with zipfile.ZipFile(path) as z:
        container = ET.fromstring(z.read('META-INF/container.xml'))
        opf_path = next(e.get('full-path') for e in container.iter() if _ns(e.tag) == 'rootfile')
        opf = ET.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {e.get('id'): e.get('href') for e in opf.iter() if _ns(e.tag) == 'item'}
        for e in opf.iter():
            if _ns(e.tag) == 'itemref':
                href = manifest[e.get('idref')]
                yield href, z.read(posixpath.join(base, href)).decode('utf-8', errors='replace')


def epub_paragraphs(path):
    """Return a list of (href, [paragraph, ...]) in reading order."""
    out = []
    for href, doc in spine_documents(path):
        p = _Paragraphs()
        p.feed(doc)
        out.append((href, p.close()))
    return out
