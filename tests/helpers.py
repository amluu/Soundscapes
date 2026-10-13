import os
import zipfile

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def make_epub(path, xhtml_files):
    """Write a minimal EPUB 2 with the given fixture files as its spine."""
    items = ''.join(f'<item id="c{i}" href="{n}" media-type="application/xhtml+xml"/>' for i, n in enumerate(xhtml_files))
    spine = ''.join(f'<itemref idref="c{i}"/>' for i in range(len(xhtml_files)))
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('mimetype', 'application/epub+zip')
        z.writestr('META-INF/container.xml',
                   '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr('OEBPS/content.opf',
                   f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0">'
                   f'<manifest>{items}</manifest><spine>{spine}</spine></package>')
        for n in xhtml_files:
            with open(os.path.join(FIXTURES, n), encoding='utf-8') as f:
                z.writestr(f'OEBPS/{n}', f.read())
    return path
