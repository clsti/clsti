"""Fill the website block of README.md from the personal website: About Me, current position and a link to
the project overview.  Run nightly by .github/workflows/profile.yml; README.md only changes when
the website did.
python tools/sync_profile.py [url] [readme]"""
import html.parser
import sys
import urllib.parse
import urllib.request

URL = "https://clsti.github.io/"
START, END = "<!-- website:start", "<!-- website:end -->"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


class Node:
    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.parent, self.children = tag, dict(attrs), parent, []

    @property
    def classes(self):
        return (self.attrs.get("class") or "").split()

    def iter(self):
        for c in self.children:
            if isinstance(c, Node):
                yield c
                yield from c.iter()

    def all(self, cls):
        return [n for n in self.iter() if cls in n.classes]

    def one(self, cls):
        found = self.all(cls)
        return found[0] if found else None

    def text(self):
        s = "".join(c if isinstance(c, str) else c.text() for c in self.children)
        return " ".join(s.split())

    def href(self):
        a = self if self.tag == "a" else next((n for n in self.iter() if n.tag == "a"), None)
        return a.attrs.get("href") if a else None


class Tree(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = self.cur = Node("root", [], None)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.cur)
        self.cur.children.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def md(text):
    return text.replace("<", "&lt;").replace("[", "\\[").replace("]", "\\]")


def link(text, href):
    return f"[{md(text)}]({href})" if href else md(text)


def block(page, base):
    def url(href):
        return urllib.parse.urljoin(base, href) if href else None

    bio = page.one("hero-bio")
    about = " ".join(p.text() for p in bio.iter() if p.tag == "p") if bio else ""
    if not about:
        sys.exit("no About Me text found on the website")
    out = [md(about), ""]

    now = []
    for item in page.all("experience-item"):
        dates, title, company = (item.one(c) for c in ("experience-dates", "experience-title", "experience-company"))
        if dates and title and "present" in dates.text().lower():
            now.append(f"{md(title.text())}, {link(company.text(), url(company.href()))}" if company
                       else md(title.text()))
    if now:
        out += ["**Now:** " + " · ".join(now), ""]

    menu = next((n for n in page.iter() if n.tag == "a" and n.text() == "Projects"), None)
    if menu:
        out.append(f"Check out my [projects]({url(menu.href())}).")
    return "\n".join(out).rstrip()


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else URL
    path = sys.argv[2] if len(sys.argv) > 2 else "README.md"
    req = urllib.request.Request(url, headers={"User-Agent": "commit-robots profile sync"})
    with urllib.request.urlopen(req, timeout=30) as r:
        tree = Tree()
        tree.feed(r.read().decode("utf-8"))
    new = block(tree.root, url)

    with open(path, encoding="utf-8") as f:
        readme = f.read()
    i, j = readme.find(START), readme.find(END)
    if i < 0 or j < i:
        sys.exit(f"{path}: no '{START} ... {END}' block")
    head = readme.index("-->", i) + 3
    updated = readme[:head] + "\n" + new + "\n\n" + readme[j:]
    if updated == readme:
        print("README.md unchanged")
        return
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated)
    print("README.md updated from", url)


if __name__ == "__main__":
    main()
