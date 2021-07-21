# Converts the README.md to HTML
import markdown
import os

README_PATH = "../README.md"
OUTPUT_DIR = "html"

INSERT =\
"""
## Talk & Demo
<div class="row ml-2 mt-3 mb-5">
<iframe width="560" height="315" src="https://www.youtube.com/embed/TNwi1kS715Q" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

<div class="row ml-2 mt-3 mb-5">
<iframe width="560" height="315" src="https://www.youtube.com/embed/V54RY8v8VmA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
"""

# convert README as html
with open(README_PATH) as f:
    md = f.read()
    md = md.replace("<!-- #<># -->", INSERT)

    content = markdown.markdown(md, extensions=["fenced_code"])

    # Replace asset paths
    content = content.replace("docs/html/assets", "assets")
    # Replace <p><code> with <pre><code>
    content = content.replace("<p><code>", "<pre><code>")
    content = content.replace("</code></p>", "</code></pre>")

# Load template
with open("_template.html") as f:
    template = f.read()
    html = template.replace("#{}#", content)

with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
    f.write(html)
