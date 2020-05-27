

from docutils import core


def rst2html(string):
    temp = core.publish_parts(string,writer_name="html")
    result = "".join([temp["stylesheet"], temp["html_body"]])
    result = result.replace('<pre class="literal-block">','<pre class=\"code\">')
    result = result.replace('<tt class="docutils literal">','<code>')
    result = result.replace("</tt>", "</code>")
    return result

file_name = "index"
rst_path = "{}.rst".format(file_name)
html_path = "{}.html".format(file_name)

with open(rst_path) as f:
    rst_str = "".join(f.readlines())
    
html = rst2html(rst_str)

with open("{}".format(html_path), "w") as f:
    f.write(html)