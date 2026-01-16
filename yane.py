from IPython.display import HTML

def yog(content):
    display(HTML(f'''<div style="color:rgb(255,70,0)">--- yog start ---</div>'''))
    display(content)
    display(HTML(f'''<div style="color:rgb(255,70,0)">--- yog end ---</div>'''))

def bar(caption='', barLen=80):
    if caption:
        display(HTML(f'''<pre style="color:rgb(255,255,0)">{'-'*((barLen - len(caption))//2)} {caption} {'-'*(barLen - 2 - len(caption) - (barLen - len(caption))//2)}</pre>'''))
    else:
        display(HTML(f'''<pre style="color:rgb(255,255,0)">{'-'*barLen}</pre>'''))