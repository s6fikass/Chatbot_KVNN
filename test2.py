

import re
line="today it will be about 30 - 40 degrees and tomorrow"
p = re.compile("\\b(\d{2} - \d{2,3}( f| degrees)+)\\b")
x=p.findall(line)
print(x)

for degrees in x:
    low="low_of_"+degrees[0].split("-")[0].strip()+"_f"
    high="high_of_"+degrees[0].split("-")[1].strip()
    high=high.split(" ")[0]+'_f'
    s=re.sub(degrees[0], low+' and '+high,line)
    print(s)
