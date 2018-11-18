import re
pattern="^((?![a-z]|[1-9]).)*"+"new york"
print(pattern)
print(re.sub(pattern,"_entity_", "check forecast for new york"))