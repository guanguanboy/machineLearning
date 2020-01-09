import collections

print('Regular dictionary:')
d1 = {}
d1['a'] = 'A'
d1['c'] = 'C'
d1['b'] = 'B'
for k, v in d1.items():
    print(k, v)

print('\nOrderedDict:')
d2 = collections.OrderedDict()
d2['a'] = 'A'
d2['c'] = 'C'
d2['b'] = 'B'

for k, v in d2.items():
    print(k, v)