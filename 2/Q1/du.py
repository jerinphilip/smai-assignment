import re

def headers(keyfilename):
    keyfile = open(keyfilename, 'r')

    def unpack(line):
        line = line.strip()
        first, rest = line.split('(')
        second, last = rest.split(')')
        last = last.strip(' ')
        return [first, second, last]
    
    extracted = list(map(unpack, keyfile))
    headers = map(lambda x: (x[1], x[2]), extracted)


    return list(headers)

def data(datafilename):
    datafile = open(datafilename, 'r')

    def unpack(line):
        remove_whitespace = lambda x : x.strip(' ')
        entry = map(remove_whitespace, line.strip().split(','))
        return list(entry)

    entries = map(unpack, datafile)
    return list(entries)

if __name__ == '__main__':
    print(headers('data/column.keys'))
    print(data('data/census-income.data'))
