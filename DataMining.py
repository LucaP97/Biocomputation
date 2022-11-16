with open('data1.txt') as f:
    lines = f.read()

#print(lines)

class network:
    def __init__(self):
        self.hweight = None
        self.oweight = None
        self.error = 0

inputNodes = 6
hiddenNodes = 6

linesCut = lines.split(' ')

for i in range(0,7):
    print(i)