def revword(word:str) -> str:
    word = word.lower()
    t = word[::-1]
    return t
def countword()->int:
    txt = open('text.txt', 'r')
    count = 0
    i = 0
    for line in txt:
        if i == 0:
            word = line.rstrip("\n")
            count =+1
            i =+1
        else:
            line = line.split()
            for ii in line:
                z = revword (ii)
                if z == word:
                    count = count + 1
                    

    return count
print((countword()))
