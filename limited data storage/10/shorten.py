f = open("en-ud-tag.v2.train.txt", "r", encoding='utf-8')
lines = f.readlines()
f.close()

f = open("en-ud-tag.v2.train.txt", "w", encoding='utf-8')
for i in range(21814):
    f.write(lines[i])
f.close()