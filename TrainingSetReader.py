__resource_path__ = "C:/Users/mikke/PycharmProjects/Theano2048/resources/"

def readSet(filename='readtest'):
    dir=__resource_path__
    f = open(dir + filename + '.txt')
    f_answ = open(dir + filename + 'answ.txt')

    arr = []
    for line in f:
        innerArr = []
        for number in line.split():
            innerArr.append(float(number))
        arr.append(innerArr)

    arrAnsw = []
    for line in f_answ:
        arrAnsw.append(int(line))

    return arr, arrAnsw