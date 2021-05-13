import random


def game(n):
    randomvar = random.randint(1,n)
    randomsum = randomvar
    #print(randomsum)
    for i in range(n-1):
        x = random.randint(1,n)
        intermideate = randomvar - x
        randomsum = randomsum + abs(intermideate)
        randomvar = x

    return randomsum

def trials(n,m):
    sample = []
    for i in range(m):
        sample.append(game(n))

    mean = sum(sample) / len(sample) 
    variance = sum([((x - mean) ** 2) for x in sample]) / len(sample) 
    std_deviation = variance ** 0.5

    return mean , std_deviation, sample

def prob_x_greater_than_j(j,sample):
    count = 0
    for i in sample:
        if i>=j:
            count = count+1
    return (count/len(sample))
def main():
    mean , stdDev, sample = trials(20,100000)
    prob = prob_x_greater_than_j(160,sample)
    print(mean)
    print(stdDev)
    print("Prob = "+str(prob))
    #print(sample)

main()