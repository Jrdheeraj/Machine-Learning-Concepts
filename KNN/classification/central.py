import statistics as stats
import csv

data = []
with open("STD.csv", "r") as file:  
    reader = csv.reader(file)
    for row in reader:
        data.append(float(row[3]))

mean = stats.mean(data)
median = stats.median(data)
mode = stats.mode(data)
variance = stats.variance(data)
std_div = stats.stdev(data)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode: {mode}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_div}")
