import numpy as np

multipliers = np.array([80, 50, 83, 31, 60, 89, 10, 37, 70, 90, 17, 40, 73, 100, 20, 41, 79, 23, 47, 30])
contestants = np.array([6, 4, 7, 2, 4, 8, 1, 3, 4, 10, 1, 3, 4, 15, 2, 3, 5, 2, 3, 2])

percentages = multipliers / 5 - contestants

for i in range(len(multipliers)):
    print(f"For the crate with multiplier {multipliers[i]}, we need {percentages[i]}% of people to choose it.")

print(percentages.sum())
print(len(multipliers))