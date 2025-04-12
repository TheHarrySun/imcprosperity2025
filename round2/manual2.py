import numpy as np

multipliers = np.array([10, 80, 37, 90, 31, 17, 50, 20, 73, 89])
inhabitants = np.array([1, 6, 3, 10, 2, 1, 4, 2, 4, 8])

percentages = multipliers / 5 - inhabitants

for i in range(len(multipliers)):
    print(f"For the crate with multiplier {multipliers[i]}, we need {percentages[i]}% of people to choose it.")
