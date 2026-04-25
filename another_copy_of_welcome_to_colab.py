import matplotlib.pyplot as plt

# Generation steps
generations = list(range(0, 151, 10))

# Synthetic Convergence Data based on your results
# GA starts high and plateaus at 5665.24
ga_fitness = [25000, 12000, 8000, 6500, 6000, 5800, 5750, 5700, 5680, 5670, 5668, 5666, 5665, 5665, 5665, 5665]

# MIP is constant at 0.5
mip_fitness = [0.5] * len(generations)

plt.figure(figsize=(10, 6))
plt.plot(generations, ga_fitness, label='Genetic Algorithm (Scruffy)', color='#de7e5d', linewidth=2, marker='o')
plt.axhline(y=0.5, label='MIP Optimal Baseline (Tidy)', color='#004c6d', linestyle='--', linewidth=2)

plt.yscale('log') # Log scale is essential due to the 11,000x difference
plt.title("Algorithm Convergence: Cost vs. Generations", fontsize=14)
plt.xlabel("Generations", fontsize=12)
plt.ylabel("Objective Score (Log Scale)", fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()

# Annotate the gap
plt.annotate('Optimality Gap', xy=(100, 10), xytext=(110, 100),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()
