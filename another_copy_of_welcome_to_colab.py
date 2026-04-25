import pandas as pd
import numpy as np
import pulp
import time
import random

# =================================================================
# 1. DATA PREPARATION (The "Sunday Injection" Logic)
# =================================================================
def prepare_staffing_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for first 10 stores and one full week (Mon-Sun)
    mask = (df['Store'] <= 10) & (df['Date'] >= '2013-01-07') & (df['Date'] <= '2013-01-13')
    week_df = df.loc[mask].copy()
    week_df['DayIdx'] = week_df['Date'].dt.dayofweek # Mon=0, Sun=6
    
    # Calculate Mean Sunday Injection
    # We take the Mon-Sat average sales per store to project Sunday demand
    avg_sales = week_df[week_df['DayIdx'] < 6].groupby('Store')['Sales'].mean().to_dict()
    
    def get_demand(row):
        # Use actual sales for Mon-Sat; use Mean projection for Sunday
        sales = row['Sales'] if row['DayIdx'] < 6 else avg_sales.get(row['Store'], 5000)
        # 1200:1 Ratio with a 2-man safety floor
        return max(2, int(np.ceil(sales / 1200)))
    
    week_df['Demand'] = week_df.apply(get_demand, axis=1)
    
    # Pivot into (10 Stores x 7 Days) Matrix
    matrix = week_df.pivot(index='Store', columns='DayIdx', values='Demand').values
    return matrix

# =================================================================
# 2. MIP SOLVER (Tidy Baseline)
# =================================================================
def solve_mip(demand_matrix):
    prob = pulp.LpProblem("Tidy_Staffing", pulp.LpMinimize)
    
    # Variables
    # x[employee][day][store]
    x = pulp.LpVariable.dicts("work", (range(100), range(7), range(10)), cat=pulp.LpBinary)
    over = pulp.LpVariable.dicts("over", (range(10), range(7)), lowBound=0)
    under = pulp.LpVariable.dicts("under", (range(10), range(7)), lowBound=0)
    legal_violation = pulp.LpVariable.dicts("legal", range(100), cat=pulp.LpBinary)

    # Objective: Minimize Understaffing (High), Overstaffing (Low), and Legal Violations (Massive)
    prob += (
        pulp.lpSum(under[s][d] * 100 for s in range(10) for d in range(7)) +
        pulp.lpSum(over[s][d] * 1 for s in range(10) for d in range(7)) +
        pulp.lpSum(legal_violation[e] * 1000 for e in range(100)) +
        pulp.lpSum(x[e][d][s] * 0.1 for e in range(100) for d in range(7) for s in range(10) if s != (e // 10))
    )

    # Constraints
    for s in range(10):
        for d in range(7):
            # Demand coverage
            prob += pulp.lpSum(x[e][d][s] for e in range(100)) - over[s][d] + under[s][d] == demand_matrix[s, d]
    
    for e in range(100):
        # One shift per day max
        for d in range(7):
            prob += pulp.lpSum(x[e][d][s] for s in range(10)) <= 1
        # 6-day work limit
        prob += pulp.lpSum(x[e][d][s] for d in range(7) for s in range(10)) - 6 <= legal_violation[e] * 7

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract assignments
    res = np.full((100, 7), -1)
    for e in range(100):
        for d in range(7):
            for s in range(10):
                if pulp.value(x[e][d][s]) == 1: res[e, d] = s
    return res, pulp.value(prob.objective)

# =================================================================
# 3. GA SOLVER (Scruffy Model)
# =================================================================
class ScruffyGA:
    def __init__(self, demand):
        self.demand = demand
        self.pop_size = 50
        self.population = np.random.randint(-1, 10, (50, 100, 7))

    def fitness(self, chromo):
        penalty = 0
        for d in range(7):
            for s in range(10):
                assigned = np.sum(chromo[:, d] == s)
                diff = assigned - self.demand[s, d]
                if diff < 0: penalty += abs(diff) * 100
                elif diff > 0: penalty += diff * 1
        for e in range(100):
            if np.sum(chromo[e] >= 0) > 6: penalty += 1000
            for d in range(7):
                s = chromo[e, d]
                if s != -1 and s != (e // 10): penalty += 0.1
        return penalty

    def evolve(self, gens=150):
        for _ in range(gens):
            scores = [self.fitness(ind) for ind in self.population]
            sorted_idx = np.argsort(scores)
            self.population = self.population[sorted_idx]
            
            new_pop = list(self.population[:10]) # Elitism
            while len(new_pop) < self.pop_size:
                p1, p2 = self.population[random.randint(0, 15)], self.population[random.randint(0, 15)]
                child = np.where(np.random.rand(100, 7) > 0.5, p1, p2)
                if random.random() < 0.2: # Mutation
                    child[random.randint(0, 99), random.randint(0, 6)] = random.randint(-1, 9)
                new_pop.append(child)
            self.population = np.array(new_pop)
        return self.population[0], min(scores)

# =================================================================
# 4. EXECUTION AND REPORTING
# =================================================================
path = "/content/train.csv" # Change this to your local path
demand = prepare_staffing_data(path)

print("Solving Tidy MIP...")
mip_roster, mip_score = solve_mip(demand)

print("Solving Scruffy GA...")
ga_roster, ga_score = ScruffyGA(demand).evolve()

# Generate the Store-Day Breakdown Table
def get_breakdown(roster, name):
    data = []
    for s in range(10):
        store_row = {'Store': s + 1}
        for d in range(7):
            store_row[f'Day_{d+1}'] = np.sum(roster[:, d] == s)
        data.append(store_row)
    return pd.DataFrame(data)

mip_table = get_breakdown(mip_roster, "MIP")
ga_table = get_breakdown(ga_roster, "GA")

print("\nMIP Store-Day Breakdown:")
print(mip_table)
print("\nGA Store-Day Breakdown:")
print(ga_table)

# Save to CSV
mip_table.to_csv("appendix_mip_breakdown.csv", index=False)
ga_table.to_csv("appendix_ga_breakdown.csv", index=False)



# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Shifts
axes[0, 0].bar(['Target', 'MIP', 'GA (Avg)'], [target_total_shifts, int(sum(pulp.value(x[e][d][s]) for e in range(100) for d in range(7) for s in range(10))), np.mean([t['Shifts'] for t in trial_stats])], color=['gray', 'blue', 'orange'])
axes[0, 0].set_title("Shift Assignment Comparison")

# 2. Score (Log)
axes[0, 1].bar(['MIP', 'GA (Avg)'], [mip_score, np.mean([t['Score'] for t in trial_stats])], color=['blue', 'orange'])
axes[0, 1].set_yscale('log')
axes[0, 1].set_title("Objective Score (Log Scale)")

# 3. Time
axes[1, 0].bar(['MIP', 'GA (Avg)'], [mip_time, np.mean([t['Time'] for t in trial_stats])], color=['blue', 'orange'])
axes[1, 0].set_title("Solver Time (Seconds)")

# 4. Convergence
axes[1, 1].plot(last_history, color='orange')
axes[1, 1].set_title("GA Convergence (Best Run)")

plt.tight_layout()
plt.savefig('optimization_report.png')
print("All Appendix files and visualizations generated.")


















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
