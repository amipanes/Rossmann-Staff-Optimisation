import pandas as pd
import numpy as np
import pulp
import time
import random
import matplotlib.pyplot as plt

# =================================================================
# 1. CONSTANTS & WEIGHTS (Objective Function)
# =================================================================
W_UNDER = 100.0   
W_OVER = 1.0      
W_MOBILITY = 0.1  
W_LEGAL = 1000.0  

# =================================================================
# 2. DATA PREPARATION (Adjusted Path)
# =================================================================
def get_final_demand():
    try:
        # PATH ADJUSTED FOR GOOGLE COLAB
        df = pd.read_csv('/content/train.csv', low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'])
        mask = (df['Store'].isin(range(1, 11))) & (df['Date'] >= '2013-01-07') & (df['Date'] <= '2013-01-13')
        week_df = df.loc[mask].copy()
        week_df['DayIdx'] = week_df['Date'].dt.dayofweek
        
        avg_sales = week_df[week_df['DayIdx'] < 6].groupby('Store')['Sales'].mean().to_dict()
        def calculate_need(row):
            sales = row['Sales'] if row['DayIdx'] < 6 else avg_sales.get(row['Store'], 5000)
            return max(2, int(np.ceil(sales / 1200)))
        
        week_df['StaffNeeded'] = week_df.apply(calculate_need, axis=1)
        return week_df.sort_values(['Store', 'DayIdx']).pivot(index='Store', columns='DayIdx', values='StaffNeeded').values
    except Exception as e:
        print(f"Error loading CSV: {e}. Falling back to synthetic.")
        return np.random.randint(2, 6, (10, 7))

demand_matrix = get_final_demand()
target_total_shifts = int(np.sum(demand_matrix))

# =================================================================
# 3. MIP SOLVER (Tidy Baseline)
# =================================================================
print("Solving MIP Model...")
t0_mip = time.time()
prob = pulp.LpProblem("Appendix_MIP", pulp.LpMinimize)
x = pulp.LpVariable.dicts("w", (range(100), range(7), range(10)), cat=pulp.LpBinary)
under = pulp.LpVariable.dicts("u", (range(10), range(7)), lowBound=0)
over = pulp.LpVariable.dicts("o", (range(10), range(7)), lowBound=0)
v = pulp.LpVariable.dicts("v", range(100), cat=pulp.LpBinary)

prob += (
    pulp.lpSum(under[s][d] * W_UNDER for s in range(10) for d in range(7)) +
    pulp.lpSum(over[s][d] * W_OVER for s in range(10) for d in range(7)) +
    pulp.lpSum(x[e][d][s] * W_MOBILITY for e in range(100) for d in range(7) for s in range(10) if s != (e // 10)) +
    pulp.lpSum(v[e] * W_LEGAL for e in range(100))
)

for s in range(10):
    for d in range(7):
        prob += pulp.lpSum(x[e][d][s] for e in range(100)) - over[s][d] + under[s][d] == demand_matrix[s, d]
for e in range(100):
    prob += pulp.lpSum(x[e][d][s] for d in range(7) for s in range(10)) - 6 <= v[e] * 7
    for d in range(7): prob += pulp.lpSum(x[e][d][s] for s in range(10)) <= 1

prob.solve(pulp.PULP_CBC_CMD(msg=0))
mip_time = time.time() - t0_mip
mip_score = pulp.value(prob.objective)

# =================================================================
# 4. GA 30-TRIAL STRESS TEST
# =================================================================
class StandardGA:
    def __init__(self, demand, pop_size=60):
        self.demand = demand
        self.pop_size = pop_size
        self.population = np.random.randint(-1, 10, (pop_size, 100, 7))
        self.history = []

    def fitness(self, chromo):
        score = 0
        for d in range(7):
            for s in range(10):
                assigned = np.sum(chromo[:, d] == s)
                diff = assigned - self.demand[s, d]
                if diff < 0: score += abs(diff) * W_UNDER
                elif diff > 0: score += diff * W_OVER
        for e in range(100):
            work_days = chromo[e] >= 0
            score += np.sum(work_days & (chromo[e] != (e // 10))) * W_MOBILITY
            if np.sum(work_days) > 6: score += W_LEGAL
        return score

    def run(self, gens=150):
        for _ in range(gens):
            scores = [self.fitness(ind) for ind in self.population]
            self.population = self.population[np.argsort(scores)]
            self.history.append(min(scores))
            next_gen = list(self.population[:5])
            while len(next_gen) < self.pop_size:
                p1, p2 = self.population[random.randint(0, 10)], self.population[random.randint(0, 10)]
                child = np.where(np.random.rand(100, 7) > 0.5, p1, p2)
                if random.random() < 0.1: child[random.randint(0, 99), random.randint(0, 6)] = random.randint(-1, 9)
                next_gen.append(child)
            self.population = np.array(next_gen)
        return self.population[0], self.history

trial_stats = []
best_ga_chromo = None
best_ga_score = float('inf')
last_history = []

print("Running 30 GA Trials...")
for i in range(30):
    t_start = time.time()
    ga = StandardGA(demand_matrix)
    score, history = ga.run(150)
    t_total = time.time() - t_start
    
    shifts = np.sum(ga.population[0] != -1)
    violations = sum(1 for e in range(100) if np.sum(ga.population[0][e] >= 0) > 6)
    
    trial_stats.append({'Trial': i+1, 'Score': score, 'Time': t_total, 'Shifts': shifts, 'Violations': violations})
    if score < best_ga_score:
        best_ga_score = score
        best_ga_chromo = ga.population[0]
        last_history = history

# =================================================================
# 5. CSV EXPORT & VISUALS
# =================================================================
# CSV Outputs
pd.DataFrame(trial_stats).to_csv('appendix_ga_trials.csv', index=False)

roster = []
for e in range(100):
    row = {'ID': e, 'Home': (e // 10) + 1}
    for d_idx, d_name in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
        m_s = [s + 1 for s in range(10) if pulp.value(x[e][d_idx][s]) == 1]
        row[f'MIP_{d_name}'] = m_s[0] if m_s else "OFF"
        g_s = best_ga_chromo[e, d_idx]
        row[f'GA_{d_name}'] = g_s + 1 if g_s != -1 else "OFF"
    roster.append(row)
pd.DataFrame(roster).to_csv('appendix_staff_comparison.csv', index=False)

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
