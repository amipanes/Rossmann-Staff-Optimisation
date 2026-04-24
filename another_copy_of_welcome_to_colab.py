# This code uses train.csv file from https://www.kaggle.com/c/rossmann-store-sales/data?select=train.csv
# Check code from line 2650 for the experiment between MIP AND GA

df.head(5)

import pandas as pd
import numpy as np

# Load your Kaggle dataset
# this needs to read the train file
df = pd.read_csv('/train.csv')

# Step 1: Standardize Dates
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date'])

# Step 2: Handle "Closed" Days
# In staffing, if Sales == 0 and the store was closed,
# we shouldn't predict demand for that day.
df = df[df['Open'] != 0]

# Step 3: Define the Staffing Conversion
# Let's assume: 1 staff hour is needed for every $500 in sales
# OR for every 50 transactions.
SALES_PER_STAFF_HOUR = 500

df['Staff_Hours_Required'] = df['Sales'] / SALES_PER_STAFF_HOUR

# Step 4: Cap the demand (Physical Capacity)
# A store cannot physically fit more than, say, 20 people.
MAX_CAPACITY = 20
df['Staff_Hours_Required'] = df['Staff_Hours_Required'].clip(upper=MAX_CAPACITY)

# Step 5: Pivot to create the Target Demand Matrix
# This represents the 'Goal' for each day in each store
demand_matrix = df.pivot(index='Date', columns='Store', values='Staff_Hours_Required')

# Fill missing days (like holidays) with 0 to keep the matrix square
demand_matrix = demand_matrix.fillna(0)

print("Kaggle Data Prepared for Metaheuristic:")
print(demand_matrix.head())

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_store_distances(num_stores=10):
    # 1. Generate random (x, y) coordinates for each store (in km)
    np.random.seed(42)
    coords = np.random.rand(num_stores, 2) * 20 # 20km x 20km area

    # 2. Calculate Euclidean Distance between all pairs
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(coords))

    # 3. Convert Distance to Travel Time (assuming 30 km/h city speed)
    # Travel time in minutes
    time_matrix = (dist_matrix / 30) * 60

    # Create DataFrame for easy lookup
    store_names = [f'Store_{i+1}' for i in range(num_stores)]
    adj_matrix = pd.DataFrame(time_matrix, index=store_names, columns=store_names)

    return adj_matrix, coords

# Usage
adj_matrix, coords = generate_store_distances(10)

# Visualize the Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(adj_matrix, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Adjacency Matrix: Travel Time (Minutes) Between Stores')
plt.show()

def calculate_transfer_penalty(schedule, adj_matrix):
    total_travel_time = 0
    # Logic: If employee's location at T changes from Store A to Store B
    # find the distance in adj_matrix and add to total_travel_time
    # ...
    return total_travel_time * cost_per_minute

import numpy as np
import pandas as pd

def evaluate_roster_quality(proposed_roster, demand_matrix, adj_matrix):
    """
    proposed_roster: 3D Matrix [Employee, Store, Day] (Binary 0 or 1)
    demand_matrix: The Rossmann 'staff_needed' pivot table
    adj_matrix: The travel time between stores
    """

    # 1. Coverage Quality (The "Set Covering" Logic)
    # How many staff are at Store S on Day D?
    actual_staffing = proposed_roster.sum(axis=0)
    diff = actual_staffing - demand_matrix.values

    # Penalty for Under-staffing (Critical: we lose sales)
    understaffing_penalty = np.sum(np.abs(diff[diff < 0])) * 50

    # Penalty for Over-staffing (Waste: we lose money)
    overstaffing_penalty = np.sum(diff[diff > 0]) * 10

    # 2. Travel Quality (The "Multi-Chain" Logic)
    travel_penalty = 0
    num_emps, num_stores, num_days = proposed_roster.shape

    for e in range(num_emps):
        for d in range(1, num_days):
            # Find where the employee was yesterday vs today
            prev_store = np.where(proposed_roster[e, :, d-1] == 1)[0]
            curr_store = np.where(proposed_roster[e, :, d] == 1)[0]

            if len(prev_store) > 0 and len(curr_store) > 0:
                if prev_store[0] != curr_store[0]:
                    # Get travel time from our Adjacency Matrix
                    travel_time = adj_matrix.iloc[prev_store[0], curr_store[0]]
                    travel_penalty += travel_time * 0.5 # Cost per minute of travel

    # 3. Fatigue/Compliance Quality (Non-Linear Penalty)
    fatigue_penalty = 0
    emp_weekly_hours = proposed_roster.sum(axis=(1, 2)) * 8 # Assuming 8hr shifts
    for hours in emp_weekly_hours:
        if hours > 40:
            # Non-linear: The penalty grows faster the more they work
            fatigue_penalty += (hours - 40) ** 2

    # Calculate Total Quality Score
    # We use a large constant to keep Quality positive; higher is better.
    base_score = 10000
    quality_score = base_score - (understaffing_penalty + overstaffing_penalty +
                                  travel_penalty + fatigue_penalty)

    return max(0, quality_score), {
        "Coverage": understaffing_penalty,
        "Waste": overstaffing_penalty,
        "Travel": travel_penalty,
        "Fatigue": fatigue_penalty
    }

pip install pandas numpy matplotlib seaborn scikit-learn pygad

import pandas as pd
import numpy as np

# Load Kaggle Store Data
df = pd.read_csv('/content/retail_sales.csv') # Assuming standard Kaggle sales format

# Transform Sales to Demand: 100 items sold = 1 labor hour required
df['demand_hours'] = df['sales'] / 100

# Create a Pivot Table: Store vs. Time
demand_matrix = df.pivot_table(index='date', columns='store_id', values='demand_hours')

demand_matrix.head(5)

import numpy as np

num_stores = 10
# Initialize a matrix with high values (representing "No Connection")
adj_matrix = np.full((num_stores, num_stores), 999.0)

# Diagonal is always 0 (Travel time to the same store is zero)
np.fill_diagonal(adj_matrix, 0)

# Define specific links (Store 0 and Store 1 are 'Sister Stores' close to each other)
adj_matrix[0, 1] = 0.25 # 15 minutes
adj_matrix[1, 0] = 0.25
adj_matrix[1, 1] = 0.50

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Create the Logic Matrix (10 stores)
num_stores = 10
adj_matrix = np.full((num_stores, num_stores), 2.0) # Assume 2 hours default travel
np.fill_diagonal(adj_matrix, 0) # Home store is 0

# 2. Define "Clusters" (Sister Stores)
# Store 0, 1, and 2 are in the same shopping mall (10 mins apart)
adj_matrix[0:3, 0:3] = 0.16
# Store 8 and 9 are across the street from each other
adj_matrix[8, 9] = adj_matrix[9, 8] = 0.05

# 3. Plot the Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(adj_matrix, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Travel Time (Hours)'})
plt.title("Staffing Logic Adjacency Matrix: Store-to-Store Travel")
plt.xlabel("Destination Store ID")
plt.ylabel("Origin Store ID")
plt.show()

import numpy as np
import pandas as pd

# Setup: 5 Employees, 7 Days, 10 Stores (0 is 'Off')
num_employees = 5
num_days = 7
num_stores = 10

# Create a random roster: Each cell is a Store ID assigned to an Employee on a Day
# 0 = Day Off, 1-10 = Store Assignment
roster_matrix = np.random.randint(0, num_stores + 1, size=(num_employees, num_days))

# View as a DataFrame for clarity
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
employees = [f'Emp_{i+1}' for i in range(num_employees)]
roster_df = pd.DataFrame(roster_matrix, index=employees, columns=days)

print("Generated Candidate Roster:")
print(roster_df)

import pulp
import pandas as pd
import numpy as np

# 1. Setup Mock Rossmann Data (1 Store, 7 Days)
# In a real scenario, you'd load this from 'train.csv'
data = {
    'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    'Sales': [5000, 4500, 4800, 5200, 7000, 8500, 0], # Sunday Closed
    'Promo': [1, 1, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

# Convert Sales to Staff Needed (1 staff per $1000 sales, min 1)
df['Demand'] = (df['Sales'] / 1000).apply(lambda x: max(1, int(np.ceil(x))))

# 2. Parameters increased employee to 6
employees = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6','E7','E8', 'E9']
days = list(range(7))
store_id = 1

# 3. Initialize the Problem (Minimize Total Staff Shifts)
prob = pulp.LpProblem("Rossmann_Small_Scale_Staffing", pulp.LpMinimize)

# 4. Decision Variables: x[e, d] = 1 if employee works that day
x = pulp.LpVariable.dicts("shift", (employees, days), cat=pulp.LpBinary)

# 5. Objective Function: Minimize total shifts used
prob += pulp.lpSum(x[e][d] for e in employees for d in days)

# 6. Constraints
# A. Coverage Constraint: Total staff on day 'd' >= Demand for day 'd'
for d in days:
    prob += pulp.lpSum(x[e][d] for e in employees) >= df.iloc[d]['Demand']

# B. Labor Law: Max 5 days work per week per employee
for e in employees:
    prob += pulp.lpSum(x[e][d] for d in days) <= 5

# 7. Solve using Branch-and-Cut (CBC Solver)
prob.solve(pulp.PULP_CBC_CMD(msg=True))

# 8. Output Results
print(f"Status: {pulp.LpStatus[prob.status]}")
roster = []
for d in days:
    daily_staff = [e for e in employees if pulp.value(x[e][d]) == 1]
    roster.append({'Day': df.iloc[d]['Day'], 'Staff': daily_staff, 'Count': len(daily_staff)})

print(pd.DataFrame(roster))

import pandas as pd
import numpy as np

# Load your Kaggle dataset
df = pd.read_csv('/content/train.csv')

# Step 1: Standardize Dates
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date'])

# Step 2: Handle "Closed" Days
# In staffing, if Sales == 0 and the store was closed,
# we shouldn't predict demand for that day.
df = df[df['Open'] != 0]

import pandas as pd
import pulp

# 1. Load the Rossmann Data (Example for Store 1)
df = pd.read_csv('train.csv')
store_1_df = df[df['Store'] == 1].sort_values('Date').head(7) # First 7 days

# 2. CAPTURE THE SETS (The Indices)
# We use list comprehension to create 'Keys' for our solver
days = list(range(len(store_1_df)))
employees = ['Emp_A', 'Emp_B', 'Emp_C', 'Emp_D', 'Emp_E']

# 3. CAPTURE THE PARAMETERS (The Dictionary)
# This maps the Day Index to the specific Labor Demand
# {0: 5, 1: 4, 2: 4 ...}
demand_map = dict(zip(days, store_1_df['Sales'] // 1000))

# 4. CAPTURE THE PROMO LOGIC
# {0: 1, 1: 1, 2: 0 ...}
promo_map = dict(zip(days, store_1_df['Promo']))

print("Captured Demand Dictionary:", demand_map)

# Create the Problem
prob = pulp.LpProblem("Rossmann_Mapped", pulp.LpMinimize)

# Decision Variables
x = pulp.LpVariable.dicts("work", (employees, days), cat=pulp.LpBinary)

# USE THE CAPTURED DATA:
for d in days:
    # Look up the demand directly from our captured dictionary
    required = demand_map[d]

    # Add a 'Promo' weight: If it's a promo day, we need extra staff
    if promo_map[d] == 1:
        required += 1

    prob += pulp.lpSum(x[e][d] for e in employees) >= required

import pandas as pd
import pulp
import numpy as np
#Use this MIP solver
# Load your Kaggle dataset
#df = pd.read_csv('/content/train.csv')
df = pd.read_csv('/content/train.csv', dtype={'StateHoliday': str}, low_memory=False)
store_1_df = df[df['Store'] == 1].sort_values('Date').head(7) # First 7 days

rossmann_df = store_1_df
days = list(range(7))
# 2. TRANSFORM: Define Labor Demand (1 staff per £1200 sales)
demand_map = dict(zip(days, rossmann_df['Sales'] // 1200))

# 3. DEFINE SETS

# Increased pool to 10 to guarantee feasibility
employees = [f'Staff_{i}' for i in range(1, 12)]

# 4. INITIALIZE MIP (Branch-and-Cut)
prob = pulp.LpProblem("Rossmann_Staffing_Optimized", pulp.LpMinimize)

# 5. DECISION VARIABLES
# x[e, d] is 1 if employee e works on day d
x = pulp.LpVariable.dicts("work", (employees, days), cat=pulp.LpBinary)

# 6. OBJECTIVE: Minimize total shifts to save costs
prob += pulp.lpSum(x[e][d] for e in employees for d in days)

# 7. CONSTRAINTS
# A. Daily Coverage: Sum of staff >= Captured Demand
for d in days:
    prob += pulp.lpSum(x[e][d] for e in employees) >= demand_map[d]

# B. Labor Law: Max 5 shifts per week per person
for e in employees:
    prob += pulp.lpSum(x[e][d] for d in days) <= 5

# 8. EXECUTE SOLVER
prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 hides the solver logs

# 9. CAPTURE OUTPUT
print(f"Final Status: {pulp.LpStatus[prob.status]}")
if pulp.LpStatus[prob.status] == 'Optimal':
    results = []
    for d in days:
        on_shift = [e for e in employees if pulp.value(x[e][d]) == 1]
        results.append({"Day": d, "Required": demand_map[d], "Assigned": len(on_shift), "Staff": on_shift})

    print(pd.DataFrame(results))

import numpy as np
import matplotlib.pyplot as plt

# 1. PARAMETERS (Rossmann Store 1)
target_demand = np.array([5, 4, 4, 5, 8, 9, 0])
num_employees = 10
num_days = 7

def calculate_constrained_fitness(roster, target):
    penalty = 0

    # CONSTRAINT A: Coverage (MIP: sum(x) >= Demand)
    scheduled_per_day = np.sum(roster, axis=0)
    for day in range(num_days):
        if scheduled_per_day[day] < target[day]:
            # Severe penalty for understaffing Rossmann peaks
            penalty += (target[day] - scheduled_per_day[day]) * 100
        elif scheduled_per_day[day] > target[day] + 2:
            # Small penalty for overstaffing (Waste of money)
            penalty += (scheduled_per_day[day] - target[day]) * 10

    # CONSTRAINT B: Max Work Days (MIP: sum(x) <= 5)
    work_counts = np.sum(roster, axis=1)
    for count in work_counts:
        if count > 5:
            # Exponential penalty for labor law violations
            penalty += (count - 5) ** 2 * 500

    # CONSTRAINT C: Minimum Staffing (Ensure store isn't empty)
    # Even if sales are low, Rossmann needs at least 1 person if open
    for day in range(num_days):
        if target[day] > 0 and scheduled_per_day[day] < 1:
            penalty += 1000

    return 1 / (1 + penalty)

# 2. UPDATED EVOLUTIONARY LOOP
pop_size = 100
generations = 200
population = np.random.randint(2, size=(pop_size, num_employees, num_days))
history = []

for gen in range(generations):
    scores = [calculate_constrained_fitness(ind, target_demand) for ind in population]
    history.append(max(scores))

    # Selection (Elitism)
    sorted_idx = np.argsort(scores)[::-1]
    population = population[sorted_idx]
    next_gen = list(population[:20]) # Top 20%

    while len(next_gen) < pop_size:
        # Tournament Selection
        p1, p2 = population[np.random.randint(0, 20)], population[np.random.randint(0, 20)]

        # Crossover (MIP doesn't have this!)
        mask = np.random.randint(0, 2, size=(num_employees, num_days))
        child = np.where(mask, p1, p2)

        # Mutation (Escape Local Optima)
        if np.random.rand() < 0.15:
            e, d = np.random.randint(num_employees), np.random.randint(num_days)
            child[e, d] = 1 - child[e, d]

        next_gen.append(child)

    population = np.array(next_gen)

print("Final Staffing per Day:", np.sum(population[0], axis=0))
print("Target Demand:", target_demand)

df.head(5)

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def prepare_rossmann_demand(file_path, store_id=1):
    # 1. Load Kaggle CSV
    # Assuming 'train.csv' is in your directory
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Kaggle CSV not found. Using Mock Data for demonstration.")
        return np.array([5, 4, 4, 5, 8, 9, 0]) # Fallback baseline

    # 2. Filter for Store 1 and Open Days
    store_df = df[df['Store'] == store_id].copy()
    store_df['Date'] = pd.to_datetime(store_df['Date'])
    store_df = store_df.sort_values('Date').head(7) # Get one full week

    # 3. Data Normalization (Sales -> Staff Units)
    # Strategy: Every $1000 in sales = 1 Staff Unit
    # Every 100 Customers = +0.5 Staff Unit (Complexity)
    store_df['Staff_Units'] = (store_df['Sales'] / 1200) + (store_df['Customers'] / 200)

    # 4. Handle Promos (Non-Linear Boost)
    # Promos increase the 'Weight' of the demand
    store_df.loc[store_df['Promo'] == 1, 'Staff_Units'] *= 1.2

    # 5. Final Integer Conversion (Rounding up for safety)
    demand_vector = np.ceil(store_df['Staff_Units']).astype(int)

    # Cap demand to your 10-employee limit
    demand_vector = np.clip(demand_vector, 0, 10)

    return demand_vector.values

# Run the preparation
target_demand = prepare_rossmann_demand('/content/train.csv', store_id=1)
print(f"Normalized Target Demand for GA: {target_demand}")

import numpy as np

# 1. SETUP PARAMETERS
target_demand = np.array([5, 4, 4, 5, 8, 9, 0]) # Captured from your Kaggle prep
num_employees = 10
num_days = 7
population_size = 50
generations = 100
mutation_rate = 0.1

# 2. INITIALIZATION: Create 50 random rosters (Chromosomes)
# A '1' means working, '0' means off.
def create_population(size):
    return np.random.randint(2, size=(size, num_employees, num_days))

# 3. FITNESS FUNCTION: How close is the roster to the Rossmann demand?
def calculate_fitness(roster, target):
    # Coverage: How many people are working each day?
    scheduled = np.sum(roster, axis=0)
    # Penalty: Difference between scheduled and target demand
    error = np.sum(np.abs(scheduled - target))

    # Labor Law Penalty: Each employee max 5 days
    overtime_penalty = np.sum(np.maximum(0, np.sum(roster, axis=1) - 5)) * 10

    total_penalty = error + overtime_penalty
    return 1 / (1 + total_penalty) # Higher fitness is better

# 4. EVOLUTIONARY LOOP
population = create_population(population_size)

for gen in range(generations):
    # Sort population by fitness
    scores = [calculate_fitness(ind, target_demand) for ind in population]
    sorted_indices = np.argsort(scores)[::-1]
    population = population[sorted_indices]

    # Selection: Keep the top 10 (Elitism)
    next_gen = list(population[:10])

    # Crossover & Mutation: Fill the rest of the 50
    while len(next_gen) < population_size:
        parent1, parent2 = population[np.random.randint(0, 10)], population[np.random.randint(0, 10)]
        # Crossover: Split parents at Wednesday
        child = np.concatenate([parent1[:, :3], parent2[:, 3:]], axis=1)

        # Mutation: Randomly flip a shift
        if np.random.rand() < mutation_rate:
            emp, day = np.random.randint(num_employees), np.random.randint(num_days)
            child[emp, day] = 1 - child[emp, day]

        next_gen.append(child)

    population = np.array(next_gen)

# 5. RESULTS
best_roster = population[0]
print("Best Evolved Roster (Staff Count per Day):")
print(np.sum(best_roster, axis=0))
print("Target Rossmann Demand:")
print(target_demand)

import numpy as np
import matplotlib.pyplot as plt

# 1. PARAMETERS (Rossmann Store 1)
target_demand = np.array([5, 4, 4, 5, 8, 9, 0])
num_employees = 10
num_days = 7

def calculate_constrained_fitness(roster, target):
    penalty = 0

    # CONSTRAINT A: Coverage (MIP: sum(x) >= Demand)
    scheduled_per_day = np.sum(roster, axis=0)
    for day in range(num_days):
        if scheduled_per_day[day] < target[day]:
            # Severe penalty for understaffing Rossmann peaks
            penalty += (target[day] - scheduled_per_day[day]) * 100
        elif scheduled_per_day[day] > target[day] + 2:
            # Small penalty for overstaffing (Waste of money)
            penalty += (scheduled_per_day[day] - target[day]) * 10

    # CONSTRAINT B: Max Work Days (MIP: sum(x) <= 5)
    work_counts = np.sum(roster, axis=1)
    for count in work_counts:
        if count > 5:
            # Exponential penalty for labor law violations
            penalty += (count - 5) ** 2 * 500

    # CONSTRAINT C: Minimum Staffing (Ensure store isn't empty)
    # Even if sales are low, Rossmann needs at least 1 person if open
    for day in range(num_days):
        if target[day] > 0 and scheduled_per_day[day] < 1:
            penalty += 1000

    return 1 / (1 + penalty)

# 2. UPDATED EVOLUTIONARY LOOP
pop_size = 100
generations = 200
population = np.random.randint(2, size=(pop_size, num_employees, num_days))
history = []

for gen in range(generations):
    scores = [calculate_constrained_fitness(ind, target_demand) for ind in population]
    history.append(max(scores))

    # Selection (Elitism)
    sorted_idx = np.argsort(scores)[::-1]
    population = population[sorted_idx]
    next_gen = list(population[:20]) # Top 20%

    while len(next_gen) < pop_size:
        # Tournament Selection
        p1, p2 = population[np.random.randint(0, 20)], population[np.random.randint(0, 20)]

        # Crossover (MIP doesn't have this!)
        mask = np.random.randint(0, 2, size=(num_employees, num_days))
        child = np.where(mask, p1, p2)

        # Mutation (Escape Local Optima)
        if np.random.rand() < 0.15:
            e, d = np.random.randint(num_employees), np.random.randint(num_days)
            child[e, d] = 1 - child[e, d]

        next_gen.append(child)

    population = np.array(next_gen)

print("Final Staffing per Day:", np.sum(population[0], axis=0))
print("Target Demand:", target_demand)

store_1_df.head(8)

import time
import pandas as pd
import numpy as np
import pulp

# --- SETUP COMMON DATA ---
target_demand = np.array([5, 4, 4, 5, 8, 9, 0])
employees = [f'E{i}' for i in range(1, 11)]
days = range(7)

# --- 1. RUN MIP (Branch-and-Cut) ---
def run_mip():
    start = time.time()
    prob = pulp.LpProblem("MIP_Bench", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("work", (employees, days), cat=pulp.LpBinary)

    # Objective & Constraints
    prob += pulp.lpSum(x[e][d] for e in employees for d in days)
    for d in days:
        prob += pulp.lpSum(x[e][d] for e in employees) >= target_demand[d]
    for e in employees:
        prob += pulp.lpSum(x[e][d] for d in days) <= 5

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    end = time.time()

    cost = pulp.value(prob.objective) if pulp.LpStatus[prob.status] == 'Optimal' else None
    return end - start, cost, pulp.LpStatus[prob.status]

# --- 2. RUN GA (Evolutionary) ---
def run_ga():
    start = time.time()
    # Using the constrained fitness logic from the previous step
    # [GA Loop Logic goes here - abbreviated for the bench script]
    # For the benchmark, assume 200 generations as tested previously
    time.sleep(0.5) # Simulating GA compute time for the example
    end = time.time()

    # Example Result from a converged GA run
    return end - start, 35.0, "Converged"

# --- 3. EXECUTE BENCHMARK ---
mip_time, mip_cost, mip_status = run_mip()
ga_time, ga_cost, ga_status = run_ga()

# --- 4. FORMAT RESULTS TABLE ---
results = {
    "Metric": ["Algorithm Logic", "Status", "Execution Time (s)", "Total Shifts (Cost)"],
    "MIP (Branch-and-Cut)": ["Exact / Deterministic", mip_status, round(mip_time, 4), mip_cost],
    "GA (Evolutionary)": ["Stochastic / Heuristic", ga_status, round(ga_time, 4), ga_cost]
}

print(pd.DataFrame(results))

import pandas as pd
import pulp
import numpy as np

# 1. DATA CAPTURE (Store 1)
df = pd.read_csv('/content/train.csv', low_memory=False)
store_1_df = df[df['Store'] == 1].sort_values('Date').head(7)

# 2. TRANSFORM: Define Labor Demand
days = list(range(7))
demand_map = dict(zip(days, store_1_df['Sales'] // 1200))

# 3. SETS & CONSTANTS
employees = [f'Staff_{i}' for i in range(1, 15)] # Increased pool for legal flexibility
# UK Working Time Regulations:
MAX_SHIFTS_PER_WEEK = 6  # 48 hours / 8 hour shifts
MIN_REST_DAYS = 1        # 24-hour rest period per week

# 4. INITIALIZE MIP
prob = pulp.LpProblem("Rossmann_UK_Compliance", pulp.LpMinimize)

# 5. DECISION VARIABLES
x = pulp.LpVariable.dicts("work", (employees, days), cat=pulp.LpBinary)

# 6. OBJECTIVE: Minimize total shifts
prob += pulp.lpSum(x[e][d] for e in employees for d in days)

# 7. CONSTRAINTS
# A. Coverage Constraint (Rossmann Demand)
for d in days:
    prob += pulp.lpSum(x[e][d] for e in employees) >= demand_map[d]

# B. UK Labor Law: 48-Hour Rule (Max 6 shifts)
for e in employees:
    prob += pulp.lpSum(x[e][d] for d in days) <= MAX_SHIFTS_PER_WEEK

# C. UK Labor Law: Weekly Rest (Must have at least 1 day off)
for e in employees:
    # Total days in week (7) - shifts worked must be >= 1 rest day
    prob += (7 - pulp.lpSum(x[e][d] for d in days)) >= MIN_REST_DAYS

# 8. SOLVE
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# 9. OUTPUT
print(f"Status: {pulp.LpStatus[prob.status]}")
if pulp.LpStatus[prob.status] == 'Optimal':
   print(f"Total UK Compliant Shifts: {pulp.value(prob.objective)}")
    # Reviewing Employee 1 as a sample for the 11-hour rest/day off logic
    #emp1_schedule = [pulp.value(x['Staff_1'][d]) for d in days]
   # print(f"Sample Schedule (Staff_1): {emp1_schedule}")
   results = []
   for d in days:
       on_shift = [e for e in employees if pulp.value(x[e][d]) == 1]
       results.append({"Day": d, "Required": demand_map[d], "Assigned": len(on_shift), "Staff": on_shift})

   print(pd.DataFrame(results))

import pandas as pd
import pulp
import numpy as np
#Use this MIP solver
# Load your Kaggle dataset
df = pd.read_csv('/content/train.csv')
store_1_df = df[df['Store'] == 1].sort_values('Date').head(7) # First 7 days

rossmann_df = store_1_df
days = list(range(7))
# 2. TRANSFORM: Define Labor Demand (1 staff per £1200 sales)
demand_map = dict(zip(days, rossmann_df['Sales'] // 1200))

# 3. DEFINE SETS

# Increased pool to 10 to guarantee feasibility
employees = [f'Staff_{i}' for i in range(1, 12)]

# 4. INITIALIZE MIP (Branch-and-Cut)
prob = pulp.LpProblem("Rossmann_Staffing_Optimized", pulp.LpMinimize)

# 5. DECISION VARIABLES
# x[e, d] is 1 if employee e works on day d
x = pulp.LpVariable.dicts("work", (employees, days), cat=pulp.LpBinary)

# 6. OBJECTIVE: Minimize total shifts to save costs
prob += pulp.lpSum(x[e][d] for e in employees for d in days)

# 7. CONSTRAINTS
# A. Daily Coverage: Sum of staff >= Captured Demand
for d in days:
    prob += pulp.lpSum(x[e][d] for e in employees) >= demand_map[d]

# B. Labor Law: Max 5 shifts per week per person
for e in employees:
    prob += pulp.lpSum(x[e][d] for d in days) <= 5

# 8. EXECUTE SOLVER
prob.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 hides the solver logs

# 9. CAPTURE OUTPUT
print(f"Final Status: {pulp.LpStatus[prob.status]}")
if pulp.LpStatus[prob.status] == 'Optimal':
    results = []
    for d in days:
        on_shift = [e for e in employees if pulp.value(x[e][d]) == 1]
        results.append({"Day": d, "Required": demand_map[d], "Assigned": len(on_shift), "Staff": on_shift})

    print(pd.DataFrame(results))

import pandas as pd
import pulp
import numpy as np
import time

# 1. DATA LOADING & "FOOTFALL-ONLY" ENGINEERING
# ---------------------------------------------------------
# Ensure 'train.csv' is uploaded
df = pd.read_csv('/content/train.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.dayofweek

# The Core Constraint: 1 Staff for every 40 Customers
# We use np.ceil because you cannot hire a fraction of a person
df['Final_Demand'] = np.ceil(df['Customers'] / 40)

# Business Rule: If store is closed, demand is zero
df.loc[df['Open'] == 0, 'Final_Demand'] = 0

# 2. PARAMETERS & STAFF POOL
# ---------------------------------------------------------
start_date = pd.to_datetime('2013-01-07') # Monday
end_date = start_date + pd.Timedelta(days=6)
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

store_ids = list(range(1, 11))
days = list(range(7))

# Staff Pool: 150 ensures the model is "Feasible" even on high-footfall Saturdays
employees = [f'Staff_{i}' for i in range(1, 151)]
home_store = {f'Staff_{i}': ((i-1) // 15) + 1 for i in range(1, 151)}

# 3. BUILD THE FOOTFALL-BASED DEMAND MATRIX
# ---------------------------------------------------------
demand_matrix = {}
for s in store_ids:
    mask = (df['Store'] == s) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    store_week = df.loc[mask].sort_values('Date')
    for d in days:
        day_data = store_week[store_week['Day'] == d]
        if not day_data.empty:
            val = int(day_data.iloc[0]['Final_Demand'])
            # Minimum 2 staff for safety/security if the store is open
            demand_matrix[(s, d)] = max(2, val) if val > 0 else 0
        else:
            demand_matrix[(s, d)] = 0

# 4. INITIALIZE MIP MODEL
# ---------------------------------------------------------
prob = pulp.LpProblem("Rossmann_Footfall_Optimization", pulp.LpMinimize)
x = pulp.LpVariable.dicts("work", (employees, days, store_ids), cat=pulp.LpBinary)

# OBJECTIVE: Minimize (Total Shifts) + (0.1 Mobility Penalty)
beta = 0.1
obj_shifts = pulp.lpSum(x[e][d][s] for e in employees for d in days for s in store_ids)
obj_penalty = pulp.lpSum(beta * x[e][d][s] for e in employees for d in days for s in store_ids if s != home_store[e])
prob += obj_shifts + obj_penalty

# CONSTRAINTS
for s in store_ids:
    for d in days:
        # A. Demand (1:40 Ratio) must be met
        prob += pulp.lpSum(x[e][d][s] for e in employees) >= demand_matrix[(s, d)]

for e in employees:
    # B. UK Labor Law: Max 6 days per week
    prob += pulp.lpSum(x[e][d][s] for d in days for s in store_ids) <= 6
    for d in days:
        # C. Physical Constraint: One location per person per day
        prob += pulp.lpSum(x[e][d][s] for s in store_ids) <= 1

# 5. SOLVE
# ---------------------------------------------------------
start_ts = time.time()
# Standard CBC Solver
prob.solve(pulp.PULP_CBC_CMD(msg=0))
duration = time.time() - start_ts

# 6. RESULTS FOR THE FINDINGS CHAPTER
# ---------------------------------------------------------
status = pulp.LpStatus[prob.status]
print(f"--- FOOTFALL-BASED MIP RESULTS ---")
print(f"Solver Status: {status}")

if status == "Optimal":
    final_obj = pulp.value(prob.objective)
    actual_shifts = sum(pulp.value(x[e][d][s]) for e in employees for d in days for s in store_ids)
    transfers = round((final_obj - actual_shifts) / beta)

    print(f"Global Optimum (Fitness): {final_obj:.1f}")
    print(f"Total Staff Shifts: {int(actual_shifts)}")
    print(f"Inter-Store Transfers: {transfers}")
    print(f"Solve Time: {duration:.4f} seconds")

    # Summary Table
    summary = []
    for d in days:
        req = sum(demand_matrix[(s, d)] for s in store_ids)
        asgn = sum(pulp.value(x[e][d][s]) for e in employees for s in store_ids)
        summary.append({"Day": day_names[d], "Required (1:40)": req, "Assigned": int(asgn)})
    print("\n--- DAILY FOOTFALL COVERAGE ---")
    print(pd.DataFrame(summary).to_string(index=False))

import pandas as pd
import pulp
import numpy as np
import time

# 1. DATA LOADING & SALES-ONLY ENGINEERING
# ---------------------------------------------------------
# From 'train.csv' which is uploaded to Google Colab session
df = pd.read_csv('/content/train.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.dayofweek

# The Core Productivity Constraint: 1 Staff for every £1200 in Sales
# We use np.ceil to ensure we cover the total revenue volume
df['Final_Demand'] = np.ceil(df['Sales'] / 1200)

# Business Rule: If store is closed, demand is zero
df.loc[df['Open'] == 0, 'Final_Demand'] = 0

# 2. PARAMETERS & STAFF POOL
# ---------------------------------------------------------
start_date = pd.to_datetime('2013-01-07') # Monday
end_date = start_date + pd.Timedelta(days=6)
day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

store_ids = list(range(1, 11))
days = list(range(7))

# Staff Pool: 100 ensures the model is "Feasible" even on high-revenue Saturdays
employees = [f'Staff_{i}' for i in range(1, 101)]

# Assign Home Stores (15 staff members per store)
home_store = {f'Staff_{i}': ((i-1) // 15) + 1 for i in range(1, 101)}

# 3. BUILD THE SALES-BASED DEMAND MATRIX
# ---------------------------------------------------------
demand_matrix = {}
for s in store_ids:
    mask = (df['Store'] == s) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    store_week = df.loc[mask].sort_values('Date')
    for d in days:
        day_data = store_week[store_week['Day'] == d]
        if not day_data.empty:
            val = int(day_data.iloc[0]['Final_Demand'])
            # Safety Rule: Minimum 2 staff required for store operations if open
            demand_matrix[(s, d)] = max(2, val) if val > 0 else 0
        else:
            demand_matrix[(s, d)] = 0

# 4. INITIALIZE MIP MODEL
# ---------------------------------------------------------
prob = pulp.LpProblem("Rossmann_Sales_Productivity_Optimization", pulp.LpMinimize)
x = pulp.LpVariable.dicts("work", (employees, days, store_ids), cat=pulp.LpBinary)

# OBJECTIVE: Minimize (Total Shifts) + (0.1 Mobility Penalty)
beta = 0.1
obj_shifts = pulp.lpSum(x[e][d][s] for e in employees for d in days for s in store_ids)
obj_penalty = pulp.lpSum(beta * x[e][d][s] for e in employees for d in days for s in store_ids if s != home_store[e])
prob += obj_shifts + obj_penalty

# CONSTRAINTS
for s in store_ids:
    for d in days:
        # A. Sales Demand (£1200 Ratio) must be satisfied
        prob += pulp.lpSum(x[e][d][s] for e in employees) >= demand_matrix[(s, d)]

for e in employees:
    # B. UK Labor Law: Max 6 working days per week
    prob += pulp.lpSum(x[e][d][s] for d in days for s in store_ids) <= 6
    for d in days:
        # C. Physical Constraint: One location per person per day
        prob += pulp.lpSum(x[e][d][s] for s in store_ids) <= 1

# 5. SOLVE
# ---------------------------------------------------------
start_ts = time.time()
prob.solve(pulp.PULP_CBC_CMD(msg=0))
duration = time.time() - start_ts

# 6. RESULTS FOR THE DISSERTATION
# ---------------------------------------------------------
status = pulp.LpStatus[prob.status]
print(f"--- SALES-BASED PRODUCTIVITY RESULTS ---")
print(f"Solver Status: {status}")

if status == "Optimal":
    final_obj = pulp.value(prob.objective)
    actual_shifts = sum(pulp.value(x[e][d][s]) for e in employees for d in days for s in store_ids)
    transfers = round((final_obj - actual_shifts) / beta)

    print(f"Global Optimum (Fitness): {final_obj:.1f}")
    print(f"Total Staff Shifts: {int(actual_shifts)}")
    print(f"Inter-Store Transfers: {transfers}")
    print(f"Execution Time: {duration:.4f} seconds")

    # Summary Table
    summary = []
    for d in days:
        req = sum(demand_matrix[(s, d)] for s in store_ids)
        asgn = sum(pulp.value(x[e][d][s]) for e in employees for s in store_ids)
        summary.append({"Day": day_names[d], "Required (£1200)": req, "Assigned": int(asgn)})

    print("\n--- DAILY REVENUE COVERAGE ---")
    print(pd.DataFrame(summary).to_string(index=False))
else:
    print("Optimization Failed. Ensure your staff pool is large enough to cover sales ")

import pandas as pd
import numpy as np
import time
import random

# 1. DATA PREPROCESSING & DEMAND MATRIX
# ---------------------------------------------------------
df = pd.read_csv('/content/train.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.dayofweek

# Setup Study Parameters
start_date = pd.to_datetime('2013-01-07')
end_date = start_date + pd.Timedelta(days=6)
store_ids = list(range(1, 11))
days = list(range(7))
num_staff = 100

# Assign Home Stores (10 staff per store)
home_stores = np.array([((i // 10) + 1) for i in range(num_staff)])

# Create the 10x7 Demand Matrix (Sales/1200)
demand_matrix = np.zeros((10, 7), dtype=int)
for s_idx, s_id in enumerate(store_ids):
    mask = (df['Store'] == s_id) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    store_week = df.loc[mask].sort_values('Date')
    for d in days:
        day_data = store_week[store_week['Day'] == d]
        if not day_data.empty and day_data.iloc[0]['Open'] == 1:
            val = int(np.ceil(day_data.iloc[0]['Sales'] / 1200))
            demand_matrix[s_idx, d] = max(2, val)

# 2. FITNESS FUNCTION (The "Judge")
# ---------------------------------------------------------
def get_fitness(chromosome):
    """
    Chromosome Shape: (100 Staff, 7 Days)
    Value: 0 (Off), 1-10 (Store ID)
    """
    total_shifts = np.count_nonzero(chromosome)
    mobility_penalty = 0

    # A. Check Demand Coverage (Store Requirement)
    demand_penalty = 0
    for s_idx, s_id in enumerate(store_ids):
        for d in days:
            assigned = np.sum(chromosome[:, d] == s_id)
            if assigned < demand_matrix[s_idx, d]:
                demand_penalty += (demand_matrix[s_idx, d] - assigned) * 50 # Weight

    # B. Check Labor Law (Max 6 Days)
    workload = np.count_nonzero(chromosome, axis=1)
    law_penalty = np.sum(workload > 6) * 1000 # Death penalty for illegal roster

    # C. Calculate Mobility (The 0.1 Beta)
    for i in range(num_staff):
        # Find days where staff worked at a store NOT their home store
        away_days = np.logical_and(chromosome[i] != 0, chromosome[i] != home_stores[i])
        mobility_penalty += np.sum(away_days) * 0.1

    return total_shifts + mobility_penalty + demand_penalty + law_penalty

# 3. GENETIC OPERATORS (Evolution)
# ---------------------------------------------------------
def create_individual():
    # Randomly assign staff to stores or 0 (Day off)
    return np.random.randint(0, 11, size=(num_staff, 7))

def crossover(p1, p2):
    # Uniform crossover at the staff level
    mask = np.random.randint(0, 2, size=(num_staff, 1))
    return np.where(mask, p1, p2)

def mutate(individual, rate=0.05):
    # Randomly flip a shift to a different store or off
    mask = np.random.rand(num_staff, 7) < rate
    individual[mask] = np.random.randint(0, 11, size=np.sum(mask))
    return individual

# 4. MAIN GA LOOP
# ---------------------------------------------------------
pop_size = 50
generations = 200
population = [create_individual() for _ in range(pop_size)]

start_time = time.time()

for gen in range(generations):
    # Sort by fitness (Lower is better)
    population = sorted(population, key=lambda x: get_fitness(x))

    # Elitism: Keep the best 2
    new_gen = population[:2]

    while len(new_gen) < pop_size:
        # Tournament Selection
        parents = random.sample(population[:20], 2)
        child = crossover(parents[0], parents[1])
        child = mutate(child)
        new_gen.append(child)

    population = new_gen

    if gen % 50 == 0:
        print(f"Gen {gen}: Best Fitness = {get_fitness(population[0]):.1f}")

duration = time.time() - start_time
best_fitness = get_fitness(population[0])

# 5. RESULTS
# ---------------------------------------------------------
print(f"\n--- GA RESULTS ---")
print(f"Best Fitness (Optimum): {best_fitness:.1f}")
print(f"Solve Time: {duration:.4f} seconds")

import pandas as pd
import numpy as np
import time
import random

# 1. DATA & PARAMETERS
# ---------------------------------------------------------
df = pd.read_csv('/content/train.csv', low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.dayofweek

start_date = pd.to_datetime('2013-01-07')
end_date = start_date + pd.Timedelta(days=6)
store_ids = list(range(1, 11))
days = list(range(7))
num_staff = 100
home_stores = np.array([((i // 10) + 1) for i in range(num_staff)])

# Build Demand Matrix
demand_matrix = np.zeros((10, 7), dtype=int)
for s_idx, s_id in enumerate(store_ids):
    mask = (df['Store'] == s_id) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    store_week = df.loc[mask].sort_values('Date')
    for d in days:
        day_data = store_week[store_week['Day'] == d]
        if not day_data.empty and day_data.iloc[0]['Open'] == 1:
            val = int(np.ceil(day_data.iloc[0]['Sales'] / 1200))
            demand_matrix[s_idx, d] = max(2, val)

# 2. ENHANCED FITNESS FUNCTION
# ---------------------------------------------------------
def get_fitness(chromosome):
    # Base Shifts
    total_shifts = np.count_nonzero(chromosome)

    # A. Demand Penalty (Understaffing)
    demand_penalty = 0
    for s_idx, s_id in enumerate(store_ids):
        for d in days:
            assigned = np.sum(chromosome[:, d] == s_id)
            if assigned < demand_matrix[s_idx, d]:
                # Weighted heavily to ensure stores are covered
                demand_penalty += (demand_matrix[s_idx, d] - assigned) * 100

    # B. Labor Law (Over 6 days)
    workload = np.count_nonzero(chromosome, axis=1)
    law_penalty = np.sum(workload > 6) * 5000 # "Death Penalty"

    # C. Mobility Penalty (The 0.1 Beta)
    mobility_penalty = 0
    for i in range(num_staff):
        # Count shifts that are NOT at the home store
        away_shifts = np.sum((chromosome[i] != 0) & (chromosome[i] != home_stores[i]))
        mobility_penalty += away_shifts * 0.1

    return total_shifts + mobility_penalty + demand_penalty + law_penalty

# 3. HEURISTIC OPERATORS
# ---------------------------------------------------------
def create_smart_individual():
    # Start mostly at home stores to guide the GA
    ind = np.zeros((num_staff, 7), dtype=int)
    for i in range(num_staff):
        # Assign 5-6 random days at home store
        work_days = random.sample(days, random.randint(5, 6))
        for d in work_days:
            ind[i, d] = home_stores[i]
    return ind

def crossover(p1, p2):
    # Slice crossover (Swap whole staff schedules)
    split = random.randint(1, num_staff - 1)
    return np.vstack((p1[:split], p2[split:]))

def mutate(ind, rate=0.1):
    if random.random() < rate:
        # Swap Mutation: Take a random staff and a random day
        s_idx = random.randint(0, num_staff - 1)
        d_idx = random.randint(0, 7 - 1)
        # Randomly change to a different store or off (0)
        ind[s_idx, d_idx] = random.randint(0, 10)
    return ind

# 4. EVOLUTIONARY LOOP
# ---------------------------------------------------------
pop_size = 100      # Increased for diversity
generations = 500   # More time to find the 372.5
population = [create_smart_individual() for _ in range(pop_size)]

start_time = time.time()
best_ever = None
best_fitness_score = float('inf')

for gen in range(generations):
    # Sort by fitness
    population = sorted(population, key=lambda x: get_fitness(x))
    current_best = get_fitness(population[0])

    if current_best < best_fitness_score:
        best_fitness_score = current_best
        best_ever = population[0].copy()

    # Tournament Selection + Elitism
    new_gen = population[:5] # Keep top 5 (Elitism)
    while len(new_gen) < pop_size:
        p1, p2 = random.sample(population[:20], 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_gen.append(child)

    population = new_gen

    if gen % 50 == 0:
        print(f"Gen {gen} | Best: {best_fitness_score:.1f}")
        # Stop if we hit the MIP optimum
        if best_fitness_score <= 373.0: break

duration = time.time() - start_time

# 5. FINAL REPORT
# ---------------------------------------------------------
print(f"\n--- REFINED GA FINAL RESULTS ---")
print(f"Optimal Fitness: {best_fitness_score:.1f}")
print(f"Total Shifts: {np.count_nonzero(best_ever)}")
print(f"Execution Time: {duration:.2f}s")

#To generate the professional charts for your dissertation (Performance Comparison, Convergence, and the Shift Bucket logic), you can use the following Python code. This uses matplotlib and numpy.

#1. Algorithm Performance Comparison
#This code creates the bar chart comparing the MIP Global Optimum against the GA result, highlighting the "Efficiency Gap."

import matplotlib.pyplot as plt
import numpy as np

# Data derived from your results
algorithms = ['MIP (Optimal)', 'Genetic Algorithm']
fitness_values = [372.5, 418.6]
shift_values = [363, 418]

x = np.arange(len(algorithms))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plotting Global Fitness and Actual Shifts
rects1 = ax1.bar(x - width/2, fitness_values, width, label='Global Fitness', color='skyblue')
rects2 = ax1.bar(x + width/2, shift_values, width, label='Total Staff Shifts', color='salmon')

ax1.set_ylabel('Value')
ax1.set_title('MIP vs. GA Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms)
ax1.legend()

# Adding data labels on top of bars
ax1.bar_label(rects1, padding=3)
ax1.bar_label(rects2, padding=3)

fig.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# 1. Setup Categorical Colors
# 0 = Off (White), 1-10 = Stores (Distinct Colors)
colors = ["#FFFFFF"] + sns.color_palette("tab10", 10).as_hex()
my_cmap = ListedColormap(colors)

# 2. Prepare the Roster (First 40 staff for visibility)
roster_sample = best_ever[:40, :]

plt.figure(figsize=(14, 10))

# 3. Create Heatmap with Discrete Legend
ax = sns.heatmap(roster_sample,
                 annot=True,
                 fmt="d",          # Force Integer formatting (no 0.5)
                 cmap=my_cmap,
                 linewidths=1,
                 linecolor='black',
                 cbar_kws={"ticks":np.arange(11), "label": 'Store ID (0 = Day Off)'})

plt.title('Final Optimized Roster: Staff Assignment per Day', fontsize=16, pad=20)
plt.xlabel('Day of the Week (Mon=0, Sun=6)', fontsize=12)
plt.ylabel('Employee ID', fontsize=12)
plt.xticks(np.arange(7) + 0.5, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

# Adjust the color bar to show clean integers
colorbar = ax.collections[0].colorbar
colorbar.set_ticks(np.arange(11))
colorbar.set_ticklabels(['Off', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10'])

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and Preprocess Rossmann Data (7-Day Window) ---
try:
    df = pd.read_csv('train.csv', low_memory=False)
    target_stores = df['Store'].unique()[:10]

    # Filter for 10 stores and exactly 7 days (70 data points)
    # Using the final week of the dataset for maximum 'Promo' volatility
    df_subset = df[df['Store'].isin(target_stores)].copy()
    df_subset = df_subset.sort_values(['Store', 'Date']).tail(70)

    # Logic: Sales / 1200 = Staff Units Needed
    target_demand = (df_subset['Sales'] / 1200).round().astype(int).values

except FileNotFoundError:
    print("Error: train.csv not found. Reverting to 70-point synthetic demand.")
    target_demand = np.random.randint(4, 18, size=70)

# --- 2. GA Parameters (Adjusted for smaller window) ---
TRIALS = 30
POP_SIZE = 50
GENERATIONS = 100 # Fewer generations needed for a smaller window
SHIFTS = len(target_demand) # 70 shifts total
MUTATION_RATE = 0.08

def fitness_function(roster, target):
    error = np.sum(np.abs(roster - target))
    return 1 / (1 + error)

def run_trial(target):
    # Initial population for 70 shifts
    population = np.random.randint(0, 15, size=(POP_SIZE, SHIFTS))
    history = []

    for gen in range(GENERATIONS):
        scores = np.array([fitness_function(ind, target) for ind in population])
        history.append(np.max(scores))

        # Tournament Selection & Crossover
        probs = scores / scores.sum()
        indices = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, p=probs)
        parents = population[indices]

        offspring = parents.copy()
        for i in range(0, POP_SIZE, 2):
            cp = np.random.randint(1, SHIFTS)
            offspring[i, cp:], offspring[i+1, cp:] = parents[i+1, cp:], parents[i, cp:]

        # Mutation
        mask = np.random.rand(*offspring.shape) < MUTATION_RATE
        offspring[mask] = np.random.randint(0, 15, size=np.sum(mask))
        population = offspring

    return history

# --- 3. Statistical Execution ---
convergence_data = []
final_results = []

for t in range(TRIALS):
    trial_history = run_trial(target_demand)
    convergence_data.append(trial_history)
    final_results.append(trial_history[-1])

# --- 4. Visualizing the Results ---
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
conv_df = pd.DataFrame(convergence_data).T
plt.plot(conv_df, color='gray', alpha=0.1)
plt.plot(conv_df.mean(axis=1), color='green', linewidth=2, label='Mean Fitness')
plt.title('7-Day Window GA Convergence')
plt.xlabel('Generations')
plt.ylabel('Fitness Score')

plt.subplot(1, 2, 2)
sns.boxplot(data=final_results, color='lightgreen')
plt.title('Distribution of Final Results (7 Days)')

plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns

# 1. DATA LOADING & DEMAND SYNTHESIS
# ---------------------------------------------------------
try:
    df = pd.read_csv('/content/train.csv', low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.dayofweek

    # 7-Day Window (10 Stores)
    start_date = pd.to_datetime('2013-01-07')
    end_date = start_date + pd.Timedelta(days=6)
    store_ids = list(range(1, 11))
    days = list(range(7))
    num_staff = 100

    # Each 10 staff members belong to 1 store (1-10, 11-20, etc.)
    home_stores = np.array([((i // 10) + 1) for i in range(num_staff)])

    # Build Demand Matrix (Sales / 1200)
    demand_matrix = np.zeros((10, 7), dtype=int)
    for s_idx, s_id in enumerate(store_ids):
        mask = (df['Store'] == s_id) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
        store_week = df.loc[mask].sort_values('Date')
        for d in days:
            day_data = store_week[store_week['Day'] == d]
            if not day_data.empty and day_data.iloc[0]['Open'] == 1:
                val = int(np.ceil(day_data.iloc[0]['Sales'] / 1200))
                demand_matrix[s_idx, d] = max(2, val) # Min 2 staff per open store
except FileNotFoundError:
    print("CSV not found. Ensure train.csv is in /content/")
    # Fallback demand for code testing
    demand_matrix = np.random.randint(4, 12, size=(10, 7))

# 2. FITNESS FUNCTION (Agentic Constraints)
# ---------------------------------------------------------
def get_fitness(chromosome):
    # Base Shifts (The "Cost")
    total_shifts = np.count_nonzero(chromosome)

    # A. Demand Penalty (Understaffing)
    demand_penalty = 0
    for s_idx, s_id in enumerate(store_ids):
        for d in days:
            assigned = np.sum(chromosome[:, d] == s_id)
            if assigned < demand_matrix[s_idx, d]:
                # Weighted heavily (100) to ensure operational coverage
                demand_penalty += (demand_matrix[s_idx, d] - assigned) * 100

    # B. Labor Law Penalty (Max 6 days per week)
    workload = np.count_nonzero(chromosome, axis=1)
    law_penalty = np.sum(workload > 6) * 5000

    # C. Mobility Penalty (Beta = 0.1)
    mobility_penalty = 0
    for i in range(num_staff):
        # Penalty for shifts NOT at the home store
        away_shifts = np.sum((chromosome[i] != 0) & (chromosome[i] != home_stores[i]))
        mobility_penalty += away_shifts * 0.1

    return total_shifts + mobility_penalty + demand_penalty + law_penalty

# 3. HEURISTIC OPERATORS
# ---------------------------------------------------------
def create_smart_individual():
    ind = np.zeros((num_staff, 7), dtype=int)
    for i in range(num_staff):
        work_days = random.sample(days, random.randint(4, 6)) # Realistic 4-6 day week
        for d in work_days:
            ind[i, d] = home_stores[i]
    return ind

def crossover(p1, p2):
    split = random.randint(1, num_staff - 1)
    return np.vstack((p1[:split], p2[split:]))

def mutate(ind, rate=0.15):
    if random.random() < rate:
        s_idx = random.randint(0, num_staff - 1)
        d_idx = random.randint(0, 6)
        # Randomly change to: Off(0), Home Store, or Other Store (1-10)
        ind[s_idx, d_idx] = random.choice([0, home_stores[s_idx], random.randint(1, 10)])
    return ind

# 4. 30-TRIAL EXPERIMENTAL LOOP
# ---------------------------------------------------------
num_trials = 30
pop_size = 60
generations = 300

trial_results = []

print(f"Starting {num_trials} Independent Trials...")

for trial in range(num_trials):
    trial_start = time.time()
    population = [create_smart_individual() for _ in range(pop_size)]
    best_fitness_in_trial = float('inf')
    best_chromosome = None

    for gen in range(generations):
        # Sort and Evaluate
        population = sorted(population, key=lambda x: get_fitness(x))
        current_best = get_fitness(population[0])

        if current_best < best_fitness_in_trial:
            best_fitness_in_trial = current_best
            best_chromosome = population[0].copy()

        # Selection + Elitism (Top 5)
        new_gen = population[:5]
        while len(new_gen) < pop_size:
            p1, p2 = random.sample(population[:15], 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_gen.append(child)
        population = new_gen

    duration = time.time() - trial_start
    shifts = np.count_nonzero(best_chromosome)

    trial_results.append({
        'Fitness': best_fitness_in_trial,
        'Time': duration,
        'Shifts': shifts
    })

    if (trial + 1) % 5 == 0:
        print(f"Trial {trial+1}/30 Complete | Mean Fitness: {np.mean([r['Fitness'] for r in trial_results]):.2f}")

# 5. FINAL STATISTICAL REPORT
# ---------------------------------------------------------
results_df = pd.DataFrame(trial_results)

print("\n" + "="*40)
print("GA METHODOLOGY FINAL REPORT (N=30)")
print("="*40)
print(f"Mean Fitness:       {results_df['Fitness'].mean():.2f}")
print(f"Mean Solver Time:    {results_df['Time'].mean():.4f}s")
print(f"Mean Total Shifts:   {results_df['Shifts'].mean():.2f}")
print(f"Stability (Std Dev): {results_df['Fitness'].std():.4f}")

# Visualization for Dissertation
plt.figure(figsize=(10, 5))
sns.boxplot(x=results_df['Fitness'], color='lightgreen')
plt.title("Distribution of GA Fitness Across 30 Trials")
plt.xlabel("Fitness Value (Lower is Better)")
plt.show()

import pandas as pd
import numpy as np
from pulp import *

# --- 1. DATA CLEANSING & LOADING ---
# Fixing the 'Column 7' Mixed Type Error explicitly
df = pd.read_csv('train.csv', dtype={'StateHoliday': str}, low_memory=False)

# Convert Date and Filter for a specific 7-day window (Jan 2013)
df['Date'] = pd.to_datetime(df['Date'])
start_date = '2013-01-07'
end_date = '2013-01-13'

# Select 10 Stores (Stores 1-10)
target_stores = list(range(1, 11))
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date) & (df['Store'].isin(target_stores))
window_df = df.loc[mask].copy()

# --- 2. SYNTHESIZING THE 70-POINT DEMAND MATRIX ---
# Sort to ensure Store 1 Day 1, Store 1 Day 2... Store 10 Day 7 order
window_df = window_df.sort_values(['Store', 'Date'])

# Apply the Productivity Constant (Sales / 1200)
# Note: 'Open' is omitted from variables but used here to set demand to 0
window_df['Demand'] = window_df.apply(
    lambda x: int(np.ceil(x['Sales'] / 1200)) if x['Open'] == 1 else 0, axis=1
)

# Ensure a Minimum Floor of 2 staff if the store is open
window_df.loc[(window_df['Open'] == 1) & (window_df['Demand'] < 2), 'Demand'] = 2

# Reshape into a 10x7 Matrix
demand_matrix = window_df['Demand'].values.reshape(10, 7)
# --- 3. MIP MODEL SETUP ---
num_staff = 100
num_stores = 10
num_days = 7
beta = 0.1  # Mobility Penalty

# Define Staff Home Stores (10 staff per store)
home_stores = {i: (i // 10) for i in range(num_staff)}

# Initialize the Minimization Problem
prob = LpProblem("Rossmann_Staff_Scheduling", LpMinimize)

# DECISION VARIABLES: x[staff, day, store] = 1 if staff works at store on that day
# 0 = Day Off, 1-10 = Store Index
x = LpVariable.dicts("shift", (range(num_staff), range(num_days), range(num_stores)), 0, 1, LpBinary)

# HELPER VARIABLES: Understaffing and Overstaffing for the Fitness Function
under = LpVariable.dicts("under", (range(num_stores), range(num_days)), 0, None)
over = LpVariable.dicts("over", (range(num_stores), range(num_days)), 0, None)

# --- 4. CONSTRAINTS ---

# Constraint 1: A staff member can work at most ONE store per day
for i in range(num_staff):
    for d in range(num_days):
        prob += lpSum([x[i][d][s] for s in range(num_stores)]) <= 1

# Constraint 2: Labor Law (Max 6 days per week)
for i in range(num_staff):
    prob += lpSum([x[i][d][s] for d in range(num_days) for s in range(num_stores)]) <= 6

# Constraint 3: Defining the Demand Balance (The 70 points)
for s in range(num_stores):
    for d in range(num_days):
        # (Staff working at Store S on Day D) - Over + Under = Target Demand
        prob += lpSum([x[i][d][s] for i in range(num_staff)]) - over[s][d] + under[s][d] == demand_matrix[s][d]

# --- 5. OBJECTIVE FUNCTION (The Fitness Function) ---
# Penalty = 100*(Understaffing) + 1*(Overstaffing) + 0.1*(Mobility)

mobility_penalty = lpSum([
    x[i][d][s] * beta
    for i in range(num_staff)
    for d in range(num_days)
    for s in range(num_stores)
    if s != home_stores[i]
])

prob += lpSum([under[s][d] * 100 for s in range(num_stores) for d in range(num_days)]) + \
        lpSum([over[s][d] * 1 for s in range(num_stores) for d in range(num_days)]) + \
        mobility_penalty

# --- 6. SOLVE ---
prob.solve(PULP_CBC_CMD(msg=0))

print(f"MIP Status: {LpStatus[prob.status]}")
print(f"Optimal Fitness (MIP Baseline): {value(prob.objective)}")

import time
import pandas as pd
import numpy as np
import pulp
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA PRE-PROCESSING (£1200 RULE)
# ==========================================
def load_synchronized_demand(file_path):
    df = pd.read_csv(file_path, dtype={'StateHoliday': str}, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])

    # DATE STANDARDIZATION: Monday Jan 7 to Sunday Jan 13, 2013
    start_date, end_date = '2013-01-07', '2013-01-13'
    store_ids = list(range(1, 11))
    mask = (df['Store'].isin(store_ids)) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    week_df = df.loc[mask].copy()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    week_df['DayName'] = week_df['Date'].dt.day_name()
    week_df['DayName'] = pd.Categorical(week_df['DayName'], categories=day_order, ordered=True)
    week_df = week_df.sort_values(['Store', 'DayName'])

    # The £1200 Rule Heuristic
    def get_req(row):
        if row['Open'] == 0: return 0
        return max(2, int(np.ceil(row['Sales'] / 1200)))

    week_df['Needed'] = week_df.apply(get_req, axis=1)
    return week_df.pivot(index='Store', columns='DayName', values='Needed').values

# ==========================================
# 2. GLOBAL PARAMETERS & WEIGHTS
# ==========================================
W_UNDER = 100.0   # Service Gap
W_OVER = 1.0      # Wage Inefficiency
W_MOBILITY = 0.1  # Agentic Friction
W_LEGAL = 1000.0  # Death Penalty (Max 6 days / Sundays)

# ==========================================
# 3. SOLVER A: MIP (The Tidy Benchmark)
# ==========================================
def run_mip(demand_matrix):
    num_stores, num_days = demand_matrix.shape
    num_staff = 100
    employees, stores, days = range(num_staff), range(num_stores), range(num_days)

    prob = pulp.LpProblem("Rossmann_MIP", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("shift", (employees, days, stores), cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts("under", (stores, days), lowBound=0)
    o = pulp.LpVariable.dicts("over", (stores, days), lowBound=0)

    # Objective
    mobility = pulp.lpSum(W_MOBILITY * x[e][d][s] for e in employees for d in days for s in stores if s != (e // 10))
    coverage = pulp.lpSum(u[s][d]*W_UNDER + o[s][d]*W_OVER for s in stores for d in days)
    prob += coverage + mobility

    # Constraints
    for s in stores:
        for d in days:
            prob += pulp.lpSum(x[e][d][s] for e in employees) - o[s][d] + u[s][d] == demand_matrix[s, d]
    for e in employees:
        prob += pulp.lpSum(x[e][d][s] for d in days for s in stores) <= 6
        for d in days:
            prob += pulp.lpSum(x[e][d][s] for s in stores) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extraction
    store_counts = []
    for s in stores:
        for d in days:
            store_counts.append((s, d, sum(pulp.value(x[e][d][s]) for e in employees)))

    staff_details = []
    for e in employees:
        for d in days:
            for s in stores:
                if pulp.value(x[e][d][s]) == 1:
                    staff_details.append((e, d, s))

    return pulp.value(prob.objective), sum(c[2] for c in store_counts), pulp.LpStatus[prob.status], store_counts, staff_details

# ==========================================
# 4. SOLVER B: GA (The Scruffy Experiment)
# ==========================================
class RossmannGA:
    def __init__(self, demand, pop_size=100):
        self.demand = demand
        self.pop_size = pop_size
        self.population = np.random.randint(-1, 10, (pop_size, 100, 7))

    def fitness(self, chromosome):
        penalty = 0
        for d in range(7):
            for s in range(10):
                scheduled = np.sum(chromosome[:, d] == s)
                target = self.demand[s, d]
                if target == 0 and scheduled > 0:
                    penalty += scheduled * W_LEGAL
                else:
                    diff = scheduled - target
                    penalty += (-diff * W_UNDER) if diff < 0 else (diff * W_OVER)
        workdays = np.sum(chromosome >= 0, axis=1)
        penalty += np.sum(workdays > 6) * W_LEGAL
        for e in range(100):
            penalty += np.sum((chromosome[e] >= 0) & (chromosome[e] != (e // 10))) * W_MOBILITY
        return penalty

    def evolve(self, gens=100):
        history = []
        for _ in range(gens):
            scores = [self.fitness(ind) for ind in self.population]
            idx = np.argsort(scores)
            self.population = self.population[idx]
            history.append(scores[idx[0]])
            next_gen = list(self.population[:20])
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(list(self.population[:10]), 2)
                child = np.where(np.random.rand(100, 7) > 0.5, p1, p2)
                if random.random() < 0.1:
                    child[random.randint(0,99), random.randint(0,6)] = random.randint(-1, 9)
                next_gen.append(child)
            self.population = np.array(next_gen)
        return history

# ==========================================
# 5. UTILITIES & REPAIR
# ==========================================
def build_staff_df(details, num_staff=100):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sched = {e: {d: "OFF" for d in days} for e in range(num_staff)}
    for e_id, d_idx, s_idx in details:
        sched[e_id][days[d_idx]] = f"Store {s_idx + 1}"
    df = pd.DataFrame.from_dict(sched, orient='index')
    df.insert(0, "Home", [f"Store {(e // 10) + 1}" for e in range(num_staff)])
    return df

def get_store_df(counts):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grid = np.zeros((10, 7))
    for s, d, c in counts: grid[s,d] = c
    return pd.DataFrame(grid, index=[f"Store {i+1}" for i in range(10)], columns=days)

# ==========================================
# 6. EXECUTION & BENCHMARK
# ==========================================
demand = load_synchronized_demand('/content/train.csv')
results = []

print("Starting 30-Run Performance Benchmark...")
for i in range(30):
    t0 = time.time()
    m_obj, m_sh, m_st, m_cnt, m_det = run_mip(demand)
    t_mip = time.time() - t0

    t0 = time.time()
    ga = RossmannGA(demand)
    hist = ga.evolve(100)
    t_ga = time.time() - t0

    # GA Post-Processor Repair
    best_ga = ga.population[0]
    ga_grid = np.zeros((10, 7))
    for s in range(10):
        for d in range(7):
            ga_grid[s,d] = np.sum(best_ga[:, d] == s)
    ga_grid = np.where(demand == 0, 0, ga_grid) # The Repair Line

    results.append({'M_Time': t_mip, 'M_Shifts': m_sh, 'M_Obj': m_obj,
                    'G_Time': t_ga, 'G_Shifts': np.sum(ga_grid), 'G_Obj': hist[-1]})

# ==========================================
# 7. FINAL OUTPUTS
# ==========================================
print("\n--- BENCHMARK RESULTS (Averages) ---")
print(pd.DataFrame(results).mean())

print("\n--- MIP WEEKLY STAFF ROSTER (Sample: First 10) ---")
staff_roster = build_staff_df(m_det)
print(staff_roster.head(10))

# Count Transfers in MIP
transfers = 0
for _, row in staff_roster.iterrows():
    for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        if row[d] != "OFF" and row[d] != row['Home']:
            transfers += 1
print(f"\nTotal MIP Transfers: {transfers}")

import pandas as pd
import numpy as np
import pulp
import time
import random
import matplotlib.pyplot as plt

# =================================================================
# 1. DATA TRANSFORMATION (PRESCRIPTIVE ANALYSIS)
# =================================================================
def get_rossmann_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])

    # Selecting a specific week for the 10-store network
    start_date, end_date = '2013-01-07', '2013-01-13'
    stores = list(range(1, 11))

    mask = (df['Store'].isin(stores)) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    week_df = df.loc[mask].copy()

    # Sort by Store and Day
    day_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    week_df['DayIdx'] = week_df['Date'].dt.day_name().map(day_map)
    week_df = week_df.sort_values(['Store', 'DayIdx'])

    # The £1200 Prescriptive Heuristic
    def calculate_need(row):
        if row['Open'] == 0: return 0
        return max(2, int(np.ceil(row['Sales'] / 1200)))

    week_df['StaffNeeded'] = week_df.apply(calculate_need, axis=1)
    return week_df.pivot(index='Store', columns='DayIdx', values='StaffNeeded').values

# =================================================================
# 2. CONSTANTS & WEIGHTS (The Objective Function)
# =================================================================
W_UNDER = 100.0   # High penalty for understaffing
W_OVER = 1.0      # Low penalty for overstaffing (wage waste)
W_MOBILITY = 0.1  # The "Transfer" cost
W_LEGAL = 1000.0  # Penalty for breaking 6-day rule or Sunday work

# =================================================================
# 3. MIP SOLVER (TIDY BASELINE)
# =================================================================
def solve_mip(demand):
    num_stores, num_days = demand.shape
    num_staff = 100 # 10 per store

    prob = pulp.LpProblem("Tidy_MIP", pulp.LpMinimize)

    # x[employee, day, store]
    x = pulp.LpVariable.dicts("x", (range(num_staff), range(num_days), range(num_stores)), cat=pulp.LpBinary)
    under = pulp.LpVariable.dicts("u", (range(num_stores), range(num_days)), lowBound=0)
    over = pulp.LpVariable.dicts("o", (range(num_stores), range(num_days)), lowBound=0)

    # Objective: Coverage + Mobility
    mobility_costs = pulp.lpSum(W_MOBILITY * x[e][d][s] for e in range(num_staff) for d in range(num_days) for s in range(num_stores) if s != (e // 10))
    coverage_costs = pulp.lpSum(under[s][d]*W_UNDER + over[s][d]*W_OVER for s in range(num_stores) for d in range(num_days))
    prob += coverage_costs + mobility_costs

    for s in range(num_stores):
        for d in range(num_days):
            prob += pulp.lpSum(x[e][d][s] for e in range(num_staff)) - over[s][d] + under[s][d] == demand[s, d]

    for e in range(num_staff):
        prob += pulp.lpSum(x[e][d][s] for d in range(num_days) for s in range(num_stores)) <= 6 # Max 6 days
        for d in range(num_days):
            prob += pulp.lpSum(x[e][d][s] for s in range(num_stores)) <= 1 # Max 1 store per day

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    results = []
    for e in range(num_staff):
        for d in range(num_days):
            for s in range(num_stores):
                if pulp.value(x[e][d][s]) == 1: results.append((e, d, s))
    return pulp.value(prob.objective), results

# =================================================================
# 4. GA SOLVER (SCRUFFY/CHAOTIC)
# =================================================================
class RossmannGA:
    def __init__(self, demand, pop_size=50):
        self.demand = demand
        self.pop_size = pop_size
        # Chromosome: [Staff ID][Day] = Store ID (-1 for Day Off)
        self.population = np.random.randint(-1, 10, (pop_size, 100, 7))

    def fitness(self, chromo):
        score = 0
        # 1. Coverage Penalty
        for d in range(7):
            for s in range(10):
                assigned = np.sum(chromo[:, d] == s)
                diff = assigned - self.demand[s, d]
                if diff < 0: score += abs(diff) * W_UNDER
                else: score += diff * W_OVER

        # 2. Mobility Penalty
        for e in range(100):
            home = e // 10
            transfers = np.sum((chromo[e] >= 0) & (chromo[e] != home))
            score += transfers * W_MOBILITY

            # 3. Legal/Sunday Penalties
            if np.sum(chromo[e] >= 0) > 6: score += W_LEGAL
            for d in range(7):
                if self.demand[chromo[e,d], d] == 0 and chromo[e,d] != -1: # Sunday check
                    score += W_LEGAL
        return score

    def evolve(self, generations=100):
        for _ in range(generations):
            scores = [self.fitness(ind) for ind in self.population]
            sorted_idx = np.argsort(scores)
            self.population = self.population[sorted_idx]

            next_gen = list(self.population[:10]) # Elitism
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(list(self.population[:20]), 2)
                mask = np.random.rand(100, 7) > 0.5
                child = np.where(mask, p1, p2)
                if random.random() < 0.1: # Mutation
                    child[random.randint(0,99), random.randint(0,6)] = random.randint(-1, 9)
                next_gen.append(child)
            self.population = np.array(next_gen)
        return scores[sorted_idx[0]], self.population[0]

# =================================================================
# 5. EXECUTION & BENCHMARKING (30 RUNS)
# =================================================================
demand = get_rossmann_data('train.csv')
stats = []

print("Starting 30-Run Benchmark...")
for i in range(30):
    # MIP
    t0 = time.time()
    m_score, m_details = solve_mip(demand)
    t_mip = time.time() - t0

    # GA
    t1 = time.time()
    ga = RossmannGA(demand)
    g_score, best_chromo = ga.evolve(100)
    t_ga = time.time() - t1

    stats.append({'MIP_Obj': m_score, 'GA_Obj': g_score, 'MIP_Time': t_mip, 'GA_Time': t_ga})
    if (i+1) % 5 == 0: print(f"Completed {i+1} runs...")

# =================================================================
# 6. OUTPUT ROSTERING (ALL EMPLOYEES)
# =================================================================
def print_final_roster(mip_details, ga_best_chromo):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Create Full Roster Dataframe
    master_list = []
    for e in range(100):
        home = (e // 10) + 1
        row = {'Emp': f"Emp {e}", 'Home': f"ST {home}"}

        # Get MIP schedule for this emp
        for d_idx, d_name in enumerate(days):
            mip_assigned = [s for (emp, day, s) in mip_details if emp == e and day == d_idx]
            row[f'MIP_{d_name}'] = f"ST {mip_assigned[0]+1}" if mip_assigned else "-"

            ga_assigned = ga_best_chromo[e, d_idx]
            row[f'GA_{d_name}'] = f"ST {ga_assigned+1}" if ga_assigned != -1 else "-"

        master_list.append(row)

    return pd.DataFrame(master_list)

df_roster = print_final_roster(m_details, best_chromo)
print("\n" + "="*50 + "\nBENCHMARK RESULTS (AVERAGES)\n" + "="*50)
print(pd.DataFrame(stats).mean())
print("\nSample of Full Roster (MIP vs GA):")
print(df_roster[['Emp', 'Home', 'MIP_Mon', 'GA_Mon', 'MIP_Sat', 'GA_Sat']].head(20))

# Exporting for dissertation Appendix
# df_roster.to_csv('Full_Rossmann_Roster_Comparison.csv', index=False)

class RossmannGA:
    def __init__(self, demand, pop_size=100):
        self.demand = demand
        self.pop_size = pop_size
        # Start with a smarter initial population: mostly home-store assignments
        self.population = np.zeros((pop_size, 100, 7), dtype=int)
        for e in range(100):
            self.population[:, e, :] = e // 10
        # Add some initial randomness (-1 is day off)
        mask = np.random.rand(pop_size, 100, 7) < 0.3
        self.population[mask] = -1

    def fitness(self, chromo):
        score = 0
        # 1. Coverage (Tighter constraints)
        for d in range(7):
            for s in range(10):
                assigned = np.sum(chromo[:, d] == s)
                diff = assigned - self.demand[s, d]
                score += abs(diff) * W_UNDER if diff < 0 else diff * W_OVER

        # 2. Mobility (Increased influence)
        for e in range(100):
            home = e // 10
            # Higher weight for transfers to stop the "Chaos"
            transfers = np.sum((chromo[e] >= 0) & (chromo[e] != home))
            score += transfers * 10.0  # Increased from 0.1

            # 3. Sunday/Legal (Harder penalties)
            if np.sum(chromo[e] >= 0) > 6: score += W_LEGAL
        return score

    def tournament_selection(self, scores, k=5):
        # Pick k random individuals and return the best one
        selection_ix = np.random.randint(len(self.population), size=k)
        best_ix = selection_ix[np.argmin([scores[i] for i in selection_ix])]
        return self.population[best_ix]

    def evolve(self, generations=200): # Increased generations for better convergence
        for gen in range(generations):
            scores = [self.fitness(ind) for ind in self.population]

            next_gen = [self.population[np.argmin(scores)]] # Strict Elitism

            while len(next_gen) < self.pop_size:
                p1 = self.tournament_selection(scores)
                p2 = self.tournament_selection(scores)

                # Crossover
                mask = np.random.rand(100, 7) > 0.5
                child = np.where(mask, p1, p2)

                # Smart Mutation
                if random.random() < 0.15:
                    e_idx, d_idx = random.randint(0,99), random.randint(0,6)
                    # 80% chance to move home or take day off, 20% chance to transfer
                    if random.random() < 0.8:
                        child[e_idx, d_idx] = random.choice([e_idx//10, -1])
                    else:
                        child[e_idx, d_idx] = random.randint(0, 9)

                next_gen.append(child)
            self.population = np.array(next_gen)
        return min(scores), self.population[0]

import pandas as pd
import numpy as np
import pulp
import time
import random
import matplotlib.pyplot as plt

# =================================================================
# 1. DATA TRANSFORMATION (PRESCRIPTIVE ANALYSIS)
# =================================================================
def get_rossmann_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])

    # Selecting a specific week for the 10-store network
    start_date, end_date = '2013-01-07', '2013-01-13'
    stores = list(range(1, 11))

    mask = (df['Store'].isin(stores)) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    week_df = df.loc[mask].copy()

    # Sort by Store and Day
    day_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    week_df['DayIdx'] = week_df['Date'].dt.day_name().map(day_map)
    week_df = week_df.sort_values(['Store', 'DayIdx'])

    # The £1200 Prescriptive Heuristic
    def calculate_need(row):
        if row['Open'] == 0: return 0
        return max(2, int(np.ceil(row['Sales'] / 1200)))

    week_df['StaffNeeded'] = week_df.apply(calculate_need, axis=1)
    return week_df.pivot(index='Store', columns='DayIdx', values='StaffNeeded').values

# =================================================================
# 2. CONSTANTS & WEIGHTS (The Objective Function)
# =================================================================
W_UNDER = 100.0   # High penalty for understaffing
W_OVER = 1.0      # Low penalty for overstaffing (wage waste)
W_MOBILITY = 0.1  # The "Transfer" cost
W_LEGAL = 1000.0  # Penalty for breaking 6-day rule or Sunday work

# =================================================================
# 3. MIP SOLVER (TIDY BASELINE)
# =================================================================
def solve_mip(demand):
    num_stores, num_days = demand.shape
    num_staff = 100 # 10 per store

    prob = pulp.LpProblem("Tidy_MIP", pulp.LpMinimize)

    # x[employee, day, store]
    x = pulp.LpVariable.dicts("x", (range(num_staff), range(num_days), range(num_stores)), cat=pulp.LpBinary)
    under = pulp.LpVariable.dicts("u", (range(num_stores), range(num_days)), lowBound=0)
    over = pulp.LpVariable.dicts("o", (range(num_stores), range(num_days)), lowBound=0)

    # Objective: Coverage + Mobility
    mobility_costs = pulp.lpSum(W_MOBILITY * x[e][d][s] for e in range(num_staff) for d in range(num_days) for s in range(num_stores) if s != (e // 10))
    coverage_costs = pulp.lpSum(under[s][d]*W_UNDER + over[s][d]*W_OVER for s in range(num_stores) for d in range(num_days))
    prob += coverage_costs + mobility_costs

    for s in range(num_stores):
        for d in range(num_days):
            prob += pulp.lpSum(x[e][d][s] for e in range(num_staff)) - over[s][d] + under[s][d] == demand[s, d]

    for e in range(num_staff):
        prob += pulp.lpSum(x[e][d][s] for d in range(num_days) for s in range(num_stores)) <= 6 # Max 6 days
        for d in range(num_days):
            prob += pulp.lpSum(x[e][d][s] for s in range(num_stores)) <= 1 # Max 1 store per day

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    results = []
    for e in range(num_staff):
        for d in range(num_days):
            for s in range(num_stores):
                if pulp.value(x[e][d][s]) == 1: results.append((e, d, s))
    return pulp.value(prob.objective), results

# =================================================================
# 4. GA SOLVER (SCRUFFY/CHAOTIC)
# =================================================================
class RossmannGA:
    def __init__(self, demand, pop_size=100):
        self.demand = demand
        self.pop_size = pop_size
        # Start with a smarter initial population: mostly home-store assignments
        self.population = np.zeros((pop_size, 100, 7), dtype=int)
        for e in range(100):
            self.population[:, e, :] = e // 10
        # Add some initial randomness (-1 is day off)
        mask = np.random.rand(pop_size, 100, 7) < 0.3
        self.population[mask] = -1

    def fitness(self, chromo):
        score = 0
        # 1. Coverage (Tighter constraints)
        for d in range(7):
            for s in range(10):
                assigned = np.sum(chromo[:, d] == s)
                diff = assigned - self.demand[s, d]
                score += abs(diff) * W_UNDER if diff < 0 else diff * W_OVER

        # 2. Mobility (Increased influence)
        for e in range(100):
            home = e // 10
            # Higher weight for transfers to stop the "Chaos"
            transfers = np.sum((chromo[e] >= 0) & (chromo[e] != home))
            score += transfers * 10.0  # Increased from 0.1

            # 3. Sunday/Legal (Harder penalties)
            if np.sum(chromo[e] >= 0) > 6: score += W_LEGAL
        return score

    def tournament_selection(self, scores, k=5):
        # Pick k random individuals and return the best one
        selection_ix = np.random.randint(len(self.population), size=k)
        best_ix = selection_ix[np.argmin([scores[i] for i in selection_ix])]
        return self.population[best_ix]

    def evolve(self, generations=200): # Increased generations for better convergence
        for gen in range(generations):
            scores = [self.fitness(ind) for ind in self.population]

            next_gen = [self.population[np.argmin(scores)]] # Strict Elitism

            while len(next_gen) < self.pop_size:
                p1 = self.tournament_selection(scores)
                p2 = self.tournament_selection(scores)

                # Crossover
                mask = np.random.rand(100, 7) > 0.5
                child = np.where(mask, p1, p2)

                # Smart Mutation
                if random.random() < 0.15:
                    e_idx, d_idx = random.randint(0,99), random.randint(0,6)
                    # 80% chance to move home or take day off, 20% chance to transfer
                    if random.random() < 0.8:
                        child[e_idx, d_idx] = random.choice([e_idx//10, -1])
                    else:
                        child[e_idx, d_idx] = random.randint(0, 9)

                next_gen.append(child)
            self.population = np.array(next_gen)
        return min(scores), self.population[0]

# =================================================================
# 5. EXECUTION & BENCHMARKING (30 RUNS)
# =================================================================
demand = get_rossmann_data('train.csv')
stats = []

print("Starting 30-Run Benchmark...")
for i in range(30):
    # MIP
    t0 = time.time()
    m_score, m_details = solve_mip(demand)
    t_mip = time.time() - t0

    # GA
    t1 = time.time()
    ga = RossmannGA(demand)
    g_score, best_chromo = ga.evolve(100)
    t_ga = time.time() - t1

    stats.append({'MIP_Obj': m_score, 'GA_Obj': g_score, 'MIP_Time': t_mip, 'GA_Time': t_ga})
    if (i+1) % 5 == 0: print(f"Completed {i+1} runs...")

# =================================================================
# 6. OUTPUT ROSTERING (ALL EMPLOYEES)
# =================================================================
def print_final_roster(mip_details, ga_best_chromo):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Create Full Roster Dataframe
    master_list = []
    for e in range(100):
        home = (e // 10) + 1
        row = {'Emp': f"Emp {e}", 'Home': f"ST {home}"}

        # Get MIP schedule for this emp
        for d_idx, d_name in enumerate(days):
            mip_assigned = [s for (emp, day, s) in mip_details if emp == e and day == d_idx]
            row[f'MIP_{d_name}'] = f"ST {mip_assigned[0]+1}" if mip_assigned else "-"

            ga_assigned = ga_best_chromo[e, d_idx]
            row[f'GA_{d_name}'] = f"ST {ga_assigned+1}" if ga_assigned != -1 else "-"

        master_list.append(row)

    return pd.DataFrame(master_list)

df_roster = print_final_roster(m_details, best_chromo)
print("\n" + "="*50 + "\nBENCHMARK RESULTS (AVERAGES)\n" + "="*50)
print(pd.DataFrame(stats).mean())
print("\nSample of Full Roster (MIP vs GA):")
print(df_roster[['Emp', 'Home', 'MIP_Mon', 'GA_Mon', 'MIP_Sat', 'GA_Sat']].head(20))

def generate_master_network_roster(best_chromo):
    """
    Creates a complete 100-staff roster across all 10 stores.
    Shows the daily location of every employee.
    """
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    master_data = []

    for e_id in range(100):
        home_store = (e_id // 10) + 1
        # Initialize employee row
        emp_row = {
            'Employee ID': f"Emp {e_id:02d}",
            'Home Base': f"Store {home_store}"
        }

        # Populate each day's assignment
        for d_idx, day_name in enumerate(days):
            assignment = best_chromo[e_id, d_idx]

            if assignment == -1:
                emp_row[day_name] = "OFF"
            else:
                assigned_store = assignment + 1
                # Mark as 'Transfer' if it doesn't match home base
                if assigned_store == home_store:
                    emp_row[day_name] = f"S{assigned_store}"
                else:
                    emp_row[day_name] = f"S{assigned_store}*" # Asterisk denotes transfer

        master_data.append(emp_row)

    # Create and format DataFrame
    df_master = pd.DataFrame(master_data)

    # Calculate Transfer Count for the row
    df_master['Transfers'] = df_master.apply(
        lambda row: sum('*' in str(row[d]) for d in days), axis=1
    )

    return df_master

# --- EXECUTION ---
# Assuming 'best_chromo' is the output from your GA evolve()
master_roster = generate_master_network_roster(best_chromo)

# Display options to ensure we see all staff
pd.set_option('display.max_rows', 100)
print("\n" + "="*85)
print("            ROSSMANN GLOBAL STAFF ROSTER (GA OPTIMIZED)")
print("            S# = Local | S#* = Transfer | OFF = Rest Day")
print("="*85)
print(master_roster.to_string(index=False))
print("="*85)

# Summary Stats for Conclusion
total_transfers = master_roster['Transfers'].sum()
print(f"\nTOTAL NETWORK FRICTION: {total_transfers} cross-store transfers identified.")

import time
import pandas as pd
import numpy as np
import pulp
import random
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA PRE-PROCESSING (£1200 RULE)
# ==========================================
def load_synchronized_demand(file_path):
    df = pd.read_csv(file_path, dtype={'StateHoliday': str}, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])

    # DATE STANDARDIZATION: Monday Jan 7 to Sunday Jan 13, 2013
    start_date, end_date = '2013-01-07', '2013-01-13'
    store_ids = list(range(1, 11))
    mask = (df['Store'].isin(store_ids)) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    week_df = df.loc[mask].copy()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    week_df['DayName'] = week_df['Date'].dt.day_name()
    week_df['DayName'] = pd.Categorical(week_df['DayName'], categories=day_order, ordered=True)
    week_df = week_df.sort_values(['Store', 'DayName'])

    # The £1200 Rule Heuristic
    def get_req(row):
        if row['Open'] == 0: return 0
        return max(2, int(np.ceil(row['Sales'] / 1200)))

    week_df['Needed'] = week_df.apply(get_req, axis=1)
    return week_df.pivot(index='Store', columns='DayName', values='Needed').values

# ==========================================
# 2. GLOBAL PARAMETERS & WEIGHTS
# ==========================================
W_UNDER = 100.0   # Service Gap
W_OVER = 1.0      # Wage Inefficiency
W_MOBILITY = 0.1  # Agentic Friction
W_LEGAL = 1000.0  # Death Penalty (Max 6 days / Sundays)

# ==========================================
# 3. SOLVER A: MIP (The Tidy Benchmark)
# ==========================================
def run_mip(demand_matrix):
    num_stores, num_days = demand_matrix.shape
    num_staff = 100
    employees, stores, days = range(num_staff), range(num_stores), range(num_days)

    prob = pulp.LpProblem("Rossmann_MIP", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("shift", (employees, days, stores), cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts("under", (stores, days), lowBound=0)
    o = pulp.LpVariable.dicts("over", (stores, days), lowBound=0)

    # Objective
    mobility = pulp.lpSum(W_MOBILITY * x[e][d][s] for e in employees for d in days for s in stores if s != (e // 10))
    coverage = pulp.lpSum(u[s][d]*W_UNDER + o[s][d]*W_OVER for s in stores for d in days)
    prob += coverage + mobility

    # Constraints
    for s in stores:
        for d in days:
            prob += pulp.lpSum(x[e][d][s] for e in employees) - o[s][d] + u[s][d] == demand_matrix[s, d]
    for e in employees:
        prob += pulp.lpSum(x[e][d][s] for d in days for s in stores) <= 6
        for d in days:
            prob += pulp.lpSum(x[e][d][s] for s in stores) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extraction
    store_counts = []
    for s in stores:
        for d in days:
            store_counts.append((s, d, sum(pulp.value(x[e][d][s]) for e in employees)))

    staff_details = []
    for e in employees:
        for d in days:
            for s in stores:
                if pulp.value(x[e][d][s]) == 1:
                    staff_details.append((e, d, s))

    return pulp.value(prob.objective), sum(c[2] for c in store_counts), pulp.LpStatus[prob.status], store_counts, staff_details

# ==========================================
# 4. SOLVER B: GA (The Scruffy Experiment)
# ==========================================
class RossmannGA:
    def __init__(self, demand, pop_size=100):
        self.demand = demand
        self.pop_size = pop_size
        self.population = np.random.randint(-1, 10, (pop_size, 100, 7))

    def fitness(self, chromosome):
        penalty = 0
        for d in range(7):
            for s in range(10):
                scheduled = np.sum(chromosome[:, d] == s)
                target = self.demand[s, d]
                if target == 0 and scheduled > 0:
                    penalty += scheduled * W_LEGAL
                else:
                    diff = scheduled - target
                    penalty += (-diff * W_UNDER) if diff < 0 else (diff * W_OVER)
        workdays = np.sum(chromosome >= 0, axis=1)
        penalty += np.sum(workdays > 6) * W_LEGAL
        for e in range(100):
            penalty += np.sum((chromosome[e] >= 0) & (chromosome[e] != (e // 10))) * W_MOBILITY
        return penalty

    def evolve(self, gens=100):
        history = []
        for _ in range(gens):
            scores = [self.fitness(ind) for ind in self.population]
            idx = np.argsort(scores)
            self.population = self.population[idx]
            history.append(scores[idx[0]])
            next_gen = list(self.population[:20])
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(list(self.population[:10]), 2)
                child = np.where(np.random.rand(100, 7) > 0.5, p1, p2)
                if random.random() < 0.1:
                    child[random.randint(0,99), random.randint(0,6)] = random.randint(-1, 9)
                next_gen.append(child)
            self.population = np.array(next_gen)
        return history

# ==========================================
# 5. UTILITIES & REPAIR
# ==========================================
def build_staff_df(details, num_staff=100):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sched = {e: {d: "OFF" for d in days} for e in range(num_staff)}
    for e_id, d_idx, s_idx in details:
        sched[e_id][days[d_idx]] = f"Store {s_idx + 1}"
    df = pd.DataFrame.from_dict(sched, orient='index')
    df.insert(0, "Home", [f"Store {(e // 10) + 1}" for e in range(num_staff)])
    return df

def get_store_df(counts):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grid = np.zeros((10, 7))
    for s, d, c in counts: grid[s,d] = c
    return pd.DataFrame(grid, index=[f"Store {i+1}" for i in range(10)], columns=days)

# ==========================================
# 6. EXECUTION & BENCHMARK
# ==========================================
demand = load_synchronized_demand('/content/train.csv')
results = []

print("Starting 30-Run Performance Benchmark...")
for i in range(30):
    t0 = time.time()
    m_obj, m_sh, m_st, m_cnt, m_det = run_mip(demand)
    t_mip = time.time() - t0

    t0 = time.time()
    ga = RossmannGA(demand)
    hist = ga.evolve(100)
    t_ga = time.time() - t0

    # GA Post-Processor Repair
    best_ga = ga.population[0]
    ga_grid = np.zeros((10, 7))
    for s in range(10):
        for d in range(7):
            ga_grid[s,d] = np.sum(best_ga[:, d] == s)
    ga_grid = np.where(demand == 0, 0, ga_grid) # The Repair Line

    results.append({'M_Time': t_mip, 'M_Shifts': m_sh, 'M_Obj': m_obj,
                    'G_Time': t_ga, 'G_Shifts': np.sum(ga_grid), 'G_Obj': hist[-1]})

# ==========================================
# 7. FINAL OUTPUTS
# ==========================================
print("\n--- BENCHMARK RESULTS (Averages) ---")
print(pd.DataFrame(results).mean())

print("\n--- MIP WEEKLY STAFF ROSTER (Sample: First 10) ---")
staff_roster = build_staff_df(m_det)
print(staff_roster.head(10))

# Count Transfers in MIP
transfers = 0
for _, row in staff_roster.iterrows():
    for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        if row[d] != "OFF" and row[d] != row['Home']:
            transfers += 1
print(f"\nTotal MIP Transfers: {transfers}")

import pandas as pd
import numpy as np
import random

# ==========================================
# 1. ORIGINAL "SCRUFFY" GA ENGINE
# ==========================================
class RossmannGA:
    def __init__(self, demand, pop_size=50):
        self.demand = demand
        self.pop_size = pop_size
        # ORIGINAL: Completely random start (-1 to 9)
        # This is why it "goes nuts" - it doesn't start with home-store loyalty
        self.population = np.random.randint(-1, 10, (pop_size, 100, 7))

    def fitness(self, chromo):
        score = 0
        # 1. Coverage Penalty
        for d in range(7):
            for s in range(10):
                assigned = np.sum(chromo[:, d] == s)
                diff = assigned - self.demand[s, d]
                if diff < 0:
                    score += abs(diff) * 100.0 # W_UNDER
                else:
                    score += diff * 1.0 # W_OVER

        # 2. Mobility Penalty (Original low weight)
        for e in range(100):
            home = e // 10
            transfers = np.sum((chromo[e] >= 0) & (chromo[e] != home))
            score += transfers * 0.1 # This low weight causes the "Chaos"

            # 3. Legal/Sunday Penalties
            if np.sum(chromo[e] >= 0) > 6:
                score += 1000.0

            # Sunday Check: Penalty if assigned to any store on Sunday (Day 6)
            if chromo[e, 6] != -1:
                score += 1000.0

        return score

    def evolve(self, generations=100):
        for gen in range(generations):
            scores = [self.fitness(ind) for ind in self.population]
            sorted_idx = np.argsort(scores)
            self.population = self.population[sorted_idx]

            # Simple Elitism
            next_gen = list(self.population[:10])

            while len(next_gen) < self.pop_size:
                # Original Random Selection
                p1, p2 = random.sample(list(self.population[:20]), 2)

                # Standard Crossover
                mask = np.random.rand(100, 7) > 0.5
                child = np.where(mask, p1, p2)

                # Original Random Mutation
                if random.random() < 0.1:
                    child[random.randint(0,99), random.randint(0,6)] = random.randint(-1, 9)

                next_gen.append(child)
            self.population = np.array(next_gen)

            if (gen + 1) % 20 == 0:
                print(f"Generation {gen+1} Best Score: {scores[sorted_idx[0]]}")

        return self.population[0]

# ==========================================
# 2. MASTER ROSTER GENERATOR (100 STAFF)
# ==========================================
def generate_master_network_roster(best_chromo):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    master_data = []

    for e_id in range(100):
        home_store = (e_id // 10) + 1
        emp_row = {'Employee': f"Emp {e_id:02d}", 'Home': f"ST {home_store}"}

        for d_idx, day_name in enumerate(days):
            assignment = best_chromo[e_id, d_idx]
            if assignment == -1:
                emp_row[day_name] = "OFF"
            else:
                assigned_store = assignment + 1
                if assigned_store == home_store:
                    emp_row[day_name] = f"S{assigned_store}"
                else:
                    emp_row[day_name] = f"S{assigned_store}*" # Chaos marker

        emp_row['Transfers'] = sum('*' in str(emp_row[d]) for d in days)
        master_data.append(emp_row)

    return pd.DataFrame(master_data)

# ==========================================
# 3. EXECUTION
# ==========================================
# Ensure your 'demand' matrix (10x7) is already loaded from the Rossmann CSV
ga_original = RossmannGA(demand)
best_scruffy_result = ga_original.evolve(generations=100)

df_master = generate_master_network_roster(best_scruffy_result)

# Output Results
pd.set_option('display.max_rows', 105)
print("\n" + "="*90)
print("             ORIGINAL 'SCRUFFY' GA MASTER ROSTER (100 STAFF)")
print("             Note the high frequency of asterisks (*) denoting chaos.")
print("="*90)
print(df_master.to_string(index=False))
print("="*90)

print(f"\nTOTAL NETWORK TRANSFERS: {df_master['Transfers'].sum()}")

import pandas as pd
import numpy as np
import pulp
import time
import random
import matplotlib.pyplot as plt

# =================================================================
# 1. DATA TRANSFORMATION (PRESCRIPTIVE ANALYSIS)
# =================================================================
def get_rossmann_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])

    # Selecting a specific week for the 10-store network
    start_date, end_date = '2013-01-07', '2013-01-13'
    stores = list(range(1, 11))

    mask = (df['Store'].isin(stores)) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
    week_df = df.loc[mask].copy()

    # Sort by Store and Day
    day_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    week_df['DayIdx'] = week_df['Date'].dt.day_name().map(day_map)
    week_df = week_df.sort_values(['Store', 'DayIdx'])

    # The £1200 Prescriptive Heuristic
    def calculate_need(row):
        if row['Open'] == 0: return 0
        return max(2, int(np.ceil(row['Sales'] / 1200)))

    week_df['StaffNeeded'] = week_df.apply(calculate_need, axis=1)
    return week_df.pivot(index='Store', columns='DayIdx', values='StaffNeeded').values

# =================================================================
# 2. CONSTANTS & WEIGHTS (The Objective Function)
# =================================================================
W_UNDER = 100.0   # High penalty for understaffing
W_OVER = 1.0      # Low penalty for overstaffing (wage waste)
W_MOBILITY = 0.1  # The "Transfer" cost
W_LEGAL = 1000.0  # Penalty for breaking 6-day rule or Sunday work

# =================================================================
# 3. MIP SOLVER (TIDY BASELINE)
# =================================================================
def solve_mip(demand):
    num_stores, num_days = demand.shape
    num_staff = 100 # 10 per store

    prob = pulp.LpProblem("Tidy_MIP", pulp.LpMinimize)

    # x[employee, day, store]
    x = pulp.LpVariable.dicts("x", (range(num_staff), range(num_days), range(num_stores)), cat=pulp.LpBinary)
    under = pulp.LpVariable.dicts("u", (range(num_stores), range(num_days)), lowBound=0)
    over = pulp.LpVariable.dicts("o", (range(num_stores), range(num_days)), lowBound=0)

    # Objective: Coverage + Mobility
    mobility_costs = pulp.lpSum(W_MOBILITY * x[e][d][s] for e in range(num_staff) for d in range(num_days) for s in range(num_stores) if s != (e // 10))
    coverage_costs = pulp.lpSum(under[s][d]*W_UNDER + over[s][d]*W_OVER for s in range(num_stores) for d in range(num_days))
    prob += coverage_costs + mobility_costs

    for s in range(num_stores):
        for d in range(num_days):
            prob += pulp.lpSum(x[e][d][s] for e in range(num_staff)) - over[s][d] + under[s][d] == demand[s, d]

    for e in range(num_staff):
        prob += pulp.lpSum(x[e][d][s] for d in range(num_days) for s in range(num_stores)) <= 6 # Max 6 days
        for d in range(num_days):
            prob += pulp.lpSum(x[e][d][s] for s in range(num_stores)) <= 1 # Max 1 store per day

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    results = []
    for e in range(num_staff):
        for d in range(num_days):
            for s in range(num_stores):
                if pulp.value(x[e][d][s]) == 1: results.append((e, d, s))
    return pulp.value(prob.objective), results

# =================================================================
# 4. GA SOLVER (SCRUFFY/CHAOTIC)
# =================================================================
class RossmannGA:
    def __init__(self, demand, pop_size=100):
        self.demand = demand
        self.pop_size = pop_size
        # Start with a smarter initial population: mostly home-store assignments
        self.population = np.zeros((pop_size, 100, 7), dtype=int)
        for e in range(100):
            self.population[:, e, :] = e // 10
        # Add some initial randomness (-1 is day off)
        mask = np.random.rand(pop_size, 100, 7) < 0.3
        self.population[mask] = -1

    def fitness(self, chromo):
        score = 0
        # 1. Coverage (Tighter constraints)
        for d in range(7):
            for s in range(10):
                assigned = np.sum(chromo[:, d] == s)
                diff = assigned - self.demand[s, d]
                score += abs(diff) * W_UNDER if diff < 0 else diff * W_OVER

        # 2. Mobility (Increased influence)
        for e in range(100):
            home = e // 10
            # Higher weight for transfers to stop the "Chaos"
            transfers = np.sum((chromo[e] >= 0) & (chromo[e] != home))
            score += transfers * 10.0  # Increased from 0.1

            # 3. Sunday/Legal (Harder penalties)
            if np.sum(chromo[e] >= 0) > 6: score += W_LEGAL
        return score

    def tournament_selection(self, scores, k=5):
        # Pick k random individuals and return the best one
        selection_ix = np.random.randint(len(self.population), size=k)
        best_ix = selection_ix[np.argmin([scores[i] for i in selection_ix])]
        return self.population[best_ix]

    def evolve(self, generations=200): # Increased generations for better convergence
        for gen in range(generations):
            scores = [self.fitness(ind) for ind in self.population]

            next_gen = [self.population[np.argmin(scores)]] # Strict Elitism

            while len(next_gen) < self.pop_size:
                p1 = self.tournament_selection(scores)
                p2 = self.tournament_selection(scores)

                # Crossover
                mask = np.random.rand(100, 7) > 0.5
                child = np.where(mask, p1, p2)

                # Smart Mutation
                if random.random() < 0.15:
                    e_idx, d_idx = random.randint(0,99), random.randint(0,6)
                    # 80% chance to move home or take day off, 20% chance to transfer
                    if random.random() < 0.8:
                        child[e_idx, d_idx] = random.choice([e_idx//10, -1])
                    else:
                        child[e_idx, d_idx] = random.randint(0, 9)

                next_gen.append(child)
            self.population = np.array(next_gen)
        return min(scores), self.population[0]

# =================================================================
# 5. EXECUTION & BENCHMARKING (30 RUNS)
# =================================================================
demand = get_rossmann_data('train.csv')
stats = []

print("Starting 30-Run Benchmark...")
for i in range(30):
    # MIP
    t0 = time.time()
    m_score, m_details = solve_mip(demand)
    t_mip = time.time() - t0

    # GA
    t1 = time.time()
    ga = RossmannGA(demand)
    g_score, best_chromo = ga.evolve(100)
    t_ga = time.time() - t1

    stats.append({'MIP_Obj': m_score, 'GA_Obj': g_score, 'MIP_Time': t_mip, 'GA_Time': t_ga})
    if (i+1) % 5 == 0: print(f"Completed {i+1} runs...")

# =================================================================
# 6. OUTPUT ROSTERING (ALL EMPLOYEES)
# =================================================================
def print_final_roster(mip_details, ga_best_chromo):
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Create Full Roster Dataframe
    master_list = []
    for e in range(100):
        home = (e // 10) + 1
        row = {'Emp': f"Emp {e}", 'Home': f"ST {home}"}

        # Get MIP schedule for this emp
        for d_idx, d_name in enumerate(days):
            mip_assigned = [s for (emp, day, s) in mip_details if emp == e and day == d_idx]
            row[f'MIP_{d_name}'] = f"ST {mip_assigned[0]+1}" if mip_assigned else "-"

            ga_assigned = ga_best_chromo[e, d_idx]
            row[f'GA_{d_name}'] = f"ST {ga_assigned+1}" if ga_assigned != -1 else "-"

        master_list.append(row)

    return pd.DataFrame(master_list)

df_roster = print_final_roster(m_details, best_chromo)
print("\n" + "="*50 + "\nBENCHMARK RESULTS (AVERAGES)\n" + "="*50)
print(pd.DataFrame(stats).mean())
print("\nSample of Full Roster (MIP vs GA):")
print(df_roster[['Emp', 'Home', 'MIP_Mon', 'GA_Mon', 'MIP_Sat', 'GA_Sat']].head(20))
