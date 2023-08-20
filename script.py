import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
from colors import type_colors

"""Prepare Data"""
# Import pokemon dataset
pokemon_data = pd.read_csv("pokemon_data.csv")

# Display basic information about the dataset
print(pokemon_data.head(10))
print(pokemon_data.info())
print(pokemon_data.describe())


"""Clean Data"""
# Replace missing values in 'Type 2' column with 'None'
pokemon_data['Type 2'] = pokemon_data['Type 2'].fillna('None')

# Clean column names
pokemon_data.columns = pokemon_data.columns.str.replace(' ', '_').str.lower()
column_names = {
    'att': 'attack',
    'def': 'defense',
    'spa': 'special_attack',
    'spd': 'special_defense',
    'spe': 'speed',
    'bst': 'base_stat_total',
}
pokemon_data.rename(columns=column_names, inplace=True)

# Test cleaned dataset
print(pokemon_data.info())
print(pokemon_data.columns)


"""Exploring Data"""
# Pokemon's types distribution
count_by_type = pokemon_data.type_1.value_counts()
color_values = [type_colors[i] for i in count_by_type.index]

plt.figure(figsize=(10,6))
bar_chart = plt.bar(count_by_type.index, count_by_type.values, color=color_values) 
plt.title("Count of Pokemons by Type 1")
plt.xlabel("Type 1")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()

for bar in bar_chart:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom', ha='center', fontweight='bold')

plt.show()

# Create a correlation heatmap
stats = ['hp','attack','defense','special_attack','speed','special_defense','base_stat_total']
corr_matrix = pokemon_data[stats].corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix,  cmap='Greens', interpolation='nearest')
plt.colorbar()
plt.xticks(np.arange(len(corr_matrix)), corr_matrix.columns, rotation=20)
plt.yticks(np.arange(len(corr_matrix)), corr_matrix.columns)
plt.title("Correlation Heatmap of Stats")
plt.show()


"""Answer Questions with Data"""
# 1. Exploring Attribute Relationships
correlation_attack_bst = corr_matrix.loc['attack', 'base_stat_total']
plt.figure(figsize=(10, 6))
plt.scatter(pokemon_data.attack, pokemon_data.base_stat_total, alpha=0.7, color='#4A818D')
plt.title("Relationship between Attack and Base Stat Total")
plt.xlabel("Attack")
plt.ylabel("Base Stat Total")
slope, intercept = np.polyfit(pokemon_data.attack, pokemon_data.base_stat_total, 1)
plt.plot(pokemon_data.attack, slope * pokemon_data.attack + intercept, color='red', label=f'Trend Line (Correlation: {correlation_attack_bst:.2f})')
plt.legend()
plt.show()


# 2. Comparing Attack Distribution by Types
plt.figure(figsize=(12, 6))

# Box plot of Attack values by Type 1
plt.boxplot([pokemon_data[pokemon_data['type_1'] == i]['attack'] for i in count_by_type.index], vert=False, labels=count_by_type.index, patch_artist=True)
plt.title("Distribution of Attack Stats by Type 1")
plt.xlabel("Attack")
plt.ylabel("Type 1")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Perform a two-sample t-test between Legendary and Non-Legendary Pokémon
legendary_attack = pokemon_data[pokemon_data['legendary'] == 1]['attack']
non_legendary_attack = pokemon_data[pokemon_data['legendary'] == 0]['attack']
t_stat, p_value = ttest_ind(legendary_attack, non_legendary_attack, equal_var=False)
print("Two-sample t-test between Legendary and Non-Legendary Pokémon:")
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("The difference in mean Attack values between Legendary and Non-Legendary Pokémon is statistically significant.")
else:
    print("There is no statistically significant difference in mean Attack values between Legendary and Non-Legendary Pokémon.")

# 3. Understanding BMI Trends
avg_bmi_by_generation = pokemon_data.groupby('generation')['bmi'].mean()
plt.figure(figsize=(10, 6))
plt.plot(avg_bmi_by_generation.index, avg_bmi_by_generation.values, marker='o', color='#FAD61D')
plt.title("Average BMI Trends across Generations")
plt.xlabel("Generation")
plt.ylabel("Average BMI")
plt.xticks(rotation=45)

for x, y in zip(avg_bmi_by_generation.index, avg_bmi_by_generation.values):
    plt.annotate(f'{y:.0f}', (x, y), textcoords="offset points", xytext=(-15,5), ha='center', color='#3466AF', fontweight='bold')
    
plt.tight_layout()
plt.show()

# 4. Impact of Mega Evolution on Strength
avg_bst_by_mega_evolution = pokemon_data.groupby('mega_evolution')['base_stat_total'].mean()
x_labels = ['No Mega Evolution', 'Mega Evolution']
plt.figure(figsize=(6, 6))
bar_chart2 = plt.bar(avg_bst_by_mega_evolution.index, avg_bst_by_mega_evolution.values, color=['gray', 'purple'])
plt.title("Impact of Mega Evolution on Base Stat Total")
plt.ylabel("Average Base Stat Total")
plt.xticks(avg_bst_by_mega_evolution.index, x_labels, rotation=0)
plt.tight_layout()
for bar in bar_chart2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), ha='center', va='bottom', color=bar.get_facecolor(), fontweight='bold')
plt.show()

# 5. Generational Evolution of Pokémon Attributes
plt.figure(figsize=(16, 24))
stats_to_visualize = ['hp', 'attack', 'defense', 'speed', 'special_attack', 'special_defense']
stat_labels = ['HP', 'Attack', 'Defense', 'Speed', 'Special Attack', 'Special Defense']
generations = sorted(pokemon_data['generation'].unique())

plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, wspace=0.3, hspace=0.5)  

for i, stat in enumerate(stats_to_visualize, 1):
    plt.subplot(3, 2, i)
    stat_values = [pokemon_data[pokemon_data['generation'] == gen][stat].mean() for gen in generations]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(stats_to_visualize)))
    plt.plot(generations, stat_values, label=stat, color=colors[i-1])
    plt.title(f"{stat_labels[i-1]}", fontsize=14, fontweight='bold', color=colors[i-1])
    plt.xlabel("Generation")
    plt.ylabel("Average " + stat_labels[i-1])

plt.suptitle("Generational Evolution of Pokémon Attributes", y=0.98, fontsize=16, fontweight='bold')

plt.show()
