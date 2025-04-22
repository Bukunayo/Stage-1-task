#funct 1
def dna_to_protein(dna_sequence):
    # Convert DNA to RNA by replacing T with U
    rna_sequence = dna_sequence.replace('T', 'U')




    # Codon dictionary mapping RNA triplets to amino acids
    codon_table = {
        'UUU': 'phe', 'UUC': 'phe',
        'UUA': 'Leu', 'UUG': 'Leu'
    }




    # Split RNA sequence into codons (groups of three)
    protein = []
    for i in range(0, len(rna_sequence) - 2, 3):
        codon = rna_sequence[i:i+3]
        amino_acid = codon_table.get(codon, '?')  # '?' for unknown codons
        protein.append(f'{codon} -> {amino_acid}')




    # Join and return the formatted output
    return ' '.join(protein)




# Example usage
dna_sequence = "TTTTTCTTATTG"
print("Protein:", dna_to_protein(dna_sequence))


#func 2


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to generate a single logistic growth curve using the user's formula
def generate_logistic_curve_v2(time_points=100, L=1.0):
    t = np.linspace(0, 24, time_points)  # Time from 0 to 24 hours
    k = np.random.uniform(0.5, 2.0)      # Random growth rate
    x0 = np.random.uniform(4, 12)        # Random midpoint (lag + exponential phase)
   
    # Logistic growth formula using your notation
    population = L / (1 + np.exp(-k * (t - x0)))
   
    return pd.DataFrame({
        'time': t,
        'population': population,
        'k': k,
        'x0': x0
    })


# Generate 100 growth curves
num_curves = 100
growth_curves_v2 = [generate_logistic_curve_v2() for _ in range(num_curves)]
all_curves_v2 = pd.concat([
    df.assign(curve_id=i) for i, df in enumerate(growth_curves_v2)
], ignore_index=True)


# Function to find time when population reaches 80% of carrying capacity
def time_to_80_percent_v2(df, L=1.0):
    threshold = 0.8 * L
    hit = df[df['population'] >= threshold]
    return hit['time'].iloc[0] if not hit.empty else None


# Compute time to reach 80% of carrying capacity for each curve
times_to_80_v2 = all_curves_v2.groupby('curve_id').apply(
    lambda g: time_to_80_percent_v2(g), include_groups=False
).reset_index(name='time_to_80_percent')


# Plot a few sample curves
def plot_sample_curves_v2(data, num_samples=5):
    plt.figure(figsize=(10, 6))
    for curve_id in data['curve_id'].unique()[:num_samples]:
        subset = data[data['curve_id'] == curve_id]
        plt.plot(subset['time'], subset['population'], label=f'Curve {curve_id}')
    plt.xlabel("Time (hrs)")
    plt.ylabel("Population")
    plt.title("Sample Logistic Growth Curves (User Formula)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plot the curves
plot_sample_curves_v2(all_curves_v2)


# Show sample outputs
print("Sample Growth Data:")
print(all_curves_v2.head())


print("\nTime to Reach 80% of Carrying Capacity:")
print(times_to_80_v2.head())




#func 3


def hamming_distance(str1, str2):
    # Pad the shorter string with spaces
    max_len = max(len(str1), len(str2))
    str1 = str1.ljust(max_len)
    str2 = str2.ljust(max_len)




    # Compute Hamming distance
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))




# Inputs
slack_username = "Haleemah"
x_handle = "Haleemah Afolayan"




# Calculate and print
distance = hamming_distance(slack_username, x_handle)
print(f"Hamming Distance between Slack username and X handle: {distance}")
