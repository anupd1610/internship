import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Set seed for exact statistical reproducibility
np.random.seed(42)

# Generate synthetic dataset simulating customer purchasing & web platform engagement
n = 500
data = {
    'Customer_ID': range(1, n + 1),
    'UI_Version': np.random.choice(['Control_A', 'Variant_B'], size=n, p=[0.5, 0.5]),
    'Customer_Segment': np.random.choice(['Basic', 'Premium', 'VIP'], size=n, p=[0.5, 0.3, 0.2]),
    'Device_Type': np.random.choice(['Desktop', 'Mobile', 'Tablet'], size=n, p=[0.4, 0.5, 0.1]),
    'Conversion': np.random.choice([0, 1], size=n, p=[0.8, 0.2])
}

df = pd.DataFrame(data)

# Inject statistically significant differences for testing
# 1. Purchase Amount (t-test): Variant_B spends $12 more on average
df['Purchase_Amount'] = np.where(
    df['UI_Version'] == 'Variant_B',
    np.random.normal(loc=112, scale=25, size=n),
    np.random.normal(loc=100, scale=25, size=n)
)

# 2. Session Duration (ANOVA): VIPs stay longer than Premium, who stay longer than Basic
duration_map = {'Basic': 15, 'Premium': 22, 'VIP': 30}
df['Session_Duration'] = df['Customer_Segment'].map(lambda x: np.random.normal(duration_map[x], 5))

# ---------------------------------------------------------
# Test 1: Two-Sample Independent t-Test (UI_Version vs. Purchase Amount)
# H0: Mean purchase amount for Control_A == Variant_B
# H1: Mean purchase amount for Control_A != Variant_B
# ---------------------------------------------------------
group_a = df[df['UI_Version'] == 'Control_A']['Purchase_Amount']
group_b = df[df['UI_Version'] == 'Variant_B']['Purchase_Amount']

t_stat, p_val_ttest = stats.ttest_ind(group_a, group_b)

plt.figure(figsize=(8, 5))
sns.boxplot(x='UI_Version', y='Purchase_Amount', data=df, palette='Set2')
plt.title(f't-Test: Purchase Amount by UI Version\n(t-stat: {t_stat:.2f}, p-value: {p_val_ttest:.4e})', fontsize=12, fontweight='bold')
plt.xlabel('UI Design Version')
plt.ylabel('Purchase Amount ($)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('stat_test1_ttest.png')
plt.close()

# ---------------------------------------------------------
# Test 2: Chi-Square Test of Independence (Device Type vs. Conversion)
# H0: Device Type and Conversion status are independent
# H1: Device Type and Conversion status are dependent
# ---------------------------------------------------------
contingency_table = pd.crosstab(df['Device_Type'], df['Conversion'])
chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table)

plt.figure(figsize=(8, 5))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Chi-Square Test: Device Type vs Conversion\n(Chi2: {chi2_stat:.2f}, p-value: {p_val_chi2:.4f})', fontsize=12, fontweight='bold')
plt.xlabel('Converted (0 = No, 1 = Yes)')
plt.ylabel('Device Type')
plt.tight_layout()
plt.savefig('stat_test2_chisq.png')
plt.close()

# ---------------------------------------------------------
# Test 3: One-Way ANOVA (Customer Segment vs. Session Duration)
# H0: Mean session duration is equal across all customer segments
# H1: At least one customer segment has a different mean session duration
# ---------------------------------------------------------
f_stat, p_val_anova = stats.f_oneway(
    df[df['Customer_Segment'] == 'Basic']['Session_Duration'],
    df[df['Customer_Segment'] == 'Premium']['Session_Duration'],
    df[df['Customer_Segment'] == 'VIP']['Session_Duration']
)

plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='Session_Duration', hue='Customer_Segment', fill=True, common_norm=False, palette='crest')
plt.title(f'ANOVA: Session Duration across Customer Segments\n(F-stat: {f_stat:.2f}, p-value: {p_val_anova:.4e})', fontsize=12, fontweight='bold')
plt.xlabel('Session Duration (Minutes)')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('stat_test3_anova.png')
plt.close()

print("Statistical Tests Summary:")
print(f"1. Independent t-Test p-value: {p_val_ttest:.5f}")
print(f"2. Chi-Square Test p-value:    {p_val_chi2:.5f}")
print(f"3. One-Way ANOVA p-value:       {p_val_anova:.5f}")
print("\nAll visualizations generated and saved successfully.")