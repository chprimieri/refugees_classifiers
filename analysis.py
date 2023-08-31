import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Creating a DataFrame from the dataset already prepared
data = pd.read_csv('datasets/data.csv')

# Create a .csv with the colunm description
data.describe().to_csv('datasets/analysis.csv', index=False)

# Create a dataframe only with the top 100 quantity of refugees
data['Refugees'] = pd.to_numeric(data['Refugees'], errors='coerce')
data_100 = data[['Year','Country of origin','Country of asylum','Refugees','Origin Region','Asylum Region','Distance (Km)']].sort_values(
    by=['Refugees'], ascending=False).head(100)
data_100.to_csv('datasets/analysis2.csv', index=False)
data_100.describe().to_csv('datasets/analysis3.csv', index=False)

# Plot a graph with the Refugees by year
plt.figure(figsize=(14, 6))
sns.boxplot(data=data, x="Year", y="Refugees")
plt.yscale("linear")
plt.title("Refugees by Year - Complete Data")
plt.savefig('figures/boxplot_year_refugees.png')

# Plot a graph with the Refugees by year with Origin Region for the top 100
g = sns.relplot(data=data_100, x="Year", y="Refugees", hue='Origin Region')
g.ax.set_title("Refugees by Origin Region - Top 100 migrations")
g.fig.set_figheight(6)
g.fig.set_figwidth(12)
plt.yscale("linear")
plt.savefig('figures/relplot_year_refugees_with_origin_region.png')

# Plot a graph with the Refugees by year with Asylum Region for the top 100
g = sns.relplot(data=data_100, x="Year", y="Refugees", hue='Asylum Region')
g.ax.set_title("Refugees by Asylum Region - Top 100 migrations")
g.fig.set_figheight(6)
g.fig.set_figwidth(12)
plt.yscale("linear")
plt.savefig('figures/relplot_year_refugees_with_asylum_region.png')

# Plot a graph with the Refugees by Distance for the top 100
plt.figure(figsize=(14, 6))
g = sns.relplot(data=data_100, x="Distance (Km)", y="Refugees", kind="line", errorbar="sd")
g.ax.set_title("Refugees by Distance - Top 100 migrations")
g.fig.set_figheight(8)
g.fig.set_figwidth(10)
plt.savefig('figures/relplot_100_distance_refugees.png')

# Plot a graph with the Refugees by Distance for all data
plt.figure(figsize=(14, 6))
g = sns.relplot(data=data, x="Distance (Km)", y="Refugees", kind="line", errorbar="sd")
g.ax.set_title("Refugees by Distance - Complete Data")
g.fig.set_figheight(8)
g.fig.set_figwidth(10)
plt.savefig('figures/relplot_complete_distance_refugees.png')