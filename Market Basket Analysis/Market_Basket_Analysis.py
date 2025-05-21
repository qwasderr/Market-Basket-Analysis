import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import numpy as np
from itertools import combinations
from collections import Counter

# Load the dataset
df = pd.read_csv("Groceries_dataset.csv")

# 1. Data Preprocessing

# Group purchases by customer and date to obtain a list of items per transaction
transactions = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).tolist()

# Use TransactionEncoder to transform the transaction list into a one-hot encoded format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

#print(df.head())
#print(transactions)
#print(te.columns_[:5])
#print(f"Total Transactions: {len(transactions)}")
#print(df_encoded.head())

# 2. Exploratory Data Analysis (EDA)

# Count the occurrence of each item
item_counts = df_encoded.sum().sort_values(ascending=False)

# Plot the top 10 most popular items
plt.figure(figsize=(12, 6))
sns.barplot(x=item_counts[:10].index, y=item_counts[:10].values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Top 10 Most Popular Items")
plt.xlabel("Items")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Build a co-occurrence matrix for frequently bought-together items
basket_data = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list)

# Generate pairs of items that appear together in a transaction
item_pairs = []
for items in basket_data:
    item_pairs.extend(combinations(items, 2))

# Count the frequency of item pairs
pair_counts = Counter(item_pairs)
item_counts = Counter(df["itemDescription"])

# Select the top 10 most popular items
n = 10
top_n_items = [item for item, _ in item_counts.most_common(n)]

# Create a co-occurrence matrix for the top 10 items
co_occurrence_matrix = pd.DataFrame(index=top_n_items, columns=top_n_items).fillna(0)
for (item1, item2), count in pair_counts.items():
    if item1 in top_n_items and item2 in top_n_items:
        co_occurrence_matrix.loc[item1, item2] = count
        co_occurrence_matrix.loc[item2, item1] = count
co_occurrence_matrix = co_occurrence_matrix.astype("int64")

# Display a heatmap of co-occurrence between top 10 items
plt.figure(figsize=(10, 10))
sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", linewidths=0.5, cbar=False, cmap="coolwarm", square=True)
plt.title("Co-occurrence of Top 10 Items")
plt.tight_layout()
plt.show()

# Build an association graph using NetworkX
top_pairs = dict(pair_counts.most_common(80))
G = nx.Graph()

# Add nodes with their frequency
items_set = {item: item_counts[item] for item1, item2 in top_pairs.keys() for item in (item1, item2)}
for node, freq in items_set.items():
    G.add_node(node, frequency=freq)

# Add edges between items
for (item1, item2), weight in top_pairs.items():
    if item1 != item2:
        G.add_edge(item1, item2, weight=weight)

# Graph visualization
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)
node_size = [items_set[node] * 1.4 for node in G.nodes()]
node_colors = [items_set[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis, node_size=node_size, alpha=0.95)
edges = [(u, v) for u, v in G.edges()]
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w / max(weights) * 2 for w in weights], edge_color="gray", alpha=0.9)
label_pos = {node: (x, y + 0.09) for node, (x, y) in pos.items()}
nx.draw_networkx_labels(G, label_pos, font_size=8.5, font_weight="bold", font_color="black")
plt.title("Item Association Graph")
plt.tight_layout()
plt.show()

# 3. Apriori Algorithm for Association Rule Mining

# Find frequent itemsets using the Apriori algorithm
frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
print(f"Generated {len(rules)} association rules")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Filter rules based on confidence and lift
filtered_rules = rules[(rules['confidence'] > 0.04) & (rules['lift'] > 1)]
print(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 4. Visualization of Association Rules

# Scatter plot: Support vs Confidence
plt.figure(figsize=(10, 6))
sns.scatterplot(x=filtered_rules['support'], y=filtered_rules['confidence'], size=filtered_rules['lift'], hue=filtered_rules['lift'], palette='coolwarm', sizes=(20, 200))
plt.title("Support vs Confidence in Association Rules")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.show()

# Scatter plot: Support vs Lift
sns.scatterplot(filtered_rules, x="support", y="lift")
plt.title("Support vs Lift in Association Rules")
plt.xlabel("Support")
plt.ylabel("Lift")
plt.show()

# Build a directed graph for association rules
G = nx.DiGraph()
for _, row in filtered_rules.iterrows():
    antecedents = list(row['antecedents'])
    consequents = list(row['consequents'])
    for antecedent in antecedents:
        for consequent in consequents:
            G.add_edge(antecedent, consequent, weight=row['confidence'])

# Graph visualization settings
degree_dict = dict(G.degree())
node_size = [degree_dict[node] * 100 for node in G.nodes()]
node_colors = [degree_dict[node] for node in G.nodes()]
plt.figure(figsize=(16, 16))
pos = nx.spring_layout(G, seed=41, k=0.55)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.viridis, node_size=node_size, alpha=0.95)
edges = [(u, v) for u, v in G.edges()]
weights = [G[u][v]['weight'] for u, v in edges]
nx.draw_networkx_edges(
    G, pos, edgelist=edges,
    width=[w * 3 for w in weights],
    edge_color="black", alpha=0.9,
    arrowstyle='->', arrowsize=20,
    connectionstyle="arc3,rad=0.2"
)
label_pos = {node: (x, y + 0.07) for node, (x, y) in pos.items()}
nx.draw_networkx_labels(G, label_pos, font_size=6.5, font_weight="bold", font_color="black")
plt.title(f"Directed Network Graph for filtered rules")
plt.axis("off")
plt.show() 
