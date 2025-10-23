"""
Utility functions for circuit analysis and visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import networkx as nx
from pathlib import Path

from circuit_discovery import Circuit


def visualize_circuit(circuit: Circuit, 
                     save_path: str = None,
                     figsize: Tuple[int, int] = (12, 8)):
    """
    Visualize a circuit as a network graph
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in circuit.nodes:
        layer, feat = node
        G.add_node(f"L{layer}F{feat}", layer=layer, feature=feat)
    
    # Add edges
    for edge in circuit.edges:
        src, dst = edge
        src_label = f"L{src[0]}F{src[1]}"
        dst_label = f"L{dst[0]}F{dst[1]}"
        G.add_edge(src_label, dst_label)
    
    # Layout
    pos = {}
    layers = {}
    for node in G.nodes():
        layer = G.nodes[node]['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(node)
    
    for layer_idx, nodes in layers.items():
        y_positions = np.linspace(0, 1, len(nodes) + 1)[1:]
        for i, node in enumerate(nodes):
            pos[node] = (layer_idx, y_positions[i])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, 
                          width=2, alpha=0.6, ax=ax)
    
    ax.set_title(f"Circuit: {circuit.name}\n"
                f"Attribution Score: {circuit.attribution_score:.3f}",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Layer", fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Circuit visualization saved to {save_path}")
    
    return fig


def plot_attribution_heatmap(attributions: Dict[int, np.ndarray],
                            save_path: str = None,
                            figsize: Tuple[int, int] = (10, 8)):
    """
    Plot heatmap of attribution scores across layers
    """
    layers = sorted(attributions.keys())
    
    # Stack attributions
    data = np.array([attributions[l] for l in layers])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, cmap='RdYlBu_r', center=0, 
                xticklabels=False, yticklabels=layers,
                cbar_kws={'label': 'Attribution Score'}, ax=ax)
    
    ax.set_title('Attribution Scores Across Layers', fontsize=14, fontweight='bold')
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attribution heatmap saved to {save_path}")
    
    return fig


def plot_feature_activations(activations: Dict[str, List[float]],
                             title: str = "Feature Activations",
                             save_path: str = None,
                             figsize: Tuple[int, int] = (10, 6)):
    """
    Plot feature activation distributions
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    positions = range(len(activations))
    labels = list(activations.keys())
    
    for i, (label, acts) in enumerate(activations.items()):
        ax.violinplot([acts], positions=[i], widths=0.7,
                     showmeans=True, showmedians=True)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Activation Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature activation plot saved to {save_path}")
    
    return fig


def plot_testing_results(results_df,
                        save_path: str = None,
                        figsize: Tuple[int, int] = (12, 6)):
    """
    Visualize hypothesis testing results
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Pass/Fail distribution
    pass_counts = results_df['passed'].value_counts()
    colors = ['#2ecc71' if x else '#e74c3c' for x in pass_counts.index]
    axes[0].bar(['Failed', 'Passed'], [pass_counts.get(False, 0), pass_counts.get(True, 0)],
               color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Hypothesis Test Results', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Precision vs Specificity scatter
    colors = ['#2ecc71' if p else '#e74c3c' for p in results_df['passed']]
    axes[1].scatter(results_df['precision'], results_df['specificity'],
                   c=colors, alpha=0.6, s=100, edgecolor='black')
    axes[1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Specificity threshold')
    axes[1].axvline(x=0.8, color='blue', linestyle='--', alpha=0.5, label='Precision threshold')
    axes[1].set_xlabel('Precision', fontsize=12)
    axes[1].set_ylabel('Specificity', fontsize=12)
    axes[1].set_title('Precision vs Specificity', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Testing results plot saved to {save_path}")
    
    return fig


def compare_circuits(circuits: List[Circuit],
                    save_path: str = None,
                    figsize: Tuple[int, int] = (12, 6)):
    """
    Compare multiple circuits
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    names = [c.name for c in circuits]
    
    # Plot 1: Number of nodes
    node_counts = [len(c.nodes) for c in circuits]
    axes[0].bar(range(len(circuits)), node_counts, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(len(circuits)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylabel('Node Count', fontsize=12)
    axes[0].set_title('Circuit Size', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Attribution scores
    attr_scores = [c.attribution_score for c in circuits]
    axes[1].bar(range(len(circuits)), attr_scores, color='coral', alpha=0.7)
    axes[1].set_xticks(range(len(circuits)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].set_ylabel('Attribution Score', fontsize=12)
    axes[1].set_title('Circuit Importance', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Edge counts
    edge_counts = [len(c.edges) for c in circuits]
    axes[2].bar(range(len(circuits)), edge_counts, color='mediumseagreen', alpha=0.7)
    axes[2].set_xticks(range(len(circuits)))
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].set_ylabel('Edge Count', fontsize=12)
    axes[2].set_title('Circuit Connectivity', fontsize=12, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Circuit comparison plot saved to {save_path}")
    
    return fig


def export_circuit_summary(circuits: List[Circuit], 
                          output_path: str):
    """
    Export detailed circuit summary to text file
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CIRCUIT DISCOVERY SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total circuits discovered: {len(circuits)}\n\n")
        
        for i, circuit in enumerate(circuits, 1):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"CIRCUIT {i}: {circuit.name}\n")
            f.write(f"{'=' * 80}\n\n")
            
            f.write(f"Fact Type: {circuit.fact_type}\n")
            f.write(f"Attribution Score: {circuit.attribution_score:.4f}\n")
            f.write(f"Number of Nodes: {len(circuit.nodes)}\n")
            f.write(f"Number of Edges: {len(circuit.edges)}\n\n")
            
            f.write("Nodes:\n")
            for node in circuit.nodes[:20]:  # Show first 20
                f.write(f"  Layer {node[0]}, Feature {node[1]}\n")
            if len(circuit.nodes) > 20:
                f.write(f"  ... and {len(circuit.nodes) - 20} more\n")
            
            f.write("\nEdges:\n")
            for edge in circuit.edges[:10]:  # Show first 10
                src, dst = edge
                f.write(f"  L{src[0]}F{src[1]} → L{dst[0]}F{dst[1]}\n")
            if len(circuit.edges) > 10:
                f.write(f"  ... and {len(circuit.edges) - 10} more\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Circuit summary exported to {output_path}")


def analyze_circuit_overlap(circuits: List[Circuit]) -> Dict:
    """
    Analyze overlap between circuits
    """
    all_nodes = [set(c.nodes) for c in circuits]
    
    overlap_matrix = np.zeros((len(circuits), len(circuits)))
    
    for i in range(len(circuits)):
        for j in range(len(circuits)):
            if i != j:
                intersection = len(all_nodes[i] & all_nodes[j])
                union = len(all_nodes[i] | all_nodes[j])
                overlap_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Find shared nodes
    shared_nodes = set.intersection(*all_nodes) if all_nodes else set()
    
    return {
        'overlap_matrix': overlap_matrix,
        'shared_nodes': list(shared_nodes),
        'circuit_names': [c.name for c in circuits]
    }


def plot_circuit_overlap(circuits: List[Circuit],
                        save_path: str = None,
                        figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize overlap between circuits
    """
    overlap_info = analyze_circuit_overlap(circuits)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(overlap_info['overlap_matrix'],
                xticklabels=overlap_info['circuit_names'],
                yticklabels=overlap_info['circuit_names'],
                annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': 'Jaccard Similarity'}, ax=ax)
    
    ax.set_title('Circuit Overlap Analysis', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Circuit overlap plot saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Utilities module loaded successfully!")
    print("Available functions:")
    print("  - visualize_circuit()")
    print("  - plot_attribution_heatmap()")
    print("  - plot_feature_activations()")
    print("  - plot_testing_results()")
    print("  - compare_circuits()")
    print("  - export_circuit_summary()")
    print("  - plot_circuit_overlap()")
