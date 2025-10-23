"""
Main script demonstrating the full circuit discovery and testing pipeline
Run this to discover factual recall circuits in Gemma 2B
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import json
from datetime import datetime

# Now import from src
from circuit_discovery import CircuitDiscovery, Circuit
from testing_pipeline import CircuitTester, FeatureHypothesis
from utils import (
    visualize_circuit, 
    compare_circuits, 
    export_circuit_summary,
    plot_testing_results,
    plot_circuit_overlap
)


def create_factual_dataset():
    """
    Create a comprehensive dataset of factual recall prompts
    """
    
    dataset = {
        'entity_location': [
            {'clean': 'The Eiffel Tower is located in Paris', 
             'corrupted': 'The Eiffel Tower is located in London'},
            {'clean': 'Mount Everest is in the Himalayas',
             'corrupted': 'Mount Everest is in the Alps'},
            {'clean': 'The Statue of Liberty is in New York',
             'corrupted': 'The Statue of Liberty is in Boston'},
            {'clean': 'The Great Wall is in China',
             'corrupted': 'The Great Wall is in Japan'},
            {'clean': 'The Colosseum is in Rome',
             'corrupted': 'The Colosseum is in Athens'},
        ],
        
        'capital_country': [
            {'clean': 'The capital of France is Paris',
             'corrupted': 'The capital of France is Berlin'},
            {'clean': 'Tokyo is the capital of Japan',
             'corrupted': 'Tokyo is the capital of China'},
            {'clean': 'The capital of Germany is Berlin',
             'corrupted': 'The capital of Germany is Munich'},
            {'clean': 'London is the capital of the UK',
             'corrupted': 'London is the capital of France'},
            {'clean': 'The capital of Italy is Rome',
             'corrupted': 'The capital of Italy is Milan'},
        ],
        
        'historical_date': [
            {'clean': 'World War II ended in 1945',
             'corrupted': 'World War II ended in 1950'},
            {'clean': 'The moon landing was in 1969',
             'corrupted': 'The moon landing was in 1965'},
            {'clean': 'The Declaration of Independence was signed in 1776',
             'corrupted': 'The Declaration of Independence was signed in 1780'},
            {'clean': 'The Berlin Wall fell in 1989',
             'corrupted': 'The Berlin Wall fell in 1985'},
        ],
        
        'person_occupation': [
            {'clean': 'Albert Einstein was a physicist',
             'corrupted': 'Albert Einstein was a chemist'},
            {'clean': 'Shakespeare was a playwright',
             'corrupted': 'Shakespeare was a painter'},
            {'clean': 'Marie Curie was a scientist',
             'corrupted': 'Marie Curie was an artist'},
            {'clean': 'Beethoven was a composer',
             'corrupted': 'Beethoven was a writer'},
        ],
    }
    
    return dataset


def create_custom_hypotheses():
    """
    Create specific hypotheses to test
    """
    
    hypotheses = [
        # Hypothesis 1: Location detector
        FeatureHypothesis(
            feature_id=(8, 100),
            hypothesis="Detects geographical location mentions",
            test_prompts=[
                "Paris is a beautiful city",
                "Tokyo is the largest city in Japan",
                "New York is known for skyscrapers",
                "London has many historic landmarks",
            ],
            control_prompts=[
                "The number seven is prime",
                "Red is a bright color",
                "Python is a programming language",
                "The function returns true",
            ],
            expected_activation_threshold=0.4
        ),
        
        # Hypothesis 2: Capital relation detector
        FeatureHypothesis(
            feature_id=(12, 150),
            hypothesis="Detects 'capital of' relationships",
            test_prompts=[
                "Paris is the capital of France",
                "The capital of Japan is Tokyo",
                "Berlin serves as Germany's capital",
                "Rome is the Italian capital",
            ],
            control_prompts=[
                "The river flows to the sea",
                "Mountains rise above the valley",
                "The code runs efficiently",
                "Time passes slowly",
            ],
            expected_activation_threshold=0.5
        ),
        
        # Hypothesis 3: Year/date detector
        FeatureHypothesis(
            feature_id=(10, 200),
            hypothesis="Detects year and date mentions",
            test_prompts=[
                "The year 1945 marked the end of the war",
                "In 2024, technology has advanced",
                "The event occurred in 1969",
                "The treaty was signed in 1776",
            ],
            control_prompts=[
                "The sky is blue today",
                "She walked to the store",
                "The program compiled successfully",
                "Green grass covers the field",
            ],
            expected_activation_threshold=0.45
        ),
        
        # Hypothesis 4: Famous person detector
        FeatureHypothesis(
            feature_id=(14, 180),
            hypothesis="Detects mentions of famous historical figures",
            test_prompts=[
                "Albert Einstein developed relativity",
                "Shakespeare wrote many plays",
                "Marie Curie won Nobel Prizes",
                "Leonardo da Vinci was a polymath",
            ],
            control_prompts=[
                "The function takes two arguments",
                "Water boils at high temperature",
                "The algorithm is efficient",
                "Colors blend on the canvas",
            ],
            expected_activation_threshold=0.5
        ),
    ]
    
    return hypotheses


def main():
    """
    Main pipeline for circuit discovery and validation
    """
    
    print("=" * 80)
    print("FACTUAL RECALL CIRCUIT DISCOVERY IN GEMMA 2B")
    print("=" * 80)
    print()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cpu':
        print("\nWARNING: Running on CPU. This will be slow.")
        print("For faster execution, use a GPU with CUDA support.\n")
    
    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"circuit_discovery_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {run_dir}\n")
    
    # Step 1: Initialize discovery system
    print("Step 1: Initializing circuit discovery system...")
    discovery = CircuitDiscovery(device=device)
    print("✓ Initialization complete\n")
    
    # Step 2: Load dataset
    print("Step 2: Loading factual recall dataset...")
    dataset = create_factual_dataset()
    total_prompts = sum(len(prompts) for prompts in dataset.values())
    print(f"✓ Loaded {len(dataset)} fact types with {total_prompts} total prompts\n")
    
    # Step 3: Discover circuits
    print("Step 3: Discovering circuits...")
    print("This may take several minutes...\n")
    
    circuits = discovery.discover_all_circuits(
        dataset, 
        train_sae=True
    )
    
    print(f"\n✓ Discovered {len(circuits)} circuits\n")
    
    # Step 4: Visualize circuits
    print("Step 4: Visualizing circuits...")
    for i, circuit in enumerate(circuits):
        viz_path = run_dir / f"circuit_{i}_{circuit.name}.png"
        visualize_circuit(circuit, save_path=str(viz_path))
    
    # Create comparison plot
    comp_path = run_dir / "circuit_comparison.png"
    compare_circuits(circuits, save_path=str(comp_path))
    
    # Create overlap plot
    overlap_path = run_dir / "circuit_overlap.png"
    plot_circuit_overlap(circuits, save_path=str(overlap_path))
    
    print("✓ Visualizations created\n")
    
    # Step 5: Run testing pipeline
    print("Step 5: Running automated testing pipeline...")
    tester = CircuitTester(discovery)
    
    validation_results = tester.run_full_validation(
        circuits=circuits,
        test_dataset=dataset,
        output_dir=str(run_dir)
    )
    
    print("✓ Testing complete\n")
    
    # Step 6: Test custom hypotheses
    print("Step 6: Testing custom hypotheses...")
    custom_hypotheses = create_custom_hypotheses()
    
    for hyp in custom_hypotheses:
        print(f"\nTesting: {hyp.hypothesis}")
        try:
            result = tester.validator.test_hypothesis(hyp, verbose=True)
        except Exception as e:
            print(f"  Skipped: {str(e)}")
    
    # Step 7: Export results
    print("\nStep 7: Exporting results...")
    
    # Export circuit summary
    summary_path = run_dir / "circuit_summary.txt"
    export_circuit_summary(circuits, str(summary_path))
    
    # Export circuits as JSON
    circuits_json = []
    for circuit in circuits:
        circuits_json.append({
            'name': circuit.name,
            'fact_type': circuit.fact_type,
            'num_nodes': len(circuit.nodes),
            'num_edges': len(circuit.edges),
            'attribution_score': circuit.attribution_score,
            'nodes': [{'layer': n[0], 'feature': n[1]} for n in circuit.nodes[:50]],
            'edges': [[{'layer': e[0][0], 'feature': e[0][1]}, 
                      {'layer': e[1][0], 'feature': e[1][1]}] 
                     for e in circuit.edges[:50]]
        })
    
    with open(run_dir / "circuits.json", 'w') as f:
        json.dump(circuits_json, f, indent=2)
    
    # Create final report
    report_path = run_dir / "REPORT.md"
    with open(report_path, 'w') as f:
        f.write("# Factual Recall Circuit Discovery Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model:** Gemma 2B\n\n")
        f.write(f"**Device:** {device}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Circuits Discovered:** {len(circuits)}\n")
        f.write(f"- **Fact Types:** {', '.join(dataset.keys())}\n")
        f.write(f"- **Total Prompts Tested:** {total_prompts}\n\n")
        
        f.write("## Discovered Circuits\n\n")
        for i, circuit in enumerate(circuits, 1):
            f.write(f"### {i}. {circuit.name}\n\n")
            f.write(f"- **Fact Type:** {circuit.fact_type}\n")
            f.write(f"- **Nodes:** {len(circuit.nodes)}\n")
            f.write(f"- **Edges:** {len(circuit.edges)}\n")
            f.write(f"- **Attribution Score:** {circuit.attribution_score:.4f}\n\n")
        
        f.write("## Validation Results\n\n")
        for result in validation_results['circuit_results']:
            f.write(f"### {result['name']}\n\n")
            f.write(f"- **Hypothesis Pass Rate:** {result['hypothesis_pass_rate']:.1%}\n")
            f.write(f"- **Average Precision:** {result['avg_precision']:.1%}\n")
            f.write(f"- **Average Specificity:** {result['avg_specificity']:.1%}\n")
            f.write(f"- **Faithfulness Score:** {result['faithfulness']['completeness_score']:.3f}\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `circuit_summary.txt` - Detailed circuit information\n")
        f.write("- `circuits.json` - Machine-readable circuit data\n")
        f.write("- `circuit_*.png` - Individual circuit visualizations\n")
        f.write("- `circuit_comparison.png` - Comparison across circuits\n")
        f.write("- `circuit_overlap.png` - Circuit overlap analysis\n")
        f.write("- `*_results.csv` - Hypothesis testing results\n")
        f.write("- `validation_summary.json` - Complete validation data\n")
    
    print(f"✓ Results exported to {run_dir}\n")
    
    # Summary
    print("=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nDiscovered {len(circuits)} circuits for factual recall in Gemma 2B:")
    for circuit in circuits:
        print(f"  • {circuit.name}: {len(circuit.nodes)} nodes, "
              f"attribution score = {circuit.attribution_score:.3f}")
    
    print(f"\nAll results saved to: {run_dir}")
    print("\nView the REPORT.md file for a complete summary!")
    
    return circuits, validation_results


if __name__ == "__main__":
    circuits, results = main()
