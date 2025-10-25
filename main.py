import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import json
from datetime import datetime

from circuit_discovery import CircuitDiscovery
from testing_pipeline import CircuitTester, FeatureHypothesis
from utils import (
    visualize_circuit, 
    compare_circuits, 
    export_circuit_summary,
    plot_circuit_overlap
)


def create_factual_dataset():
    return {
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


def create_hypotheses():
    return [
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


def export_results(circuits, validation_results, dataset, run_dir, device):
    export_circuit_summary(circuits, str(run_dir / "circuit_summary.txt"))
    
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
    
    total_prompts = sum(len(prompts) for prompts in dataset.values())
    
    with open(run_dir / "REPORT.md", 'w') as f:
        f.write("# Factual Recall Circuit Discovery Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model:** Gemma 2B\n")
        f.write(f"**Device:** {device}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Circuits discovered: {len(circuits)}\n")
        f.write(f"Fact types: {', '.join(dataset.keys())}\n")
        f.write(f"Total prompts: {total_prompts}\n\n")
        
        f.write("## Circuits\n\n")
        for circuit in circuits:
            f.write(f"### {circuit.name}\n\n")
            f.write(f"- Fact type: {circuit.fact_type}\n")
            f.write(f"- Nodes: {len(circuit.nodes)}\n")
            f.write(f"- Edges: {len(circuit.edges)}\n")
            f.write(f"- Attribution score: {circuit.attribution_score:.4f}\n\n")
        
        f.write("## Validation\n\n")
        for result in validation_results['circuit_results']:
            f.write(f"### {result['name']}\n\n")
            f.write(f"- Hypothesis pass rate: {result['hypothesis_pass_rate']:.1%}\n")
            f.write(f"- Precision: {result['avg_precision']:.1%}\n")
            f.write(f"- Specificity: {result['avg_specificity']:.1%}\n")
            f.write(f"- Faithfulness: {result['faithfulness']['completeness_score']:.3f}\n\n")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path('./outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"circuit_discovery_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    print(f"Device: {device}")
    print(f"Output: {run_dir}\n")
    
    discovery = CircuitDiscovery(device=device)
    dataset = create_factual_dataset()
    
    print("Discovering circuits...")
    circuits = discovery.discover_all_circuits(dataset, train_sae=True)
    print(f"Discovered {len(circuits)} circuits\n")
    
    print("Generating visualizations...")
    for i, circuit in enumerate(circuits):
        viz_path = run_dir / f"circuit_{i}_{circuit.name}.png"
        visualize_circuit(circuit, save_path=str(viz_path))
    
    compare_circuits(circuits, save_path=str(run_dir / "circuit_comparison.png"))
    plot_circuit_overlap(circuits, save_path=str(run_dir / "circuit_overlap.png"))
    
    print("Running validation...")
    tester = CircuitTester(discovery)
    validation_results = tester.run_full_validation(
        circuits=circuits,
        test_dataset=dataset,
        output_dir=str(run_dir)
    )
    
    print("Testing hypotheses...")
    hypotheses = create_hypotheses()
    for hyp in hypotheses:
        try:
            tester.validator.test_hypothesis(hyp, verbose=False)
        except Exception:
            pass
    
    print("Exporting results...")
    export_results(circuits, validation_results, dataset, run_dir, device)
    
    print(f"\nResults saved to {run_dir}")
    
    return circuits, validation_results


if __name__ == "__main__":
    circuits, results = main()
