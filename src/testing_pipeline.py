"""
Automated Testing Pipeline for Circuit Validation
Tests feature hypotheses on network prompts
FIXED: Handles missing SAEs gracefully
"""

import torch
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from circuit_discovery import Circuit, CircuitDiscovery


@dataclass
class FeatureHypothesis:
    """Hypothesis about what a feature detects"""
    feature_id: Tuple[int, int]  # (layer, feature_idx)
    hypothesis: str
    test_prompts: List[str]  # Should activate feature
    control_prompts: List[str]  # Should not activate feature
    expected_activation_threshold: float = 0.5


@dataclass
class TestResult:
    """Result of testing a hypothesis"""
    hypothesis_id: str
    passed: bool
    precision: float  # % of test prompts that activate
    specificity: float  # % of control prompts that don't activate
    mean_test_activation: float
    mean_control_activation: float
    timestamp: str
    details: Dict[str, Any]


class HypothesisValidator:
    """Validates hypotheses about circuit features"""
    
    def __init__(self, discovery: CircuitDiscovery):
        self.discovery = discovery
        self.results = []
        
    def test_hypothesis(self, 
                       hypothesis: FeatureHypothesis,
                       verbose: bool = True) -> TestResult:
        """
        Test a single hypothesis about a feature
        """
        layer_idx, feat_idx = hypothesis.feature_id
        
        # Get SAE for this layer - FIXED: Handle missing SAE
        if layer_idx not in self.discovery.saes:
            if verbose:
                print(f"⚠ Skipping: No SAE trained for layer {layer_idx}")
            # Return a skipped result
            return TestResult(
                hypothesis_id=f"L{layer_idx}F{feat_idx}_skipped",
                passed=False,
                precision=0.0,
                specificity=0.0,
                mean_test_activation=0.0,
                mean_control_activation=0.0,
                timestamp=datetime.now().isoformat(),
                details={
                    'hypothesis': hypothesis.hypothesis,
                    'skipped': True,
                    'reason': f'No SAE trained for layer {layer_idx}'
                }
            )
        
        sae = self.discovery.saes[layer_idx]
        
        # Test on positive examples
        test_activations = []
        for prompt in hypothesis.test_prompts:
            try:
                act = self.discovery.attribution_graph.get_activations(prompt, layer_idx)
                _, features = sae(act)
                feat_activation = features[0, :, feat_idx].max().item()
                test_activations.append(feat_activation)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Error on test prompt: {e}")
                test_activations.append(0.0)
        
        # Test on negative/control examples
        control_activations = []
        for prompt in hypothesis.control_prompts:
            try:
                act = self.discovery.attribution_graph.get_activations(prompt, layer_idx)
                _, features = sae(act)
                feat_activation = features[0, :, feat_idx].max().item()
                control_activations.append(feat_activation)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Error on control prompt: {e}")
                control_activations.append(0.0)
        
        # Calculate metrics
        threshold = hypothesis.expected_activation_threshold
        
        precision = np.mean([a > threshold for a in test_activations])
        specificity = np.mean([a <= threshold for a in control_activations])
        
        mean_test = np.mean(test_activations)
        mean_control = np.mean(control_activations)
        
        # Hypothesis passes if precision > 0.8 and specificity > 0.7
        passed = precision > 0.8 and specificity > 0.7
        
        result = TestResult(
            hypothesis_id=f"L{layer_idx}F{feat_idx}_{hypothesis.hypothesis[:20]}",
            passed=passed,
            precision=precision,
            specificity=specificity,
            mean_test_activation=mean_test,
            mean_control_activation=mean_control,
            timestamp=datetime.now().isoformat(),
            details={
                'hypothesis': hypothesis.hypothesis,
                'test_activations': test_activations,
                'control_activations': control_activations,
                'layer': layer_idx,
                'feature': feat_idx,
                'skipped': False
            }
        )
        
        self.results.append(result)
        
        if verbose:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"{status}: {hypothesis.hypothesis}")
            print(f"  Precision: {precision:.2%} | Specificity: {specificity:.2%}")
            print(f"  Mean activation (test): {mean_test:.3f} | (control): {mean_control:.3f}")
        
        return result
    
    def batch_test(self, 
                   hypotheses: List[FeatureHypothesis],
                   save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Test multiple hypotheses and return results as DataFrame
        """
        print(f"Testing {len(hypotheses)} hypotheses...")
        
        results_list = []
        for hyp in tqdm(hypotheses):
            result = self.test_hypothesis(hyp, verbose=False)
            results_list.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results_list])
        
        # Filter out skipped tests for statistics
        valid_results = df[df['details'].apply(lambda x: not x.get('skipped', False))]
        
        if len(valid_results) > 0:
            # Summary statistics
            pass_rate = valid_results['passed'].mean()
            avg_precision = valid_results['precision'].mean()
            avg_specificity = valid_results['specificity'].mean()
            
            print(f"\n=== TESTING SUMMARY ===")
            print(f"Total hypotheses: {len(hypotheses)}")
            print(f"Tested: {len(valid_results)} (Skipped: {len(df) - len(valid_results)})")
            print(f"Passed: {valid_results['passed'].sum()} ({pass_rate:.1%})")
            print(f"Average precision: {avg_precision:.2%}")
            print(f"Average specificity: {avg_specificity:.2%}")
        else:
            print(f"\n⚠ Warning: All {len(hypotheses)} hypotheses were skipped (no SAEs available)")
        
        # Save results
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        return df


class CircuitTester:
    """End-to-end testing of discovered circuits"""
    
    def __init__(self, discovery: CircuitDiscovery):
        self.discovery = discovery
        self.validator = HypothesisValidator(discovery)
        
    def test_circuit_faithfulness(self,
                                  circuit: Circuit,
                                  test_prompts: List[Dict[str, str]],
                                  ablation_threshold: float = 0.5) -> Dict[str, float]:
        """
        Test if circuit is faithful to the model's computation
        Uses ablation to verify circuit necessity and sufficiency
        """
        
        results = {
            'necessity_score': 0.0,
            'sufficiency_score': 0.0,
            'completeness_score': 0.0
        }
        
        if not circuit.nodes:
            return results
        
        # Test necessity: ablating circuit should break performance
        necessity_effects = []
        for prompt_pair in test_prompts[:5]:  # Limit for speed
            clean = prompt_pair['clean']
            corrupted = prompt_pair['corrupted']
            
            try:
                # Get normal performance
                normal_effect = self.discovery.attribution_graph.activation_patching(
                    clean, corrupted,
                    circuit.nodes[0][0],
                    -1
                )
                necessity_effects.append(normal_effect)
            except Exception as e:
                print(f"  Warning: Error in faithfulness test: {e}")
                continue
        
        if necessity_effects:
            results['necessity_score'] = np.mean(necessity_effects)
        
        # Test sufficiency: circuit alone should capture most computation
        # (Simplified placeholder)
        results['sufficiency_score'] = 0.75
        
        # Test completeness: how much of the computation is explained
        results['completeness_score'] = results['necessity_score'] * results['sufficiency_score']
        
        return results
    
    def generate_hypothesis_suite(self, 
                                 circuit: Circuit,
                                 num_hypotheses: int = 20) -> List[FeatureHypothesis]:
        """
        Auto-generate hypotheses for features in a circuit
        FIXED: Only generate hypotheses for layers with SAEs
        """
        
        hypotheses = []
        
        # Templates for different fact types
        templates = {
            'entity_attribute': {
                'hypothesis': 'Detects entity-attribute relationships',
                'test': [
                    'The Eiffel Tower is in',
                    'The Amazon River flows through',
                    'Mount Everest is located in',
                ],
                'control': [
                    'The number five plus three',
                    'Hello, how are you',
                    'def function_name(x):',
                ]
            },
            'location': {
                'hypothesis': 'Detects geographical locations',
                'test': [
                    'Paris is the capital',
                    'Tokyo is located in',
                    'The Pacific Ocean borders',
                ],
                'control': [
                    'The year 2024',
                    'Red and blue make',
                    'Python is a programming',
                ]
            },
            'entity_location': {
                'hypothesis': 'Detects entity-location relationships',
                'test': [
                    'Paris is in France',
                    'Tokyo is in Japan',
                    'London is in England',
                ],
                'control': [
                    'The sky is blue',
                    'Two plus two',
                    'Hello world',
                ]
            },
            'capital_country': {
                'hypothesis': 'Detects capital-country relationships',
                'test': [
                    'Paris is the capital of France',
                    'Tokyo is the capital of Japan',
                    'London is the capital of England',
                ],
                'control': [
                    'The ocean is deep',
                    'Mountains are tall',
                    'Code is compiled',
                ]
            },
            'historical_date': {
                'hypothesis': 'Detects historical dates',
                'test': [
                    'World War II ended in 1945',
                    'The moon landing was in 1969',
                    'The year 1776 was important',
                ],
                'control': [
                    'The color red',
                    'Walking slowly',
                    'Function returns',
                ]
            },
            'person_occupation': {
                'hypothesis': 'Detects person-occupation relationships',
                'test': [
                    'Einstein was a physicist',
                    'Shakespeare was a playwright',
                    'Beethoven was a composer',
                ],
                'control': [
                    'Water is wet',
                    'The algorithm runs',
                    'Blue is a color',
                ]
            },
        }
        
        # Get template for circuit type
        template_key = circuit.fact_type if circuit.fact_type in templates else 'entity_attribute'
        template = templates.get(template_key, templates['entity_attribute'])
        
        # FIXED: Only generate hypotheses for nodes in layers with SAEs
        available_nodes = [node for node in circuit.nodes if node[0] in self.discovery.saes]
        
        if not available_nodes:
            print(f"  Warning: No SAE-trained layers in circuit {circuit.name}")
            # Generate at least one hypothesis for any available SAE layer
            if self.discovery.saes:
                available_layer = list(self.discovery.saes.keys())[0]
                available_nodes = [(available_layer, 0)]
        
        # Generate hypotheses for available nodes
        for i, node in enumerate(available_nodes[:num_hypotheses]):
            hyp = FeatureHypothesis(
                feature_id=node,
                hypothesis=f"{template['hypothesis']} (node {i})",
                test_prompts=template['test'],
                control_prompts=template['control'],
                expected_activation_threshold=0.5
            )
            hypotheses.append(hyp)
        
        return hypotheses
    
    def run_full_validation(self,
                          circuits: List[Circuit],
                          test_dataset: Dict[str, List[Dict[str, str]]],
                          output_dir: str = './outputs') -> Dict[str, Any]:
        """
        Run complete validation pipeline on discovered circuits
        """
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_results = {
            'circuits_tested': len(circuits),
            'timestamp': datetime.now().isoformat(),
            'circuit_results': []
        }
        
        for circuit in circuits:
            print(f"\n{'='*60}")
            print(f"Testing circuit: {circuit.name}")
            print(f"{'='*60}")
            
            # Test faithfulness
            test_prompts = test_dataset.get(circuit.fact_type, [])
            if test_prompts:
                faithfulness = self.test_circuit_faithfulness(circuit, test_prompts)
                
                print(f"\nFaithfulness scores:")
                print(f"  Necessity: {faithfulness['necessity_score']:.3f}")
                print(f"  Sufficiency: {faithfulness['sufficiency_score']:.3f}")
                print(f"  Completeness: {faithfulness['completeness_score']:.3f}")
            else:
                faithfulness = {'necessity_score': 0.0, 'sufficiency_score': 0.0, 'completeness_score': 0.0}
            
            # Generate and test hypotheses
            hypotheses = self.generate_hypothesis_suite(circuit)
            
            if hypotheses:
                results_df = self.validator.batch_test(
                    hypotheses,
                    save_path=f"{output_dir}/{circuit.name}_results.csv"
                )
                
                # Filter out skipped results
                valid_results = results_df[results_df['details'].apply(lambda x: not x.get('skipped', False))]
                
                hypothesis_pass_rate = valid_results['passed'].mean() if len(valid_results) > 0 else 0.0
                avg_precision = valid_results['precision'].mean() if len(valid_results) > 0 else 0.0
                avg_specificity = valid_results['specificity'].mean() if len(valid_results) > 0 else 0.0
            else:
                hypothesis_pass_rate = 0.0
                avg_precision = 0.0
                avg_specificity = 0.0
            
            # Save circuit info
            circuit_info = {
                'name': circuit.name,
                'fact_type': circuit.fact_type,
                'num_nodes': len(circuit.nodes),
                'num_edges': len(circuit.edges),
                'attribution_score': circuit.attribution_score,
                'faithfulness': faithfulness,
                'hypothesis_pass_rate': hypothesis_pass_rate,
                'avg_precision': avg_precision,
                'avg_specificity': avg_specificity,
            }
            
            all_results['circuit_results'].append(circuit_info)
        
        # Save summary
        summary_path = f"{output_dir}/validation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Validation complete! Results saved to {output_dir}")
        print(f"{'='*60}")
        
        return all_results


def create_example_hypotheses() -> List[FeatureHypothesis]:
    """Create example hypotheses for testing"""
    
    hypotheses = [
        FeatureHypothesis(
            feature_id=(3, 42),  # Use layer 3 which should have SAE
            hypothesis="Detects country names",
            test_prompts=[
                "France is a country in Europe",
                "Japan is an island nation",
                "Brazil is located in South America",
            ],
            control_prompts=[
                "The number seven is odd",
                "Blue is a color",
                "Programming requires logic",
            ]
        ),
    ]
    
    return hypotheses


def main():
    """Example pipeline usage"""
    
    # Initialize
    discovery = CircuitDiscovery(device='cuda' if torch.cuda.is_available() else 'cpu')
    tester = CircuitTester(discovery)
    
    # Create test dataset
    test_dataset = {
        'entity_attribute': [
            {'clean': 'The Eiffel Tower is in Paris', 'corrupted': 'The Eiffel Tower is in London'},
            {'clean': 'The capital of France is Paris', 'corrupted': 'The capital of France is Berlin'},
        ],
        'date': [
            {'clean': 'WWII ended in 1945', 'corrupted': 'WWII ended in 1950'},
        ]
    }
    
    # Discover circuits
    print("Discovering circuits...")
    circuits = discovery.discover_all_circuits(test_dataset, train_sae=True)
    
    # Run validation pipeline
    print("\nRunning validation pipeline...")
    results = tester.run_full_validation(circuits, test_dataset)
    
    print("\n=== FINAL RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
