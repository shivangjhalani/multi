"""
Reasoning Quality Analysis Tools for Multimodal CoCoNuT

This module implements tools to inspect continuous thought representations,
visualize latent space reasoning progression, and create comparison metrics
between discrete and continuous reasoning approaches.

Key features:
- Continuous thought representation inspection
- Latent space reasoning progression visualization
- Comparison metrics between discrete CoT and continuous CoCoNuT
- Reasoning step analysis and quality assessment
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Some visualization features will be limited.")


@dataclass
class ReasoningStep:
    """Single reasoning step with continuous thought representation"""
    step_index: int
    step_text: Optional[str]  # None for latent steps
    hidden_state: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    is_latent: bool = False
    confidence_score: Optional[float] = None


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a single sample"""
    sample_id: str
    question: str
    image_path: str
    reasoning_steps: List[ReasoningStep]
    final_answer: str
    ground_truth_answer: str
    is_correct: bool
    stage: int  # Training stage (0 = CoT, >0 = CoCoNuT)
    
    def get_latent_steps(self) -> List[ReasoningStep]:
        """Get only the latent reasoning steps"""
        return [step for step in self.reasoning_steps if step.is_latent]
    
    def get_text_steps(self) -> List[ReasoningStep]:
        """Get only the textual reasoning steps"""
        return [step for step in self.reasoning_steps if not step.is_latent]


class ReasoningQualityAnalyzer:
    """
    Analyzer for reasoning quality in multimodal CoCoNuT models.
    
    Provides tools to inspect continuous thought representations,
    visualize reasoning progression, and compare different reasoning modes.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 config,
                 device: Optional[torch.device] = None):
        """
        Initialize reasoning quality analyzer
        
        Args:
            model: Multimodal CoCoNuT model
            tokenizer: Tokenizer with special tokens
            config: Configuration object
            device: Device for computations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device or next(model.parameters()).device
        
        # Special token IDs
        self.latent_id = getattr(tokenizer, 'latent_token_id', None)
        self.start_id = getattr(tokenizer, 'start_latent_id', None)
        self.end_id = getattr(tokenizer, 'end_latent_id', None)
        
        # Analysis settings
        self.hidden_dim = model.config.hidden_size if hasattr(model, 'config') else 4096
        
        # Storage for analysis results
        self.reasoning_traces: List[ReasoningTrace] = []
        self.analysis_cache: Dict[str, Any] = {}
    
    def extract_reasoning_trace(self,
                               pixel_values: torch.Tensor,
                               input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               sample_id: str,
                               question: str,
                               image_path: str,
                               ground_truth_answer: str,
                               stage: int = 0) -> ReasoningTrace:
        """
        Extract complete reasoning trace from model forward pass
        
        Args:
            pixel_values: Image tensor
            input_ids: Input token IDs
            attention_mask: Attention mask
            sample_id: Unique sample identifier
            question: Question text
            image_path: Path to image
            ground_truth_answer: Ground truth answer
            stage: Training stage
            
        Returns:
            Complete reasoning trace with hidden states
        """
        self.model.eval()
        
        with torch.no_grad():
            # Move inputs to device
            pixel_values = pixel_values.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass with output hidden states
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )
            
            # Extract hidden states and attention weights
            hidden_states = outputs.hidden_states[-1]  # Last layer
            attention_weights = outputs.attentions[-1] if outputs.attentions else None
            
            # Generate response to get final answer
            generated_outputs = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            
            # Decode final answer
            generated_text = self.tokenizer.decode(generated_outputs.sequences[0], skip_special_tokens=True)
            final_answer = self._extract_answer_from_text(generated_text)
            
            # Build reasoning steps
            reasoning_steps = []
            input_tokens = input_ids[0].cpu().tolist()
            
            for i, token_id in enumerate(input_tokens):
                # Get hidden state for this position
                hidden_state = hidden_states[0, i, :].cpu()
                
                # Get attention weights if available
                attn_weights = attention_weights[0, :, i, :].cpu() if attention_weights is not None else None
                
                # Determine if this is a latent step
                is_latent = token_id == self.latent_id
                
                # Get step text (None for latent tokens)
                step_text = None if is_latent else self.tokenizer.decode([token_id])
                
                # Calculate confidence score (using attention entropy as proxy)
                confidence_score = None
                if attn_weights is not None:
                    # Calculate attention entropy as confidence measure
                    attn_probs = F.softmax(attn_weights.mean(0), dim=-1)
                    entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8))
                    confidence_score = 1.0 / (1.0 + entropy.item())  # Convert to confidence
                
                reasoning_step = ReasoningStep(
                    step_index=i,
                    step_text=step_text,
                    hidden_state=hidden_state,
                    attention_weights=attn_weights,
                    is_latent=is_latent,
                    confidence_score=confidence_score
                )
                
                reasoning_steps.append(reasoning_step)
            
            # Check if answer is correct
            is_correct = self._normalize_answer(final_answer) == self._normalize_answer(ground_truth_answer)
            
            reasoning_trace = ReasoningTrace(
                sample_id=sample_id,
                question=question,
                image_path=image_path,
                reasoning_steps=reasoning_steps,
                final_answer=final_answer,
                ground_truth_answer=ground_truth_answer,
                is_correct=is_correct,
                stage=stage
            )
            
            return reasoning_trace
    
    def _extract_answer_from_text(self, text: str) -> str:
        """Extract answer from generated text"""
        if "###" in text:
            return text.split("###")[-1].strip()
        return text.strip().split('\n')[-1]
    
    def _normalize_answer(self, answer: str) -> str:
        """Simple answer normalization"""
        return answer.lower().strip().replace(',', '').replace('.', '')
    
    def analyze_continuous_thoughts(self, reasoning_traces: List[ReasoningTrace]) -> Dict[str, Any]:
        """
        Analyze continuous thought representations across reasoning traces
        
        Args:
            reasoning_traces: List of reasoning traces to analyze
            
        Returns:
            Analysis results dictionary
        """
        latent_representations = []
        latent_metadata = []
        
        # Collect all latent representations
        for trace in reasoning_traces:
            latent_steps = trace.get_latent_steps()
            
            for step in latent_steps:
                latent_representations.append(step.hidden_state.numpy())
                latent_metadata.append({
                    'sample_id': trace.sample_id,
                    'step_index': step.step_index,
                    'is_correct': trace.is_correct,
                    'stage': trace.stage,
                    'confidence': step.confidence_score
                })
        
        if not latent_representations:
            return {'error': 'No latent representations found'}
        
        # Convert to numpy array
        latent_matrix = np.array(latent_representations)
        
        # Dimensionality analysis
        pca = PCA()
        pca_result = pca.fit_transform(latent_matrix)
        
        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Clustering analysis
        from sklearn.cluster import KMeans
        n_clusters = min(8, len(latent_representations) // 10)  # Reasonable number of clusters
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(latent_matrix)
        else:
            cluster_labels = np.zeros(len(latent_representations))
        
        # Similarity analysis
        similarity_matrix = cosine_similarity(latent_matrix)
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        
        # Correctness correlation
        correct_mask = np.array([meta['is_correct'] for meta in latent_metadata])
        if np.sum(correct_mask) > 0 and np.sum(~correct_mask) > 0:
            correct_representations = latent_matrix[correct_mask]
            incorrect_representations = latent_matrix[~correct_mask]
            
            # Calculate average representations
            avg_correct = np.mean(correct_representations, axis=0)
            avg_incorrect = np.mean(incorrect_representations, axis=0)
            
            # Calculate separation
            correctness_separation = cosine_similarity([avg_correct], [avg_incorrect])[0, 0]
        else:
            correctness_separation = None
        
        analysis_results = {
            'total_latent_steps': len(latent_representations),
            'dimensionality': {
                'original_dim': latent_matrix.shape[1],
                'effective_dim_95': np.argmax(cumulative_variance >= 0.95) + 1,
                'effective_dim_99': np.argmax(cumulative_variance >= 0.99) + 1,
                'explained_variance_top10': explained_variance[:10].tolist()
            },
            'clustering': {
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'silhouette_score': self._calculate_silhouette_score(latent_matrix, cluster_labels)
            },
            'similarity': {
                'average_cosine_similarity': float(avg_similarity),
                'similarity_std': float(np.std(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
            },
            'correctness_analysis': {
                'correctness_separation': float(correctness_separation) if correctness_separation is not None else None,
                'correct_samples': int(np.sum(correct_mask)),
                'incorrect_samples': int(np.sum(~correct_mask))
            },
            'pca_components': pca_result[:, :10].tolist(),  # First 10 PCA components
            'metadata': latent_metadata
        }
        
        return analysis_results
    
    def _calculate_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering"""
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) > 1:
                return float(silhouette_score(X, labels))
            else:
                return 0.0
        except:
            return 0.0
    
    def compare_reasoning_modes(self,
                               cot_traces: List[ReasoningTrace],
                               coconut_traces: List[ReasoningTrace]) -> Dict[str, Any]:
        """
        Compare reasoning quality between CoT and CoCoNuT modes
        
        Args:
            cot_traces: Reasoning traces from CoT mode (stage 0)
            coconut_traces: Reasoning traces from CoCoNuT mode (stage > 0)
            
        Returns:
            Comparison analysis results
        """
        comparison_results = {
            'cot_analysis': self._analyze_reasoning_mode(cot_traces, 'CoT'),
            'coconut_analysis': self._analyze_reasoning_mode(coconut_traces, 'CoCoNuT'),
            'comparative_metrics': {}
        }
        
        # Calculate comparative metrics
        cot_accuracy = np.mean([trace.is_correct for trace in cot_traces]) if cot_traces else 0.0
        coconut_accuracy = np.mean([trace.is_correct for trace in coconut_traces]) if coconut_traces else 0.0
        
        comparison_results['comparative_metrics'] = {
            'cot_accuracy': float(cot_accuracy),
            'coconut_accuracy': float(coconut_accuracy),
            'accuracy_improvement': float(coconut_accuracy - cot_accuracy),
            'relative_improvement': float((coconut_accuracy - cot_accuracy) / cot_accuracy * 100) if cot_accuracy > 0 else 0.0
        }
        
        # Reasoning efficiency comparison
        if cot_traces and coconut_traces:
            cot_avg_steps = np.mean([len(trace.reasoning_steps) for trace in cot_traces])
            coconut_avg_steps = np.mean([len(trace.reasoning_steps) for trace in coconut_traces])
            
            comparison_results['comparative_metrics'].update({
                'cot_avg_reasoning_steps': float(cot_avg_steps),
                'coconut_avg_reasoning_steps': float(coconut_avg_steps),
                'reasoning_efficiency': float(coconut_avg_steps / cot_avg_steps) if cot_avg_steps > 0 else 1.0
            })
        
        return comparison_results
    
    def _analyze_reasoning_mode(self, traces: List[ReasoningTrace], mode_name: str) -> Dict[str, Any]:
        """Analyze reasoning quality for a specific mode"""
        if not traces:
            return {'error': f'No traces available for {mode_name}'}
        
        # Basic statistics
        accuracy = np.mean([trace.is_correct for trace in traces])
        avg_steps = np.mean([len(trace.reasoning_steps) for trace in traces])
        
        # Confidence analysis
        all_confidences = []
        for trace in traces:
            for step in trace.reasoning_steps:
                if step.confidence_score is not None:
                    all_confidences.append(step.confidence_score)
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        confidence_std = np.std(all_confidences) if all_confidences else 0.0
        
        # Latent step analysis (for CoCoNuT)
        latent_analysis = None
        if any(trace.get_latent_steps() for trace in traces):
            latent_analysis = self.analyze_continuous_thoughts(traces)
        
        return {
            'mode': mode_name,
            'num_traces': len(traces),
            'accuracy': float(accuracy),
            'avg_reasoning_steps': float(avg_steps),
            'confidence_stats': {
                'avg_confidence': float(avg_confidence),
                'confidence_std': float(confidence_std)
            },
            'latent_analysis': latent_analysis
        }
    
    def visualize_reasoning_progression(self,
                                      reasoning_trace: ReasoningTrace,
                                      save_path: Optional[str] = None,
                                      use_plotly: bool = True) -> Optional[str]:
        """
        Visualize reasoning progression for a single trace
        
        Args:
            reasoning_trace: Reasoning trace to visualize
            save_path: Path to save visualization (optional)
            use_plotly: Whether to use Plotly for interactive visualization
            
        Returns:
            Path to saved visualization or None
        """
        if use_plotly and PLOTLY_AVAILABLE:
            return self._visualize_with_plotly(reasoning_trace, save_path)
        else:
            return self._visualize_with_matplotlib(reasoning_trace, save_path)
    
    def _visualize_with_plotly(self, reasoning_trace: ReasoningTrace, save_path: Optional[str]) -> Optional[str]:
        """Create interactive visualization with Plotly"""
        # Extract hidden states
        hidden_states = np.array([step.hidden_state.numpy() for step in reasoning_trace.reasoning_steps])
        
        # Dimensionality reduction
        if hidden_states.shape[0] > 2:
            pca = PCA(n_components=min(3, hidden_states.shape[0]))
            reduced_states = pca.fit_transform(hidden_states)
        else:
            reduced_states = hidden_states[:, :3]
        
        # Create traces for different step types
        latent_indices = [i for i, step in enumerate(reasoning_trace.reasoning_steps) if step.is_latent]
        text_indices = [i for i, step in enumerate(reasoning_trace.reasoning_steps) if not step.is_latent]
        
        fig = go.Figure()
        
        # Add latent steps
        if latent_indices:
            latent_states = reduced_states[latent_indices]
            fig.add_trace(go.Scatter3d(
                x=latent_states[:, 0],
                y=latent_states[:, 1],
                z=latent_states[:, 2] if latent_states.shape[1] > 2 else np.zeros(len(latent_states)),
                mode='markers+lines',
                marker=dict(size=8, color='red', symbol='circle'),
                name='Latent Steps',
                text=[f'Latent Step {i}' for i in latent_indices],
                hovertemplate='%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
            ))
        
        # Add text steps
        if text_indices:
            text_states = reduced_states[text_indices]
            step_texts = [reasoning_trace.reasoning_steps[i].step_text[:50] + '...' 
                         if len(reasoning_trace.reasoning_steps[i].step_text or '') > 50 
                         else reasoning_trace.reasoning_steps[i].step_text or ''
                         for i in text_indices]
            
            fig.add_trace(go.Scatter3d(
                x=text_states[:, 0],
                y=text_states[:, 1],
                z=text_states[:, 2] if text_states.shape[1] > 2 else np.zeros(len(text_states)),
                mode='markers+lines',
                marker=dict(size=6, color='blue', symbol='diamond'),
                name='Text Steps',
                text=step_texts,
                hovertemplate='%{text}<br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Reasoning Progression - {reasoning_trace.sample_id}',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            width=800,
            height=600
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            return save_path
        
        return None
    
    def _visualize_with_matplotlib(self, reasoning_trace: ReasoningTrace, save_path: Optional[str]) -> Optional[str]:
        """Create static visualization with Matplotlib"""
        # Extract hidden states
        hidden_states = np.array([step.hidden_state.numpy() for step in reasoning_trace.reasoning_steps])
        
        # Dimensionality reduction to 2D
        if hidden_states.shape[0] > 2:
            pca = PCA(n_components=2)
            reduced_states = pca.fit_transform(hidden_states)
        else:
            reduced_states = hidden_states[:, :2]
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot reasoning progression
        latent_mask = np.array([step.is_latent for step in reasoning_trace.reasoning_steps])
        
        # Plot latent steps
        if np.any(latent_mask):
            latent_states = reduced_states[latent_mask]
            ax.scatter(latent_states[:, 0], latent_states[:, 1], 
                      c='red', s=100, alpha=0.7, label='Latent Steps', marker='o')
        
        # Plot text steps
        if np.any(~latent_mask):
            text_states = reduced_states[~latent_mask]
            ax.scatter(text_states[:, 0], text_states[:, 1], 
                      c='blue', s=60, alpha=0.7, label='Text Steps', marker='s')
        
        # Connect steps with lines
        ax.plot(reduced_states[:, 0], reduced_states[:, 1], 'k--', alpha=0.3, linewidth=1)
        
        # Add step numbers
        for i, (x, y) in enumerate(reduced_states):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Reasoning Progression - {reasoning_trace.sample_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        return None
    
    def generate_reasoning_report(self,
                                 reasoning_traces: List[ReasoningTrace],
                                 output_dir: str) -> str:
        """
        Generate comprehensive reasoning quality report
        
        Args:
            reasoning_traces: List of reasoning traces to analyze
            output_dir: Directory to save report and visualizations
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Separate traces by stage
        cot_traces = [trace for trace in reasoning_traces if trace.stage == 0]
        coconut_traces = [trace for trace in reasoning_traces if trace.stage > 0]
        
        # Perform analyses
        continuous_analysis = self.analyze_continuous_thoughts(reasoning_traces)
        comparison_analysis = self.compare_reasoning_modes(cot_traces, coconut_traces)
        
        # Generate report
        report = {
            'summary': {
                'total_traces': len(reasoning_traces),
                'cot_traces': len(cot_traces),
                'coconut_traces': len(coconut_traces),
                'overall_accuracy': np.mean([trace.is_correct for trace in reasoning_traces])
            },
            'continuous_thought_analysis': continuous_analysis,
            'reasoning_mode_comparison': comparison_analysis,
            'sample_visualizations': []
        }
        
        # Create visualizations for sample traces
        sample_traces = reasoning_traces[:5]  # First 5 traces
        for i, trace in enumerate(sample_traces):
            viz_path = output_path / f'reasoning_trace_{i}.html'
            saved_path = self.visualize_reasoning_progression(trace, str(viz_path))
            if saved_path:
                report['sample_visualizations'].append(saved_path)
        
        # Save report
        report_path = output_path / 'reasoning_quality_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(report_path)