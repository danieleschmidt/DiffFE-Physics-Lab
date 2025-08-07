#!/usr/bin/env python3
"""Basic sentiment analysis example using physics-informed methods.

This example demonstrates how to use the DiffFE-Physics-Lab framework
for sentiment analysis by treating sentiment as a physical field that
evolves according to diffusion-reaction dynamics in semantic space.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import time

# Import our sentiment analysis components
from src.services.sentiment_analyzer import SentimentAnalyzer, SentimentAnalysisResult
from src.models.sentiment_problem import TextEmbeddingSentimentProblem
from src.operators.sentiment import CompositeSentimentOperator


def basic_sentiment_analysis_demo():
    """Demonstrate basic sentiment analysis capabilities."""
    print("🎭 DiffFE-Physics-Lab: Physics-Informed Sentiment Analysis Demo")
    print("=" * 60)
    
    # Sample texts with various sentiments
    sample_texts = [
        "I absolutely love this amazing product! It's fantastic!",
        "This is the worst experience I've ever had. Terrible!",
        "The weather is okay today, nothing special.",
        "I'm feeling quite happy about the recent developments.",
        "This disappointing result really upset me.",
        "The documentation could be better, but it's functional.",
        "Outstanding performance! Exceeded all my expectations!",
        "I hate waiting in long lines at the store.",
        "The movie was neither good nor bad, just average.",
        "Brilliant innovation! This will change everything!"
    ]
    
    print(f"📝 Analyzing {len(sample_texts)} sample texts...\n")
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer(
        embedding_method='tfidf',
        embedding_dim=200,
        backend='jax',
        performance_monitoring=True
    )
    
    # Analyze sentiments with detailed diagnostics
    print("🔬 Running physics-informed sentiment analysis...")
    start_time = time.time()
    
    result: SentimentAnalysisResult = analyzer.analyze(
        sample_texts,
        return_diagnostics=True,
        auto_tune_params=True
    )
    
    analysis_time = time.time() - start_time
    print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
    print(f"⚡ Processing speed: {result.tokens_per_second:.1f} tokens/second")
    print(f"💾 Memory usage: {result.memory_usage_mb:.1f} MB\n")
    
    # Display results
    print("📊 SENTIMENT ANALYSIS RESULTS")
    print("-" * 50)
    
    for i, (text, sentiment, confidence) in enumerate(zip(
        sample_texts, result.sentiments, result.confidence_scores
    )):
        # Categorize sentiment
        if sentiment > 0.3:
            category = "😊 Positive"
            color = "green"
        elif sentiment < -0.3:
            category = "😞 Negative" 
            color = "red"
        else:
            category = "😐 Neutral"
            color = "yellow"
            
        print(f"{i+1:2d}. {category} | Score: {sentiment:+.3f} | Confidence: {confidence:.3f}")
        print(f"    Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        print()
    
    # Physics analysis
    print("🔬 PHYSICS ANALYSIS")
    print("-" * 30)
    print(f"Physics Parameters:")
    print(f"  Temperature: {result.physics_parameters['temperature']:.3f}")
    print(f"  Reaction Strength: {result.physics_parameters['reaction_strength']:.3f}")
    print(f"  Integration Steps: {result.physics_parameters['num_steps']}")
    print(f"  Time Step: {result.physics_parameters['dt']:.4f}")
    print()
    
    print(f"Convergence Analysis:")
    print(f"  Converged: {result.convergence_info['converged']}")
    print(f"  Final Energy: {result.convergence_info['final_energy']:.6f}")
    print(f"  Mean Sentiment: {result.convergence_info['mean_sentiment']:.3f}")
    print(f"  Sentiment Variance: {result.convergence_info['sentiment_variance']:.3f}")
    print(f"  Extreme Sentiments: {result.convergence_info['extreme_sentiments']}")
    
    # Visualize energy evolution
    if result.energy_evolution:
        plt.figure(figsize=(10, 6))
        
        # Energy evolution plot
        plt.subplot(2, 1, 1)
        plt.plot(result.energy_evolution, 'b-o', markersize=4)
        plt.title('Physics Energy Evolution During Sentiment Analysis')
        plt.xlabel('Integration Step (×10)')
        plt.ylabel('System Energy')
        plt.grid(True, alpha=0.3)
        
        # Sentiment distribution
        plt.subplot(2, 1, 2)
        plt.hist(result.sentiments, bins=15, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/sentiment_analysis_physics.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 Physics visualization saved to /tmp/sentiment_analysis_physics.png")
    
    return result


def compare_embedding_methods():
    """Compare different text embedding methods for sentiment analysis."""
    print("\n🔍 EMBEDDING METHOD COMPARISON")
    print("=" * 40)
    
    test_texts = [
        "This product is absolutely amazing and wonderful!",
        "Terrible experience, completely disappointed.",
        "Average quality, nothing special about it.",
        "Love the innovative features and great design!"
    ]
    
    embedding_methods = ['tfidf', 'word2vec']
    results = {}
    
    for method in embedding_methods:
        print(f"\n🔬 Testing {method.upper()} embeddings...")
        
        analyzer = SentimentAnalyzer(
            embedding_method=method,
            embedding_dim=150,
            performance_monitoring=False
        )
        
        start_time = time.time()
        sentiments = analyzer.analyze(test_texts)
        processing_time = time.time() - start_time
        
        results[method] = {
            'sentiments': sentiments,
            'processing_time': processing_time
        }
        
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Sentiments: {sentiments}")
    
    # Compare results
    print("\n📊 COMPARISON SUMMARY")
    print("-" * 30)
    
    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: \"{text[:40]}...\"")
        for method in embedding_methods:
            sentiment = results[method]['sentiments'][i]
            print(f"  {method:>8s}: {sentiment:+.3f}")
    
    print("\nProcessing Speed Comparison:")
    for method in embedding_methods:
        time_taken = results[method]['processing_time']
        print(f"  {method:>8s}: {time_taken:.3f} seconds")


def physics_parameter_exploration():
    """Explore how different physics parameters affect sentiment analysis."""
    print("\n⚛️ PHYSICS PARAMETER EXPLORATION")
    print("=" * 40)
    
    test_text = ["This is a moderately positive statement with mixed feelings."]
    
    # Test different temperature values
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print("🌡️ Temperature Parameter Effects:")
    print("(Lower temperature = less diffusion, more localized sentiment)")
    
    for temp in temperatures:
        analyzer = SentimentAnalyzer(performance_monitoring=False)
        
        sentiments = analyzer.analyze(
            test_text,
            physics_params={'temperature': temp, 'reaction_strength': 0.5},
            auto_tune_params=False
        )
        
        print(f"  Temperature {temp:4.1f}: Sentiment = {sentiments[0]:+.4f}")
    
    # Test different reaction strengths
    print("\n⚡ Reaction Strength Parameter Effects:")
    print("(Higher strength = stronger sentiment polarization)")
    
    reaction_strengths = [0.1, 0.3, 0.5, 0.8, 1.2]
    
    for strength in reaction_strengths:
        analyzer = SentimentAnalyzer(performance_monitoring=False)
        
        sentiments = analyzer.analyze(
            test_text,
            physics_params={'temperature': 1.0, 'reaction_strength': strength},
            auto_tune_params=False
        )
        
        print(f"  Strength {strength:4.1f}: Sentiment = {sentiments[0]:+.4f}")


def batch_processing_demo():
    """Demonstrate batch processing capabilities."""
    print("\n📦 BATCH PROCESSING DEMO")
    print("=" * 30)
    
    # Create multiple batches of texts
    batch1 = [
        "Excellent service and great quality!",
        "Poor performance, very disappointed.",
        "Average experience, nothing remarkable."
    ]
    
    batch2 = [
        "Outstanding results exceeded expectations!",
        "Horrible experience, waste of time.",
        "Decent product with some minor issues."
    ]
    
    batch3 = [
        "Brilliant innovation, highly recommended!",
        "Complete failure, totally unsatisfied.",
        "Okay results, meets basic requirements."
    ]
    
    batches = [batch1, batch2, batch3]
    
    analyzer = SentimentAnalyzer()
    
    # Process batches with progress tracking
    def progress_callback(current, total):
        print(f"  Progress: {current}/{total} batches completed")
    
    print("📊 Processing 3 batches of texts...")
    start_time = time.time()
    
    batch_results = analyzer.analyze_batch(
        batches,
        progress_callback=progress_callback
    )
    
    total_time = time.time() - start_time
    
    print(f"✅ Batch processing completed in {total_time:.3f} seconds")
    
    # Display batch results
    for i, (batch, sentiments) in enumerate(zip(batches, batch_results)):
        print(f"\nBatch {i+1} Results:")
        for j, (text, sentiment) in enumerate(zip(batch, sentiments)):
            category = "Positive" if sentiment > 0.2 else "Negative" if sentiment < -0.2 else "Neutral"
            print(f"  {j+1}. {category:8s} ({sentiment:+.3f}): {text[:40]}...")


def main():
    """Run all demonstration examples."""
    try:
        # Basic demo
        result = basic_sentiment_analysis_demo()
        
        # Save detailed results
        result.save('/tmp/sentiment_analysis_results.json')
        print(f"\n💾 Detailed results saved to /tmp/sentiment_analysis_results.json")
        
        # Additional demos
        compare_embedding_methods()
        physics_parameter_exploration()
        batch_processing_demo()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\n🚀 Next Steps:")
        print("  1. Try your own texts with the SentimentAnalyzer")
        print("  2. Experiment with different physics parameters")
        print("  3. Train on labeled data using train_on_labeled_data()")
        print("  4. Explore sentiment explanations with get_sentiment_explanations()")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This may be due to missing dependencies or system configuration.")
        print("Please ensure all requirements are installed correctly.")


if __name__ == "__main__":
    main()