"""
Customer Service Tweet Intent Classification - EDA and Clustering
Converted from explore_data.ipynb notebook
Performs clustering-based intent discovery and comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import psutil
import time

# Configure pandas
pd.set_option('display.expand_frame_repr', False)

class CustomerServiceEDA:
    """Main class for customer service tweet EDA and clustering"""

    def __init__(self, data_path='raw/twcs.csv', output_dir='./reports'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize empty attributes
        self.df : pd.DataFrame = None 
        self.df_inbound : pd.DataFrame = None
        self.tfidf_matrix = None
        self.tfidf: TfidfVectorizer = None
        self.optimal_k: int = None
        self.final_kmeans: KMeans = None

    @staticmethod
    def monitor_memory():
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.2f} MB")
        return memory_mb

    @staticmethod
    def time_operation(func, *args, **kwargs):
        """Time how long an operation takes"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Operation took {end_time - start_time:.2f} seconds")
        return result
    

    @staticmethod
    def clean_text(text):
        """Minimal text cleaning to preserve discriminative features"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Only remove the most obvious noise
        text = re.sub(r'http\S+|www\S+|bit\.ly\S+', '', text)  # URLs
        text = re.sub(r'@\w+', '', text)  # @mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content
        text = re.sub(r'\brt\b', '', text)  # Remove RT
        
        # Fix only the most common contractions
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"n't", " not", text)
        
        # Keep numbers but normalize them slightly
        text = re.sub(r'\b\d{4,}\b', 'LONGNUM', text)  # Only very long numbers
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Keep most punctuation, just normalize spaces
        text = re.sub(r'[^\w\s!?.,]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Only remove if too short
        if len(text.split()) < 2:
            return ""
        
        return text
    
    def load_and_filter_data(self):
        """Load data and filter for customer tweets"""
        print("=== DATA LOADING AND INITIAL EXPLORATION ===")

        # Load dataset
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

        # Filter for customer tweets only
        self.df_inbound = self.df[self.df['inbound']].copy()

        print(f"Total tweets: {len(self.df)}")
        print(f"Customer tweets (inbound=True): {len(self.df_inbound)}")
        print(f"Support responses (inbound=False): {len(self.df[~self.df['inbound']])}")
        
        return self.df_inbound
    
    def preprocess_text(self):
        """Clean and preprocess tweet text"""
        print("\n=== TEXT PREPROCESSING ===")
        
        # Apply cleaning
        self.df_inbound.loc[:, 'text'] = self.df_inbound['text'].astype(str).apply(self.clean_text)
        
        print(f"Tweets before removing empty ones: {len(self.df_inbound)}")
        
        # Remove empty tweets
        self.df_inbound = self.df_inbound[self.df_inbound['text'].str.len() > 5].copy()
        print(f"Tweets after removing empty ones: {len(self.df_inbound)}")
        
        return self.df_inbound


    def create_tfidf_features(self):
        """Conservative TF-IDF to preserve meaningful features"""
        print("\n=== CONSERVATIVE TF-IDF VECTORIZATION ===")
        self.monitor_memory()
        
        # Use built-in English stop words only - don't add custom ones
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        
        # Very conservative parameters
        dataset_size = len(self.df_inbound)
        
        self.tfidf = TfidfVectorizer(
            max_features=15000,  # More features
            ngram_range=(1, 2),  # Just unigrams and bigrams
            min_df=50,  # Less aggressive - keep more terms
            max_df=0.8,  # Less aggressive - keep more terms
            stop_words='english',  # Use built-in only
            lowercase=True,
            dtype=np.float32,
            token_pattern=r'\b\w{2,}\b'
        )
        
        print("Fitting TF-IDF vectorizer...")
        self.tfidf_matrix = self.tfidf.fit_transform(self.df_inbound['text'])
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        feature_names = self.tfidf.get_feature_names_out()
        print(f"Sample features: {feature_names[:50]}")
        
        return self.tfidf_matrix
    

    def find_optimal_clusters(self):
        """Find optimal number of clusters using multiple criteria"""
        print("\n=== CLUSTERING ANALYSIS ===")

        dataset_size = len(self.df_inbound)
        print(f"Dataset size: {dataset_size:,}")

         # DEBUG: Check data quality before clustering
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"TF-IDF matrix density: {self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]):.4f}")
        print(f"Number of non-zero elements: {self.tfidf_matrix.nnz}")
        
        # Check if we have enough diverse data
        if self.tfidf_matrix.shape[1] < 50:
            print("⚠️ WARNING: Very few features extracted. Check text preprocessing.")
            print("Sample processed texts:")
            for i, text in enumerate(self.df_inbound['text'].head(5)):
                print(f"  {i+1}. '{text}'")
        
        # Test with a small K first to debug
        print("\n=== DEBUGGING CLUSTERING ===")
        test_kmeans = KMeans(n_clusters=2, random_state=42, n_init=1)
        test_labels = test_kmeans.fit_predict(self.tfidf_matrix)
        unique_labels = len(np.unique(test_labels))
        print(f"Test clustering with K=2 produced {unique_labels} unique labels")
        print(f"Label distribution: {np.bincount(test_labels)}")

        if unique_labels == 1:
            print("❌ CRITICAL ERROR: All data points assigned to same cluster!")
            print("This indicates an issue with data preprocessing or TF-IDF parameters.")
            
            # Diagnostic information
            print("\n=== DIAGNOSTIC INFORMATION ===")
            print("Unique texts sample:")
            unique_texts = self.df_inbound['text'].nunique()
            print(f"Number of unique texts: {unique_texts} out of {len(self.df_inbound)}")
            
            if unique_texts < 10:
                print("⚠️ Very few unique texts detected!")
                print("Sample texts after preprocessing:")
                for text in self.df_inbound['text'].unique()[:10]:
                    print(f"  '{text}'")
            
            # Check TF-IDF matrix properties
            print("\nTF-IDF matrix statistics:")
            print(f"  Min value: {self.tfidf_matrix.min()}")
            print(f"  Max value: {self.tfidf_matrix.max()}")
            print(f"  Mean value: {self.tfidf_matrix.mean():.6f}")
            
            raise ValueError("Clustering failed: All data points identical. Check data preprocessing.")
        
        # Configure based on dataset size
        if dataset_size > 1000000:
            use_minibatch = True
            batch_size = min(10000, dataset_size // 100)
            K_range = range(2, 16)
            silhouette_sample = 5000
            print(f"LARGE DATASET MODE: batch_size={batch_size}")
        elif dataset_size > 100000:
            use_minibatch = True
            batch_size = 5000
            K_range = range(2, 21)
            silhouette_sample = 8000
            print(f"MEDIUM DATASET MODE: batch_size={batch_size}")
        else:
            use_minibatch = False
            K_range = range(2, 25)
            silhouette_sample = min(10000, dataset_size)
            print("SMALL DATASET MODE: Using regular KMeans")
        
        # Test different K values
        inertias = []
        silhouette_scores = []
        
        print(f"\nTesting {len(K_range)} different cluster counts...")
        for i, k in enumerate(K_range):
            print(f"Progress: {i+1}/{len(K_range)} - Testing K={k}...")
            
            # Choose clustering algorithm
            if use_minibatch:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=42,
                    batch_size=batch_size,
                    n_init=5,
                    max_iter=100,
                    init_size=min(3*batch_size, dataset_size),
                    reassignment_ratio=0.01
                )
            else:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=5,
                    max_iter=100,
                    algorithm='lloyd'
                )
            
            # Fit clustering model
            labels = self.time_operation(kmeans.fit_predict, self.tfidf_matrix)
            inertias.append(kmeans.inertia_)
            
            # Calculate silhouette score
            try:
                sample_indices = np.random.choice(dataset_size, silhouette_sample, replace=False)
                sampled_labels = labels[sample_indices]
                
                # Check if sample has enough diversity
                unique_labels_in_sample = len(np.unique(sampled_labels))
                if unique_labels_in_sample < 2:
                    print(f"  ⚠️ K={k}: Sample has only {unique_labels_in_sample} unique cluster(s)")
                    # Try a larger sample
                    larger_sample_size = min(silhouette_sample * 2, dataset_size // 2)
                    if larger_sample_size > silhouette_sample:
                        sample_indices = np.random.choice(dataset_size, larger_sample_size, replace=False)
                        sampled_labels = labels[sample_indices]
                        unique_labels_in_sample = len(np.unique(sampled_labels))
                    
                    if unique_labels_in_sample < 2:
                        print(f"  ⚠️ K={k}: Even larger sample has only {unique_labels_in_sample} cluster(s), setting silhouette=0")
                        silh_score = 0.0
                    else:
                        silh_score = silhouette_score(
                            self.tfidf_matrix[sample_indices], 
                            sampled_labels
                        )
                        print(f"  ✅ K={k}: Larger sample worked, {unique_labels_in_sample} clusters in sample")
                else:
                    silh_score = silhouette_score(
                        self.tfidf_matrix[sample_indices], 
                        sampled_labels
                    )
                
            except Exception as e:
                print(f"  ❌ K={k}: Silhouette calculation failed: {str(e)}")
                silh_score = 0.0

            silhouette_scores.append(silh_score)
            print(f"  K={k}: Inertia={kmeans.inertia_:.1f}, Silhouette={silh_score:.3f}")
        
        # Find optimal K using comprehensive scoring
        self.optimal_k = self._calculate_optimal_k(K_range, inertias, silhouette_scores)
        
        # Create visualization
        self._plot_clustering_results(K_range, inertias, silhouette_scores)
        
        return self.optimal_k


    def _calculate_optimal_k(self, K_range, inertias, silhouette_scores):
        """Calculate optimal K using improved data-driven methods"""
        print("\n=== IMPROVED K SELECTION ===")
        
        # Method 1: Elbow Detection using Rate of Change
        elbow_scores = self._detect_elbow_point(K_range, inertias)
        
        # Method 2: Silhouette Analysis
        silhouette_optimal = self._find_silhouette_optimal(K_range, silhouette_scores)
        
        # Method 3: Gap Statistic (simplified)
        gap_optimal = self._estimate_gap_statistic(K_range, inertias)
        
        # Display all methods
        print(f"📊 Elbow Method suggests: K={elbow_scores['optimal_k']} (score: {elbow_scores['score']:.3f})")
        print(f"📊 Silhouette Method suggests: K={silhouette_optimal} (score: {max(silhouette_scores):.3f})")
        print(f"📊 Gap Statistic suggests: K={gap_optimal}")
        
        # Create consensus with data-driven weights
        candidates = {}
        for k in K_range:
            k_idx = list(K_range).index(k)
            
            # Elbow score (higher is better)
            elbow_score = elbow_scores['scores'].get(k, 0)
            
            # Silhouette score (higher is better)
            silh_score = silhouette_scores[k_idx] if k_idx < len(silhouette_scores) else 0
            
            # Stability score (avoid too many small clusters)
            min_cluster_size = len(self.df_inbound) / k
            stability_score = min(1.0, min_cluster_size / 1000)  # Prefer clusters with 1000+ samples
            
            # Business practicality (more flexible than before)
            if 4 <= k <= 20:  # Reasonable range for customer service
                business_score = 1.0
            elif 3 <= k <= 25:
                business_score = 0.8
            else:
                business_score = 0.3
            
            # Weighted combination (data-driven weights)
            final_score = (
                0.4 * elbow_score +        # Elbow is most important
                0.3 * silh_score +         # Silhouette is second
                0.2 * stability_score +    # Avoid tiny clusters
                0.1 * business_score       # Minimal business bias
            )
            
            candidates[k] = {
                'final_score': final_score,
                'elbow': elbow_score,
                'silhouette': silh_score,
                'stability': stability_score,
                'business': business_score
            }
        
        # Find best K
        best_k = max(candidates.keys(), key=lambda k: candidates[k]['final_score'])
        
        # Display detailed results
        print("\nTop 5 Candidates:")
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]['final_score'], reverse=True)
        for k, scores in sorted_candidates[:5]:
            print(f"   K={k}: Final={scores['final_score']:.3f} "
                f"(Elbow={scores['elbow']:.3f}, Silh={scores['silhouette']:.3f}, "
                f"Stab={scores['stability']:.3f}, Bus={scores['business']:.3f})")
        
        print(f"\n🎯 RECOMMENDED K: {best_k}")
        return best_k

    def _detect_elbow_point(self, K_range, inertias):
        """Detect elbow point using improved curvature analysis"""
        if len(inertias) < 3:
            return {'optimal_k': K_range[0], 'score': 0, 'scores': {}}
        
        K_range = list(K_range)  # Ensure it's a list for indexing
        
        # Method 1: Using the "knee" detection algorithm
        # Normalize the data to 0-1 range
        k_values = np.array(K_range)
        inertia_values = np.array(inertias)
        
        # Normalize both axes to [0,1]
        k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min())
        inertia_norm = (inertia_values - inertia_values.min()) / (inertia_values.max() - inertia_values.min())
        
        # Calculate distances from each point to the line connecting first and last points
        # This finds the point farthest from the straight line (the "elbow")
        elbow_scores = {}
        
        # Line from first to last point
        x1, y1 = k_norm[0], inertia_norm[0]
        x2, y2 = k_norm[-1], inertia_norm[-1]
        
        for i in range(len(k_values)):
            x0, y0 = k_norm[i], inertia_norm[i]
            
            # Distance from point to line formula
            distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
            elbow_scores[k_values[i]] = distance
        
        # Method 2: Alternative - using rate of change in derivatives
        if len(inertias) >= 4:  # Need at least 4 points for second derivative
            first_diff = np.diff(inertias)
            second_diff = np.diff(first_diff)
            
            # Find where the rate of improvement slows down most
            rate_change_scores = {}
            for i in range(len(second_diff)):
                k = K_range[i + 2]  # +2 because second_diff starts from K_range[2]
                # Larger positive second derivative = bigger slowdown in improvement
                rate_change_scores[k] = second_diff[i] if second_diff[i] > 0 else 0
            
            # Combine both methods
            combined_scores = {}
            for k in K_range:
                distance_score = elbow_scores.get(k, 0)
                rate_score = rate_change_scores.get(k, 0)
                
                # Normalize rate score
                max_rate = max(rate_change_scores.values()) if rate_change_scores else 1
                rate_score_norm = rate_score / max_rate if max_rate > 0 else 0
                
                # Combine scores (weight distance method more heavily)
                combined_scores[k] = 0.7 * distance_score + 0.3 * rate_score_norm
            
            elbow_scores = combined_scores
        
        # Normalize final scores
        if elbow_scores:
            max_score = max(elbow_scores.values())
            if max_score > 0:
                elbow_scores = {k: score/max_score for k, score in elbow_scores.items()}
            
            optimal_k = max(elbow_scores.keys(), key=lambda k: elbow_scores[k])
            
            # Debug output
            print(f"🔍 Elbow Detection Results:")
            sorted_scores = sorted(elbow_scores.items(), key=lambda x: x[1], reverse=True)
            for k, score in sorted_scores[:5]:
                print(f"   K={k}: Elbow score={score:.3f}")
            
            return {
                'optimal_k': optimal_k, 
                'score': elbow_scores[optimal_k],
                'scores': elbow_scores
            }
        
        return {'optimal_k': K_range[0], 'score': 0, 'scores': {}}

    def _find_silhouette_optimal(self, K_range, silhouette_scores):
        """Find K with highest silhouette score"""
        if not silhouette_scores:
            return K_range[0]
        
        max_idx = np.argmax(silhouette_scores)
        return K_range[max_idx]

    def _estimate_gap_statistic(self, K_range, inertias):
        """Simplified gap statistic estimation"""
        if len(inertias) < 2:
            return K_range[0]
        
        # Look for largest drop in inertia
        drops = []
        for i in range(1, len(inertias)):
            drop = inertias[i-1] - inertias[i]
            drops.append(drop)
        
        if drops:
            max_drop_idx = np.argmax(drops)
            return K_range[max_drop_idx + 1]  # +1 because drops is shifted
        
        return K_range[0]
    

    def _plot_clustering_results(self, K_range, inertias, silhouette_scores):
        """Create and save clustering analysis plots"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        
        plt.subplot(1, 2, 2)
        plt.plot(K_range, silhouette_scores, 'ro-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
        #plt.show()
        
        print(f"Clustering plots saved to: {self.output_dir / 'clustering_analysis.png'}")
    

    def apply_final_clustering(self):
        """Apply final clustering with optimal K"""
        print(f"\n=== FINAL CLUSTERING WITH K={self.optimal_k} ===")
        
        # Create final model
        self.final_kmeans = KMeans(n_clusters=self.optimal_k, random_state=42)
        cluster_labels = self.final_kmeans.fit_predict(self.tfidf_matrix)
        
        # Add to dataframe
        self.df_inbound.loc[:, 'cluster'] = cluster_labels
        
        print("Cluster sizes:")
        print(self.df_inbound['cluster'].value_counts().sort_index())
        
        return cluster_labels
    
    def analyze_clusters(self):
        """Analyze cluster characteristics and assign intent labels"""
        print("\n=== CLUSTER ANALYSIS ===")
        
        feature_names = self.tfidf.get_feature_names_out()
        
        # Analyze each cluster
        for cluster_id in range(self.optimal_k):
            cluster_tweets = self.df_inbound[self.df_inbound['cluster'] == cluster_id]
            print(f"\nCLUSTER {cluster_id} ({len(cluster_tweets)} tweets):")
            
            # Show sample tweets
            for i, tweet in enumerate(cluster_tweets['text'].head(3)):
                print(f"  {i+1}. {tweet}")
            
            # Show top terms
            if len(cluster_tweets) > 0:
                cluster_center = self.final_kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-5:][::-1]
                top_features = [feature_names[i] for i in top_indices]
                print(f"  Top words: {', '.join(top_features)}")
            print("-" * 50)
    
    def assign_intent_labels(self):
        """Assign business intent labels based on actual cluster analysis"""
        print("\n=== DATA-DRIVEN INTENT LABELING ===")
        
        # Step 1: Analyze what each cluster actually contains
        feature_names = self.tfidf.get_feature_names_out()
        cluster_profiles = {}
        
        for cluster_id in range(self.optimal_k):
            cluster_tweets = self.df_inbound[self.df_inbound['cluster'] == cluster_id]
            cluster_center = self.final_kmeans.cluster_centers_[cluster_id]
            
            # Get top terms that define this cluster
            top_indices = cluster_center.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Get sample tweets
            sample_tweets = cluster_tweets['text'].head(5).tolist()
            
            cluster_profiles[cluster_id] = {
                'size': len(cluster_tweets),
                'percentage': len(cluster_tweets) / len(self.df_inbound) * 100,
                'top_terms': top_terms,
                'sample_tweets': sample_tweets
            }
            
            print(f"\n📊 CLUSTER {cluster_id} Analysis:")
            print(f"   Size: {len(cluster_tweets)} tweets ({cluster_profiles[cluster_id]['percentage']:.1f}%)")
            print(f"   Key terms: {', '.join(top_terms[:5])}")
            print("   Sample tweets:")
            for i, tweet in enumerate(sample_tweets[:3], 1):
                print(f"     {i}. {tweet}")
        
        # Step 2: Smart intent assignment based on analysis
        intent_mapping = self._create_smart_intent_mapping(cluster_profiles)
        
        # Step 3: Apply mapping and validate balance
        self.df_inbound.loc[:, 'intent'] = self.df_inbound['cluster'].map(intent_mapping)
        
        # Step 4: Check for balance issues
        intent_dist = self.df_inbound['intent'].value_counts(normalize=True)
        max_intent_ratio = intent_dist.max()
        
        print("\nIntent Distribution:")
        for intent, ratio in intent_dist.items():
            print(f"   {intent}: {ratio:.1%}")
        
        # Warn about imbalance
        if max_intent_ratio > 0.4:
            print(f"⚠️  WARNING: '{intent_dist.idxmax()}' dominates with {max_intent_ratio:.1%}")
            print("   Consider splitting this into multiple intents or rebalancing")
        
        return intent_mapping

    def _create_smart_intent_mapping(self, cluster_profiles):
        """Create intent mapping with better balance and specificity"""
        # intent_mapping = {}
        
        # IMPROVED: More specific keywords with weighted scoring
        intent_keywords = {
            'Account_Issues': {
                'primary': ['account', 'login', 'password', 'sign', 'access', 'locked', 'verify', 'authentication', 'username'],
                'secondary': ['profile', 'settings', 'security', 'verification']
            },
            'Billing_Problems': {
                'primary': ['charge', 'bill', 'payment', 'refund', 'money', 'cost', 'price', 'subscription', 'invoice'],
                'secondary': ['credit', 'debit', 'bank', 'card', 'transaction', 'fee']
            },
            'Technical_Issues': {
                'primary': ['crash', 'error', 'bug', 'broken', 'freeze', 'loading', 'glitch'],
                'secondary': ['app', 'website', 'system', 'software', 'update']
            },
            'Service_Issues': {
                'primary': ['outage', 'down', 'offline', 'connection', 'network', 'server'],
                'secondary': ['service', 'availability', 'maintenance']
            },
            'Product_Questions': {
                'primary': ['how', 'what', 'when', 'where', 'feature', 'function', 'tutorial'],
                'secondary': ['guide', 'instruction', 'manual']
            },
            'Delivery_Issues': {
                'primary': ['delivery', 'shipping', 'order', 'package', 'tracking', 'shipment'],
                'secondary': ['received', 'sent', 'mail', 'post']
            },
            'Complaints': {
                'primary': ['terrible', 'worst', 'horrible', 'awful', 'disgusting', 'hate'],
                'secondary': ['bad', 'disappointed', 'angry', 'frustrated', 'upset']
            },
            'General_Inquiry': {
                'primary': ['question', 'info', 'information', 'contact', 'hours', 'location'],
                'secondary': ['inquiry', 'details', 'explain']
            }
        }
        
        print("\nSmart Intent Assignment with Balanced Scoring:")
        
        # Step 1: Score each cluster for each intent
        cluster_intent_scores = {}
        for cluster_id, profile in cluster_profiles.items():
            top_terms = profile['top_terms']
            sample_tweets = ' '.join(profile['sample_tweets']).lower()
            
            intent_scores = {}
            for intent, keywords in intent_keywords.items():
                # Primary keywords get higher weight
                primary_score = sum(3 for term in top_terms 
                                if any(keyword in term.lower() for keyword in keywords['primary']))
                
                # Secondary keywords get lower weight
                secondary_score = sum(1 for term in top_terms 
                                    if any(keyword in term.lower() for keyword in keywords['secondary']))
                
                # Bonus for keywords in sample tweets
                tweet_score = sum(1 for keyword in keywords['primary'] + keywords['secondary']
                                if keyword in sample_tweets) * 0.5
                
                total_score = primary_score + secondary_score + tweet_score
                intent_scores[intent] = total_score
            
            cluster_intent_scores[cluster_id] = intent_scores
        
        # Step 2: First pass assignment (highest scoring intent for each cluster)
        initial_assignment = {}
        for cluster_id, scores in cluster_intent_scores.items():
            if max(scores.values()) > 0:
                best_intent = max(scores, key=scores.get)
                initial_assignment[cluster_id] = best_intent
            else:
                initial_assignment[cluster_id] = 'Other'
        
        # Step 3: Balance check and redistribution
        intent_cluster_counts = {}
        for cluster_id, intent in initial_assignment.items():
            cluster_size = cluster_profiles[cluster_id]['size']
            if intent not in intent_cluster_counts:
                intent_cluster_counts[intent] = []
            intent_cluster_counts[intent].append((cluster_id, cluster_size))
        
        # Step 4: Rebalance if any intent has too many large clusters
        print("\nInitial Assignment:")
        total_tweets = sum(profile['size'] for profile in cluster_profiles.values())
        
        for intent, clusters in intent_cluster_counts.items():
            total_size = sum(size for _, size in clusters)
            percentage = (total_size / total_tweets) * 100
            print(f"   {intent}: {len(clusters)} clusters, {total_size:,} tweets ({percentage:.1f}%)")
        
        # Identify over-dominant intents (>40% of data)
        rebalanced_assignment = initial_assignment.copy()
        
        for intent, clusters in intent_cluster_counts.items():
            total_size = sum(size for _, size in clusters)
            percentage = (total_size / total_tweets) * 100
            
            if percentage > 40 and len(clusters) > 1:
                print(f"\nRebalancing '{intent}' ({percentage:.1f}% of data)")
                
                # Sort clusters by size (largest first)
                clusters.sort(key=lambda x: x[1], reverse=True)
                
                # Keep largest cluster with original intent
                largest_cluster = clusters[0]
                print(f"   Keeping cluster {largest_cluster[0]} ({largest_cluster[1]:,} tweets) as {intent}")
                
                # Reassign other clusters to secondary intents
                for i, (cluster_id, size) in enumerate(clusters[1:], 1):
                    # Find second-best intent for this cluster
                    scores = cluster_intent_scores[cluster_id]
                    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Try to assign to second-best intent, or create specific subtype
                    if len(sorted_intents) > 1 and sorted_intents[1][1] > 0:
                        new_intent = sorted_intents[1][0]
                    else:
                        # Create specific subtype
                        if intent == 'Technical_Issues':
                            new_intent = f'Technical_Issues_Type{i}'
                        else:
                            new_intent = f'{intent}_Secondary'
                    
                    rebalanced_assignment[cluster_id] = new_intent
                    print(f"   Reassigned cluster {cluster_id} ({size:,} tweets) to {new_intent}")
        
        # Step 5: Final assignment with cluster size consideration
        final_assignment = {}
        for cluster_id, intent in rebalanced_assignment.items():
            cluster_size = cluster_profiles[cluster_id]['size']
            top_terms = cluster_profiles[cluster_id]['top_terms']
            
            print(f"\n   Cluster {cluster_id} ({cluster_size:,} tweets) → {intent}")
            print(f"     Key terms: {', '.join(top_terms[:3])}")
            
            final_assignment[cluster_id] = intent
        
        return final_assignment
    
    
    def create_rule_based_intents(self):
        """Create intent labels using rule-based classification instead of clustering"""
        print("\n=== RULE-BASED INTENT CLASSIFICATION ===")
        print("Switching from clustering to keyword-based classification...")
        
        # Define comprehensive keyword rules
        intent_rules = {
            'Account_Issues': [
                # Login and authentication
                r'\b(login|log in|sign in|signin|password|username|authenticate)\b',
                r'\b(account|profile|verify|verification|reset|unlock|locked)\b',
                r'\b(two factor|2fa|security code|verification code)\b',
                r'\bcannot (access|login|sign in)\b',
                r'\bforgot (password|username)\b',
            ],
            
            'Billing_Problems': [
                # Payment and billing
                r'\b(bill|billing|charge|payment|invoice|subscription|refund)\b',
                r'\b(credit card|debit|bank|transaction|fee|cost|price|money)\b',
                r'\b(cancel|cancellation|unsubscribe|auto.?renew)\b',
                r'\b(double charged|charged twice|wrong amount)\b',
                r'\b(disputed? charge|fraudulent)\b',
            ],
            
            'Technical_Issues': [
                # App and technical problems
                r'\b(app|application|website|site|system|software)\b.*\b(crash|error|bug|freeze|slow|down|broken|not work)\b',
                r'\b(error|exception|crash|freeze|hang|stuck|loading|lag)\b',
                r'\b(404|500|timeout|connection|network|server)\b.*\b(error|problem|issue)\b',
                r'\b(update|upgrade|install|download).*\b(fail|error|problem)\b',
                r'\bnot (working|responding|loading)\b',
            ],
            
            'Service_Issues': [
                # Service availability
                r'\b(outage|down|offline|maintenance|unavailable|service interruption)\b',
                r'\b(connection|network|internet|wifi|connectivity)\b.*\b(issue|problem|trouble)\b',
                r'\bcannot connect\b',
                r'\bservice (down|unavailable|slow)\b',
            ],
            
            'Product_Questions': [
                # How-to and feature questions
                r'\b(how (do|to|can)|what (is|does)|where (do|can)|when (do|can)|why (do|can))\b',
                r'\b(feature|function|option|setting|tutorial|guide|help|instruction)\b',
                r'\b(explain|show|tell|demonstrate)\b.*\b(how|what|where|when)\b',
            ],
            
            'Delivery_Issues': [
                # Shipping and orders
                r'\b(order|delivery|shipping|ship|package|shipment|tracking)\b',
                r'\b(delivered|received|sent|mail|post|courier|fedex|ups|dhl)\b',
                r'\b(wrong (item|product|order)|damaged|missing|lost)\b',
                r'\btracking (number|code|info)\b',
            ],
            
            'Complaints': [
                # Negative sentiment
                r'\b(terrible|horrible|awful|worst|hate|disgusting|pathetic|useless)\b',
                r'\b(disappointed|frustrated|angry|upset|mad|furious)\b',
                r'\b(bad|poor|slow|rude|unprofessional)\b.*\b(service|support|experience)\b',
                r'\bnever (again|use|recommend)\b',
                r'\bwaste (of time|of money)\b',
            ],
            
            'Compliments': [
                # Positive sentiment
                r'\b(thank|thanks|grateful|appreciate|excellent|amazing|fantastic|wonderful)\b',
                r'\b(great|good|helpful|friendly|professional|quick|fast)\b.*\b(service|support|team|help)\b',
                r'\b(love|impressed|satisfied|happy)\b',
                r'\b(recommend|highly recommend|five stars|5 stars)\b',
            ]
        }
        
        # Apply rules to classify each tweet
        def classify_tweet(text):
            if pd.isna(text):
                return 'Other'
            
            text = str(text).lower()
            
            # Score each intent
            intent_scores = {}
            for intent, patterns in intent_rules.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    score += matches
                intent_scores[intent] = score
            
            # Return intent with highest score, or 'Other' if no matches
            if max(intent_scores.values()) > 0:
                return max(intent_scores, key=intent_scores.get)
            else:
                return 'General_Inquiry'
        
        # Apply classification
        print("Classifying tweets using rule-based approach...")
        self.df_inbound.loc[:, 'intent'] = self.df_inbound['text'].apply(classify_tweet)
        
        # Display results
        intent_dist = self.df_inbound['intent'].value_counts(normalize=True)
        print(f"\n📊 Rule-Based Intent Distribution:")
        for intent, ratio in intent_dist.items():
            count = self.df_inbound['intent'].value_counts()[intent]
            print(f"   {intent}: {ratio:.1%} ({count:,} tweets)")
        
        # Check balance
        max_intent_ratio = intent_dist.max()
        if max_intent_ratio > 0.4:
            print(f"\n⚠️  WARNING: '{intent_dist.idxmax()}' still dominates with {max_intent_ratio:.1%}")
        else:
            print(f"\n✅ Good balance! Largest intent is {max_intent_ratio:.1%}")
        
        # Show sample classifications
        print(f"\n📋 Sample Classifications:")
        for intent in intent_dist.index[:6]:  # Show top 6 intents
            samples = self.df_inbound[self.df_inbound['intent'] == intent]['text'].head(2)
            print(f"\n{intent}:")
            for i, tweet in enumerate(samples, 1):
                print(f"   {i}. {tweet[:100]}...")
        
        return self.df_inbound['intent']
    
    def save_results(self):
        """Save processed data and generate report"""
        print("\n=== SAVING RESULTS ===")
        
        # Save processed data
        processed_data_path = Path('../data/processed')
        processed_data_path.mkdir(parents=True, exist_ok=True)
        
        output_file = processed_data_path / 'tweets_with_intents.csv'
        self.df_inbound.to_csv(output_file, index=False)
        print(f"Processed data saved to: {output_file}")
        
        # Generate summary report
        self._generate_summary_report()
        
        return output_file
    
    def _generate_summary_report(self):
        """Generate comprehensive summary report"""
        intent_dist = self.df_inbound['intent'].value_counts()
        
        report = f"""
# Customer Service Tweet Intent Analysis - Summary Report

## Dataset Overview
- **Total tweets processed**: {len(self.df_inbound):,}
- **Optimal number of clusters**: {self.optimal_k}
- **TF-IDF features**: {self.tfidf_matrix.shape[1]:,}

## Intent Distribution
{intent_dist.to_string()}

## Cluster Quality Metrics
- **Silhouette Score**: Available in clustering analysis
- **Memory Usage**: Optimized for large datasets
- **Processing**: {'MiniBatch' if len(self.df_inbound) > 100000 else 'Standard'} KMeans

## Files Generated
1. `tweets_with_intents.csv` - Processed dataset with intent labels
2. `clustering_analysis.png` - Elbow method and silhouette analysis plots
3. `eda_summary_report.md` - This summary report

## Next Steps
1. Train classification models using intent labels
2. Develop prompt templates for each intent category
3. Implement RAG system for customer service responses
4. Add responsible AI monitoring and bias detection

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report_path = self.output_dir / 'eda_summary_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to: {report_path}")
    

    def run_full_pipeline(self):
        """Run the complete EDA pipeline with rule-based classification"""
        print("🚀 Starting Customer Service Tweet Intent Analysis Pipeline")
        print("=" * 70)
        
        try:
            # Step 1: Load and filter data
            self.load_and_filter_data()
            
            # Step 2: Preprocess text (minimal preprocessing for rule-based)
            self.preprocess_text()
            
            # Step 3: Check if clustering might work
            print("\n🔍 Evaluating data quality for clustering...")
            self.create_tfidf_features()
            
            # Quick clustering test
            test_kmeans = KMeans(n_clusters=5, random_state=42, n_init=1)
            test_labels = test_kmeans.fit_predict(self.tfidf_matrix)
            unique_labels = len(np.unique(test_labels))
            
            # Check clustering viability
            largest_cluster_ratio = np.max(np.bincount(test_labels)) / len(test_labels)
            
            if unique_labels < 3 or largest_cluster_ratio > 0.8:
                print(f"⚠️  CLUSTERING NOT VIABLE:")
                print(f"   - Only {unique_labels} unique clusters found")
                print(f"   - Largest cluster contains {largest_cluster_ratio:.1%} of data")
                print("   - Switching to rule-based approach...")
                
                # Use rule-based classification
                self.create_rule_based_intents()
                
            else:
                print("✅ Clustering seems viable, proceeding with clustering pipeline...")
                
                # Original clustering pipeline
                self.find_optimal_clusters()
                self.apply_final_clustering()
                self.analyze_clusters()
                self.assign_intent_labels()
            
            # Step 4: Save results
            output_file = self.save_results()
            
            print("\n" + "=" * 70)
            print("✅ PIPELINE COMPLETE!")
            print(f"✅ Processed data saved to: {output_file}")
            print("✅ Ready for Step 2: Model Training")
            
            return self.df_inbound
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise

def main():
    """Main entry point"""
    # Initialize and run EDA pipeline
    eda = CustomerServiceEDA()
    df_processed = eda.run_full_pipeline()
    
    return df_processed

if __name__ == "__main__":
    df_processed = main()