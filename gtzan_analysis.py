# GTZAN Music Genre Analysis with PCA and Clustering
# Author: [Your Name]
# Description: Comprehensive analysis of music genres using audio features, 
#              PCA dimensionality reduction, and various clustering techniques.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

# Scientific computing libraries
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GTZANAnalyzer:
    """
    A comprehensive analyzer for GTZAN music genre dataset.
    
    This class provides methods for:
    - Loading and preprocessing audio feature data
    - Exploratory data analysis with visualizations
    - PCA dimensionality reduction
    - Genre similarity analysis
    - Clustering analysis with multiple techniques
    - Network graph generation for external visualization tools
    """
    
    def __init__(self, data_path: str = 'features_30_sec.csv', output_dir: str = 'output'):
        """
        Initialize the GTZAN analyzer.
        
        Args:
            data_path: Path to the CSV file containing audio features
            output_dir: Directory to save output files and visualizations
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Core audio features for analysis
        self.audio_features = [
            'tempo', 'chroma_stft_mean', 'rms_mean',
            'spectral_centroid_mean', 'spectral_bandwidth_mean', 
            'rolloff_mean', 'zero_crossing_rate_mean', 
            'harmony_mean', 'perceptr_mean'
        ]
        
        # Visualization settings
        self.color_palette = 'husl'
        self.figure_size = (16, 9)
        self.dpi = 300
        
        # Data containers
        self.data = None
        self.processed_data = None
        self.pca_data = None
        self.pca_model = None
        self.scaler = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial data validation.
        
        Returns:
            DataFrame: Loaded dataset
        """
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {self.data.shape}")
            
            # Basic data validation
            if 'label' not in self.data.columns:
                raise ValueError("Dataset must contain a 'label' column for genres")
            
            missing_features = [f for f in self.audio_features if f not in self.data.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                self.audio_features = [f for f in self.audio_features if f in self.data.columns]
            
            logger.info(f"Available genres: {self.data['label'].unique()}")
            logger.info(f"Features to analyze: {len(self.audio_features)}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_correlation_heatmap(self, save: bool = True) -> None:
        """
        Create and save correlation heatmap for mean features.
        
        Args:
            save: Whether to save the plot to file
        """
        try:
            # Select features with 'mean' in column name
            mean_cols = [col for col in self.data.columns if 'mean' in col]
            corr_matrix = self.data[mean_cols].corr()
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Set up the matplotlib figure
            fig, ax = plt.subplots(figsize=(16, 11))
            
            # Generate custom colormap
            cmap = sns.diverging_palette(0, 25, as_cmap=True, s=90, l=45, n=5)
            
            # Create heatmap
            sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=0.3, center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=ax)
            
            ax.set_title('Correlation Heatmap (Mean Features)', fontsize=25)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            
            if save:
                plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=self.dpi, bbox_inches='tight')
                logger.info("Correlation heatmap saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            raise
    
    def create_feature_boxplots(self, features: Optional[List[str]] = None, save: bool = True) -> None:
        """
        Create boxplots for specified features across genres.
        
        Args:
            features: List of features to plot. If None, uses self.audio_features
            save: Whether to save plots to file
        """
        features = features or self.audio_features
        
        try:
            for feature in features:
                if feature not in self.data.columns:
                    logger.warning(f"Feature {feature} not found in dataset")
                    continue
                
                fig, ax = plt.subplots(figsize=self.figure_size)
                sns.boxplot(x="label", y=feature, data=self.data, 
                           palette=self.color_palette, ax=ax)
                
                ax.set_title(f'{feature.title()} Distribution by Genre', fontsize=20)
                ax.set_xlabel("Genre", fontsize=15)
                ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=15)
                plt.xticks(rotation=45, fontsize=12)
                plt.yticks(fontsize=10)
                
                if save:
                    filename = f"{feature}_boxplot.png"
                    plt.savefig(self.output_dir / filename, dpi=self.dpi, bbox_inches='tight')
                
                plt.show()
                
            logger.info(f"Created boxplots for {len(features)} features")
            
        except Exception as e:
            logger.error(f"Error creating boxplots: {e}")
            raise
    
    def perform_pca_analysis(self, n_components: int = 5) -> Tuple[pd.DataFrame, PCA]:
        """
        Perform PCA analysis on the audio features.
        
        Args:
            n_components: Number of principal components to extract
            
        Returns:
            Tuple of (PCA-transformed data, fitted PCA model)
        """
        try:
            # Prepare data
            y = self.data['label']
            X = self.data[self.audio_features]
            
            # Normalize features
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_scaled_df = pd.DataFrame(X_scaled, columns=self.audio_features)
            
            # Perform PCA
            self.pca_model = PCA(n_components=n_components)
            principal_components = self.pca_model.fit_transform(X_scaled)
            
            # Create DataFrame with PCA results
            pc_columns = [f'principal_component_{i+1}' for i in range(n_components)]
            self.pca_data = pd.DataFrame(data=principal_components, columns=pc_columns)
            self.pca_data = pd.concat([self.pca_data, y.reset_index(drop=True)], axis=1)
            
            # Log explained variance
            explained_variance = self.pca_model.explained_variance_ratio_
            total_variance = explained_variance.sum() * 100
            
            logger.info(f"PCA completed with {n_components} components")
            logger.info(f"Total explained variance: {total_variance:.2f}%")
            
            for i, var in enumerate(explained_variance):
                logger.info(f"PC{i+1}: {var*100:.2f}% variance")
            
            return self.pca_data, self.pca_model
            
        except Exception as e:
            logger.error(f"Error in PCA analysis: {e}")
            raise
    
    def visualize_pca_scatter(self, save: bool = True) -> None:
        """
        Create scatter plot of first two principal components.
        
        Args:
            save: Whether to save the plot to file
        """
        if self.pca_data is None:
            raise ValueError("PCA analysis must be performed first")
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            sns.scatterplot(x="principal_component_1", y="principal_component_2", 
                           data=self.pca_data, hue="label", alpha=0.7, s=100, ax=ax)
            
            ax.set_title('PCA Analysis of Music Genres', fontsize=25)
            ax.set_xlabel("Principal Component 1", fontsize=15)
            ax.set_ylabel("Principal Component 2", fontsize=15)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=10)
            
            if save:
                plt.savefig(self.output_dir / "pca_scatter.png", dpi=self.dpi, bbox_inches='tight')
                logger.info("PCA scatter plot saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error creating PCA scatter plot: {e}")
            raise
    
    def analyze_genre_similarities(self, save: bool = True) -> pd.DataFrame:
        """
        Analyze similarities between genres using centroid distances.
        
        Args:
            save: Whether to save the distance matrix plot
            
        Returns:
            DataFrame: Distance matrix between genre centroids
        """
        if self.pca_data is None:
            raise ValueError("PCA analysis must be performed first")
        
        try:
            # Calculate centroids for each genre
            pc_columns = [col for col in self.pca_data.columns if 'principal_component' in col]
            centroids = self.pca_data.groupby('label')[pc_columns].mean()
            
            # Compute pairwise distances
            labels = centroids.index
            distance_matrix = pd.DataFrame(index=labels, columns=labels)
            
            for label1 in labels:
                for label2 in labels:
                    dist = np.linalg.norm(centroids.loc[label1].values - centroids.loc[label2].values)
                    distance_matrix.loc[label1, label2] = dist
            
            distance_matrix = distance_matrix.astype(float)
            
            # Visualize distance matrix
            if save:
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax)
                ax.set_title("Genre Similarity Matrix (Euclidean Distance)", fontsize=16)
                plt.tight_layout()
                plt.savefig(self.output_dir / "genre_distance_matrix.png", dpi=self.dpi, bbox_inches='tight')
                plt.show()
            
            logger.info("Genre similarity analysis completed")
            return distance_matrix
            
        except Exception as e:
            logger.error(f"Error in genre similarity analysis: {e}")
            raise
    
    def perform_hierarchical_clustering(self, save: bool = True) -> None:
        """
        Perform hierarchical clustering on genre centroids.
        
        Args:
            save: Whether to save the dendrogram
        """
        if self.pca_data is None:
            raise ValueError("PCA analysis must be performed first")
        
        try:
            # Calculate centroids
            pc_columns = [col for col in self.pca_data.columns if 'principal_component' in col]
            centroids = self.pca_data.groupby('label')[pc_columns].mean()
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(centroids, method='ward')
            
            # Create dendrogram
            fig, ax = plt.subplots(figsize=(12, 8))
            dendrogram(linkage_matrix, labels=centroids.index.tolist(), 
                      orientation='top', distance_sort='ascending', 
                      show_leaf_counts=True, ax=ax)
            
            ax.set_title("Hierarchical Clustering of Music Genres", fontsize=16)
            ax.set_xlabel("Genre", fontsize=12)
            ax.set_ylabel("Euclidean Distance", fontsize=12)
            plt.xticks(rotation=45)
            
            if save:
                plt.savefig(self.output_dir / "genre_dendrogram.png", dpi=self.dpi, bbox_inches='tight')
                logger.info("Dendrogram saved")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
            raise
    
    def perform_kmeans_analysis(self, save: bool = True) -> Dict:
        """
        Perform K-means clustering analysis with silhouette analysis.
        
        Args:
            save: Whether to save the silhouette plot
            
        Returns:
            Dictionary containing clustering results and metrics
        """
        if self.pca_data is None:
            raise ValueError("PCA analysis must be performed first")
        
        try:
            # Prepare data
            pc_columns = [col for col in self.pca_data.columns if 'principal_component' in col]
            X_pca = self.pca_data[pc_columns].values
            n_clusters = self.pca_data['label'].nunique()
            
            # Perform K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_pca)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_pca, cluster_labels)
            sample_silhouette_values = silhouette_samples(X_pca, cluster_labels)
            
            # Create crosstab
            crosstab = pd.crosstab(self.pca_data['label'], cluster_labels)
            
            # Create cluster to genre mapping
            cluster_to_genre = {i: crosstab[i].idxmax() for i in range(n_clusters)}
            
            # Visualize silhouette analysis
            if save:
                self._plot_silhouette_analysis(X_pca, cluster_labels, sample_silhouette_values, 
                                             silhouette_avg, cluster_to_genre, n_clusters)
            
            results = {
                'silhouette_score': silhouette_avg,
                'cluster_labels': cluster_labels,
                'crosstab': crosstab,
                'cluster_to_genre': cluster_to_genre,
                'kmeans_model': kmeans
            }
            
            logger.info(f"K-means analysis completed. Silhouette score: {silhouette_avg:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in K-means analysis: {e}")
            raise
    
    def _plot_silhouette_analysis(self, X_pca: np.ndarray, cluster_labels: np.ndarray, 
                                sample_silhouette_values: np.ndarray, silhouette_avg: float,
                                cluster_to_genre: Dict, n_clusters: int) -> None:
        """Helper method to create silhouette analysis plot."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X_pca) + (n_clusters + 1) * 10])
        
        y_lower = 10
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                           0, cluster_silhouette_values,
                           facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, 
                   f"Cluster {i}\n({cluster_to_genre[i]})", fontsize=9)
            
            y_lower = y_upper + 10
        
        ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                  label=f'Average Score: {silhouette_avg:.3f}')
        ax.set_title("Silhouette Analysis for K-means Clustering", fontsize=14)
        ax.set_xlabel("Silhouette Coefficient Values", fontsize=12)
        ax.set_ylabel("Cluster Label", fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "silhouette_analysis.png", dpi=self.dpi, bbox_inches='tight')
        plt.show()
    
    def create_network_files(self, features: Optional[List[str]] = None) -> None:
        """
        Create network files for external visualization tools like Gephi.
        
        Args:
            features: List of features to include. If None, uses self.audio_features
        """
        features = features or self.audio_features
        
        try:
            # Create nodes
            nodes_tracks = self.data[['filename', 'label']].copy()
            nodes_tracks['Type'] = 'track'
            nodes_tracks = nodes_tracks.rename(columns={'filename': 'Id', 'label': 'Genre'})
            nodes_tracks['Label'] = nodes_tracks['Id']
            
            nodes_params = pd.DataFrame({
                'Id': features,
                'Label': features,
                'Type': 'parameter',
                'Genre': ''
            })
            
            nodes = pd.concat([nodes_tracks[['Id', 'Label', 'Type', 'Genre']], nodes_params], 
                            ignore_index=True)
            
            # Create edges
            edges_list = []
            for _, row in self.data.iterrows():
                track_id = row['filename']
                genre = row['label']
                for param in features:
                    if param in row:
                        edges_list.append({
                            'Source': track_id,
                            'Target': param,
                            'Weight': row[param],
                            'Genre': genre
                        })
            
            edges = pd.DataFrame(edges_list)
            
            # Save files
            nodes.to_csv(self.output_dir / "network_nodes.csv", index=False)
            edges.to_csv(self.output_dir / "network_edges.csv", index=False)
            
            logger.info("Network files saved for external visualization")
            
        except Exception as e:
            logger.error(f"Error creating network files: {e}")
            raise
    
    def generate_analysis_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            String containing the analysis report
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        report = []
        report.append("# GTZAN Music Genre Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset overview
        report.append("## Dataset Overview")
        report.append(f"- Total samples: {len(self.data)}")
        report.append(f"- Number of genres: {self.data['label'].nunique()}")
        report.append(f"- Genres: {', '.join(self.data['label'].unique())}")
        report.append(f"- Features analyzed: {len(self.audio_features)}")
        report.append("")
        
        # Genre distribution
        genre_counts = self.data['label'].value_counts()
        report.append("## Genre Distribution")
        for genre, count in genre_counts.items():
            report.append(f"- {genre}: {count} samples")
        report.append("")
        
        # PCA results
        if self.pca_model is not None:
            report.append("## PCA Analysis Results")
            report.append(f"- Components extracted: {self.pca_model.n_components_}")
            total_variance = self.pca_model.explained_variance_ratio_.sum() * 100
            report.append(f"- Total variance explained: {total_variance:.2f}%")
            
            for i, var in enumerate(self.pca_model.explained_variance_ratio_):
                report.append(f"- PC{i+1}: {var*100:.2f}% variance")
            report.append("")
        
        # Feature statistics
        report.append("## Feature Statistics")
        for feature in self.audio_features:
            if feature in self.data.columns:
                stats = self.data[feature].describe()
                report.append(f"### {feature.replace('_', ' ').title()}")
                report.append(f"- Mean: {stats['mean']:.3f}")
                report.append(f"- Std: {stats['std']:.3f}")
                report.append(f"- Range: {stats['min']:.3f} - {stats['max']:.3f}")
                report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.output_dir / "analysis_report.md", 'w') as f:
            f.write(report_text)
        
        logger.info("Analysis report generated")
        return report_text
    
    def run_complete_analysis(self) -> None:
        """
        Run the complete analysis pipeline.
        """
        logger.info("Starting complete GTZAN analysis...")
        
        try:
            # Load and explore data
            self.load_data()
            self.create_correlation_heatmap()
            self.create_feature_boxplots()
            
            # PCA analysis
            self.perform_pca_analysis()
            self.visualize_pca_scatter()
            
            # Similarity and clustering analysis
            self.analyze_genre_similarities()
            self.perform_hierarchical_clustering()
            self.perform_kmeans_analysis()
            
            # Create network files
            self.create_network_files()
            
            # Generate report
            report = self.generate_analysis_report()
            print(report)
            
            logger.info("Complete analysis finished successfully!")
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = GTZANAnalyzer(data_path='features_30_sec.csv', output_dir='analysis_output')
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    # Or run individual components
    # analyzer.load_data()
    # analyzer.perform_pca_analysis(n_components=5)
    # analyzer.visualize_pca_scatter()
    # results = analyzer.perform_kmeans_analysis()
