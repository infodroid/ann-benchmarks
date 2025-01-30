import pysolr
import requests
import os
import numpy as np  # Import numpy for array handling
from time import sleep
from typing import List, Dict
import json

from ..base.module import BaseANN


class SolrKNN(BaseANN):
    """Solr KNN search with automatic core and schema setup.

    Solr KNN is implemented with vector fields and HNSW (Hierarchical Navigable Small World) indexing for approximate nearest neighbor search.
    """

    def __init__(self, metric: str, dimension: int, index_options: dict, solr_url="http://localhost:8983/solr"):
        self.metric = metric
        self.dimension = dimension
        self.index_options = index_options
        self.num_candidates = 100
        self.solr_url = solr_url

        index_options_str = "-".join(sorted(f"{k}-{v}" for k, v in self.index_options.items()))
        self.index_name = f"{metric}-{dimension}-{index_options_str}"
        self.similarity_metric = self._vector_similarity_metric(metric)
        
        # Connect to Solr instance (replace with your Solr URL)
        self.client = pysolr.Solr(f"{solr_url}/{self.index_name}", timeout=30)
        
        self.batch_res = []
        
        # Create core and schema if they don't exist
        self._create_core_and_schema()

        # Check health status
        self._wait_for_health_status()

    def _vector_similarity_metric(self, metric: str):
        """Translate similarity metric to Solr-compatible format."""
        supported_metrics = {
            "angular": "cosine",
            "euclidean": "l2_norm",
        }
        if metric not in supported_metrics:
            raise NotImplementedError(f"{metric} is not implemented")
        return supported_metrics[metric]

    def _wait_for_health_status(self, wait_seconds=30, status="yellow"):
        """Wait for Solr to be ready."""
        for _ in range(wait_seconds):
            try:
                health = self.client.ping()
                print(f'Solr is ready: {health}')
                return
            except Exception:
                pass
            sleep(1)
        raise RuntimeError("Failed to connect to Solr")

    def _create_core_and_schema(self):
        """Create Solr core and schema if they do not exist."""
        # Step 1: Create core if it doesn't exist
        core_exists = self._check_core_exists()
        print(core_exists)
        if not core_exists:
            print(f"Core '{self.index_name}' does not exist, creating it.")
            self._create_core()

        # Step 2: Ensure schema is set up
        self._create_or_update_schema()

    def _check_core_exists(self):
        
        print("Waiting for Solr to be up...")
        sleep(10)  # Delay for 10 seconds
        
        """Check if the Solr core already exists."""
        response = requests.get(f"{self.solr_url}/admin/cores?action=STATUS&core={self.index_name}")
        return response.status_code == 200 and "error" not in response.text and "name" in response.text

    def _create_core(self):
        """Create a new Solr core."""
        response = requests.get(f"{self.solr_url}/admin/cores?action=CREATE&name={self.index_name}&instanceDir={self.index_name}&dataDir={self.index_name}&configSet=/opt/solr/server/solr/configsets/_default")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to create Solr core: {response.text}")
        else:
            print(f"Core '{self.index_name}' created successfully.")
    
    def _reload_core(self):
        response = requests.get(f"{self.solr_url}/admin/cores?action=RELOAD&core={self.index_name}")
        return response.status_code == 200 and "error" not in response.text and "name" in response.text

    def _create_or_update_schema(self):
        """Create or update the schema for the Solr core."""
        schema_url = f"{self.solr_url}/{self.index_name}/schema"
      
        print(f"Uploading schema to {schema_url}")

        # Example schema configuration
        schema_data = {
            "add-field-type": {
                 "name": "knn_vector",
                 "class": "solr.DenseVectorField",
                 "vectorDimension": self.dimension,
                 "similarityFunction": "cosine",
                 "knnAlgorithm": self.index_options.get("type", "hnsw"),
                 "hnswMaxConnections": self.index_options["m"],
                 "hnswBeamWidth":  self.index_options["ef_construction"]
            },
            "add-field": {
                "name": "id", "type": "string", "stored": "true", "indexed": "true"
            },
            "add-field": {
                "name": "vec", "type": "knn_vector", "stored": "true", "indexed": "true"            
            }
        }
        # Send the schema update request
        response = requests.post(schema_url, json=schema_data)
        self._reload_core()
        if response.status_code != 200:
            raise RuntimeError(f"Failed to update schema: {response.text}")
        print(f"Schema for core '{self.index_name}' created/updated successfully.")

    def fit(self, X: np.ndarray):
        """Index vectors in Solr."""
        print("Indexing ...")
        batch = []
        batch_size = 10000 
        for i, vec in enumerate(X):
            doc = {
                "id": str(i),
                "vec": vec.tolist(),
            }
            batch.append(doc)
            if len(batch) == batch_size:
                self.client.add(batch)
                print(f"Indexed {len(batch)} documents")
                batch = []   
        
        if len(batch) > 0:
            self.client.add(batch)
            print(f"Indexed last batch of {len(batch)} documents")
            batch = []
        
        self.client.commit()
        # self.client.optimize(waitFlush=True, waitSearcher=True)

    def set_query_arguments(self, num_candidates: int):
        """Set number of nearest neighbors to search for."""
        self.num_candidates = num_candidates

    def query(self, q: np.ndarray, n: int) -> List[int]:
        """Query Solr for nearest neighbors."""
        if n > self.num_candidates:
            raise ValueError("n must be smaller than num_candidates")
        vec_str = np.array2string(q, separator=', ')[1:-1]
        res = self.client.search(f"{{!knn f=vec topK={n}}}[{vec_str}]", params={"fl": "id", "rows": n})
        docs = res.docs
        return [int(hit['id']) for hit in docs]

    def batch_query(self, X: np.ndarray, n: int):
        """Batch query for KNN search."""
        self.batch_res = [self.query(q, n) for q in X]

    def get_batch_results(self):
        """Return batch query results."""
        return self.batch_res

    def __str__(self):
        return f"SolrKNN(index_options: {self.index_options}, num_candidates: {self.num_candidates})"

