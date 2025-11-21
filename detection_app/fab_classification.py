"""
FAB Classification Detection Module

This module provides functions to classify blood leukemia samples into:
- ALL (Acute Lymphoblastic Leukemia): L1, L2, L3
- AML (Acute Myeloid Leukemia): M0-M7

FAB classification is based on morphological features of leukemic cells:
- Cell size
- Cell shape and regularity
- Vacuolization
- Chromatin pattern
- Nuclear-to-cytoplasmic ratio (N:C ratio)
- Cytoplasm color/staining
"""

import numpy as np
from typing import Dict, Tuple
from PIL import Image
from io import BytesIO


class FABClassifier:
    """
    Classify leukemia samples into FAB morphological subtypes.
    """
    
    # Feature thresholds for classification
    CELL_SIZE_THRESHOLD = 0.5  # Normalized cell size
    CHROMATIN_DENSITY_THRESHOLD = 0.6
    VACUOLE_THRESHOLD = 0.3
    NC_RATIO_THRESHOLD = 0.65
    
    @staticmethod
    def extract_morphological_features(image_source) -> Dict[str, float]:
        """
        Extract morphological features from blood smear image.
        
        Features extracted:
        - Mean cell size (normalized)
        - Cell size uniformity (std dev)
        - Chromatin density (darkness of nucleus)
        - Vacuolization ratio (light spots in cytoplasm)
        - Nuclear-to-cytoplasmic ratio
        - Staining intensity (overall brightness)
        - Cell circularity (shape regularity)
        
        Args:
            image_source: Path or file object to blood smear image
            
        Returns:
            Dictionary of extracted morphological features
        """
        # Load image
        if isinstance(image_source, str):
            img = Image.open(image_source).convert('RGB')
        else:
            image_source.seek(0)
            img = Image.open(BytesIO(image_source.read())).convert('RGB')
        
        img = img.resize((512, 512))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Convert to grayscale for analysis
        gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        
        # Extract features
        features = {}
        
        # 1. Overall brightness/staining intensity
        features['staining_intensity'] = float(np.mean(gray))
        
        # 2. Chromatin density (nucleus darkness)
        # Dark areas = high chromatin density
        dark_pixels = np.sum(gray < 0.4)
        features['chromatin_density'] = float(dark_pixels / gray.size)
        
        # 3. Cell size estimation via edge detection
        from scipy import ndimage
        edges = ndimage.sobel(gray)
        edge_pixels = np.sum(edges > np.mean(edges))
        features['estimated_cell_size'] = float(edge_pixels / gray.size)
        
        # 4. Cell size uniformity
        # Calculate local variance in regions
        region_size = 32
        variances = []
        for i in range(0, gray.shape[0] - region_size, region_size):
            for j in range(0, gray.shape[1] - region_size, region_size):
                region = gray[i:i+region_size, j:j+region_size]
                variances.append(np.var(region))
        features['size_uniformity'] = float(1.0 - np.std(variances) / (np.mean(variances) + 1e-6))
        
        # 5. Vacuolization (light spots in cytoplasm)
        light_pixels = np.sum(gray > 0.7)
        features['vacuolization'] = float(light_pixels / gray.size)
        
        # 6. Nuclear-to-cytoplasmic ratio estimation
        # Approximate by comparing dark (nucleus) to medium (cytoplasm) regions
        dark_area = np.sum(gray < 0.4)
        medium_area = np.sum((gray >= 0.4) & (gray < 0.7))
        if medium_area > 0:
            features['nc_ratio'] = float(dark_area / medium_area)
        else:
            features['nc_ratio'] = 1.0
        
        # 7. Cell circularity (shape regularity)
        # Based on edge smoothness
        from scipy.ndimage import gaussian_filter
        smoothed_edges = gaussian_filter(edges, sigma=2)
        edge_smoothness = np.sum(smoothed_edges > 0) / (np.sum(edges > 0) + 1e-6)
        features['circularity'] = float(min(edge_smoothness, 1.0))
        
        # 8. Contrast (intensity variation)
        features['contrast'] = float(np.std(gray))
        
        return features
    
    @staticmethod
    def classify_leukemia_type(features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify if sample is ALL, AML, CLL, CML, or Normal.
        
        Args:
            features: Dictionary of morphological features
            
        Returns:
            Tuple of (leukemia_type, confidence)
        """
        staining = features['staining_intensity']
        chromatin = features['chromatin_density']
        cell_size = features['estimated_cell_size']
        nc_ratio = features['nc_ratio']
        vacuole = features['vacuolization']
        
        # Classification logic based on morphological features
        scores = {
            'ALL': 0.0,
            'AML': 0.0,
            'CLL': 0.0,
            'CML': 0.0,
            'NORMAL': 0.0
        }
        
        # ALL: Medium cell size, high N:C ratio, variable chromatin
        if 0.4 <= cell_size <= 0.7 and 0.6 <= nc_ratio <= 1.0:
            scores['ALL'] += 0.4
        
        # AML: Larger cells, moderate N:C ratio, active chromatin
        if cell_size > 0.5 and 0.5 <= nc_ratio < 0.8:
            scores['AML'] += 0.4
        
        # CLL: Small cells, high N:C ratio, dense chromatin
        if cell_size < 0.4 and nc_ratio > 0.8 and chromatin > 0.5:
            scores['CLL'] += 0.4
        
        # CML: Myeloid features, intermediate size
        if 0.4 <= cell_size <= 0.6 and 0.4 <= nc_ratio <= 0.7:
            scores['CML'] += 0.3
        
        # NORMAL: Low chromatin density, uniform cells, low N:C ratio
        if chromatin < 0.3 and cell_size < 0.3 and nc_ratio < 0.5:
            scores['NORMAL'] += 0.5
        
        # Additional scoring factors
        if vacuole > 0.2:  # Vacuoles more common in ALL
            scores['ALL'] += 0.1
            scores['AML'] += 0.05
        
        if staining < 0.3:  # Very dark staining suggests abnormal cells
            scores['ALL'] += 0.1
            scores['AML'] += 0.1
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {k: 1.0/len(scores) for k in scores}
        
        # Return most probable type with confidence
        leukemia_type = max(scores.items(), key=lambda x: x[1])
        return leukemia_type[0], leukemia_type[1]
    
    @staticmethod
    def classify_all_fab(features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify ALL subtype: L1, L2, or L3.
        
        ALL Morphology:
        - L1: Small, uniform cells, scanty cytoplasm, clumpy chromatin
        - L2: Large, variable cells, moderate cytoplasm, dispersed chromatin
        - L3: Large cells with extensive vacuolization (Burkitt-like)
        
        Args:
            features: Dictionary of morphological features
            
        Returns:
            Tuple of (fab_classification, confidence)
        """
        cell_size = features['estimated_cell_size']
        uniformity = features['size_uniformity']
        vacuole = features['vacuolization']
        nc_ratio = features['nc_ratio']
        contrast = features['contrast']
        
        scores = {'L1': 0.0, 'L2': 0.0, 'L3': 0.0}
        
        # L1: Small, uniform cells with high N:C ratio
        if cell_size < 0.4 and uniformity > 0.6 and nc_ratio > 0.7:
            scores['L1'] += 0.5
        
        if vacuole < 0.15:  # Few vacuoles
            scores['L1'] += 0.2
        
        # L2: Larger, more variable cells
        if 0.4 <= cell_size <= 0.65 and uniformity < 0.7:
            scores['L2'] += 0.4
        
        if 0.1 <= vacuole <= 0.25:  # Some vacuolization
            scores['L2'] += 0.2
        
        # L3: Large cells with heavy vacuolization (Burkitt-like)
        if cell_size > 0.5 and vacuole > 0.25:
            scores['L3'] += 0.6
        
        if contrast > 0.2:  # Higher contrast from vacuoles
            scores['L3'] += 0.15
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {k: 1.0/3.0 for k in scores}
        
        fab_type = max(scores.items(), key=lambda x: x[1])
        return fab_type[0], fab_type[1]
    
    @staticmethod
    def classify_aml_fab(features: Dict[str, float]) -> Tuple[str, float]:
        """
        Classify AML subtype: M0 through M7.
        
        AML Morphology:
        - M0: Undifferentiated, no Auer rods
        - M1: Minimal myeloid differentiation
        - M2: Myeloid with differentiation
        - M3: Promyelocytic (APL) - abundant Auer rods, heavy granulation
        - M4: Myelomonocytic
        - M5: Monocytic/Monocytoid
        - M6: Erythroid (rare)
        - M7: Megakaryocytic (rare)
        
        Args:
            features: Dictionary of morphological features
            
        Returns:
            Tuple of (fab_classification, confidence)
        """
        cell_size = features['estimated_cell_size']
        chromatin = features['chromatin_density']
        nc_ratio = features['nc_ratio']
        staining = features['staining_intensity']
        circularity = features['circularity']
        
        scores = {
            'M0': 0.0, 'M1': 0.0, 'M2': 0.0, 'M3': 0.0,
            'M4': 0.0, 'M5': 0.0, 'M6': 0.0, 'M7': 0.0
        }
        
        # M0: Undifferentiated, small-medium, high NC ratio
        if cell_size < 0.5 and nc_ratio > 0.75:
            scores['M0'] += 0.35
        
        # M1: Minimal differentiation, medium cells
        if 0.4 <= cell_size <= 0.6 and 0.6 <= nc_ratio <= 0.8:
            scores['M1'] += 0.3
        
        # M2: Myeloid differentiation, medium-large cells
        if 0.5 <= cell_size <= 0.7 and 0.5 <= nc_ratio <= 0.7:
            scores['M2'] += 0.35
        
        # M3: Promyelocytic (APL) - heavy granulation (dark staining)
        # Characterized by abundant granules = high chromatin density
        if chromatin > 0.6 and staining < 0.4 and cell_size > 0.4:
            scores['M3'] += 0.5
        
        # M4: Myelomonocytic - intermediate features
        if 0.45 <= cell_size <= 0.65 and 0.5 <= nc_ratio <= 0.75:
            scores['M4'] += 0.25
        
        # M5: Monocytic - larger cells, lower NC ratio, moderate chromatin
        if cell_size > 0.55 and nc_ratio < 0.65 and chromatin < 0.5:
            scores['M5'] += 0.3
        
        # M6: Erythroid - usually has high hemoglobin/staining
        if staining > 0.55 and cell_size > 0.5:
            scores['M6'] += 0.15
        
        # M7: Megakaryocytic - large cells, irregular shape
        if cell_size > 0.65 and circularity < 0.6:
            scores['M7'] += 0.2
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            scores = {k: 1.0/8.0 for k in scores}
        
        fab_type = max(scores.items(), key=lambda x: x[1])
        return fab_type[0], fab_type[1]
    
    @classmethod
    def detect_fab_classification(cls, image_source) -> Dict[str, any]:
        """
        Complete FAB classification pipeline.
        
        Returns dictionary with:
        - leukemia_type: ALL, AML, CLL, CML, or NORMAL
        - leukemia_confidence: Confidence score
        - fab_classification: L1/L2/L3 for ALL, M0-M7 for AML
        - fab_confidence: Confidence score for FAB classification
        
        Args:
            image_source: Path or file object to blood smear image
            
        Returns:
            Dictionary with all classification results
        """
        # Extract morphological features
        features = cls.extract_morphological_features(image_source)
        
        # Classify leukemia type
        leukemia_type, leukemia_confidence = cls.classify_leukemia_type(features)
        
        # Classify FAB subtype based on leukemia type
        fab_classification = None
        fab_confidence = 0.0
        
        if leukemia_type == 'ALL':
            fab_classification, fab_confidence = cls.classify_all_fab(features)
        elif leukemia_type == 'AML':
            fab_classification, fab_confidence = cls.classify_aml_fab(features)
        
        return {
            'leukemia_type': leukemia_type,
            'leukemia_confidence': float(leukemia_confidence),
            'fab_classification': fab_classification,
            'fab_confidence': float(fab_confidence),
            'morphological_features': {k: float(v) for k, v in features.items()}
        }
