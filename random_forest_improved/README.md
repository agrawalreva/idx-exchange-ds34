### **Random Forest Model**

- **Algorithm**: Random Forest 
- **Hyperparameters**: 
  - 300 estimators
  - Max depth of 20
  - Optimized min_samples_split and min_samples_leaf
  - Feature sampling with sqrt

### **Evaluation Metrics**

- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- 5-fold cross-validation for robust evaluation


## Features Created

### Geographical (15+ features)
- Distance to 10 regional centers
- Agglomerative clustering (3 different cluster counts)
- Coordinate-based features
- Closest region distance

### Temporal (12+ features)
- Date components (year, month, day, quarter)
- Market timing features
- Seasonality indicators
- Escrow and market duration

### Encoding (10+ features)
- Frequency encoding for locations
- Mean encoding with smoothing
- Cluster mean encoding

### Transformations (8+ features)
- Square root transforms
- Squared features
- Density metrics

### Interactions (10+ features)
- Quality scores
- Amenity combinations
- Size-quality interactions
- Ratio features

### Local Market (6+ features)
- Postal code statistics
- Price relative to local market




