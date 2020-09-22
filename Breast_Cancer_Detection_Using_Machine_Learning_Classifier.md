 # Import essential libraries 


```python
# import libraries
import pandas as pd # for data manupulation or analysis
import numpy as np # for numeric calculation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
```

# Data Load


```python
#Load breast cancer dataset
cancerdataset = pd.read_csv('cancerdataset.csv')
```

# Data Manupulation


```python
cancer_dataset
```




    {'data': array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
             1.189e-01],
            [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,
             8.902e-02],
            [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,
             8.758e-02],
            ...,
            [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,
             7.820e-02],
            [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,
             1.240e-01],
            [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,
             7.039e-02]]),
     'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
            1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
            1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
            0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
            1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
            1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
            1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
            1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
            1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
     'target_names': array(['malignant', 'benign'], dtype='<U9'),
     'DESCR': '.. _breast_cancer_dataset:\n\nBreast cancer wisconsin (diagnostic) dataset\n--------------------------------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 569\n\n    :Number of Attributes: 30 numeric, predictive attributes and the class\n\n    :Attribute Information:\n        - radius (mean of distances from center to points on the perimeter)\n        - texture (standard deviation of gray-scale values)\n        - perimeter\n        - area\n        - smoothness (local variation in radius lengths)\n        - compactness (perimeter^2 / area - 1.0)\n        - concavity (severity of concave portions of the contour)\n        - concave points (number of concave portions of the contour)\n        - symmetry \n        - fractal dimension ("coastline approximation" - 1)\n\n        The mean, standard error, and "worst" or largest (mean of the three\n        largest values) of these features were computed for each image,\n        resulting in 30 features.  For instance, field 3 is Mean Radius, field\n        13 is Radius SE, field 23 is Worst Radius.\n\n        - class:\n                - WDBC-Malignant\n                - WDBC-Benign\n\n    :Summary Statistics:\n\n    ===================================== ====== ======\n                                           Min    Max\n    ===================================== ====== ======\n    radius (mean):                        6.981  28.11\n    texture (mean):                       9.71   39.28\n    perimeter (mean):                     43.79  188.5\n    area (mean):                          143.5  2501.0\n    smoothness (mean):                    0.053  0.163\n    compactness (mean):                   0.019  0.345\n    concavity (mean):                     0.0    0.427\n    concave points (mean):                0.0    0.201\n    symmetry (mean):                      0.106  0.304\n    fractal dimension (mean):             0.05   0.097\n    radius (standard error):              0.112  2.873\n    texture (standard error):             0.36   4.885\n    perimeter (standard error):           0.757  21.98\n    area (standard error):                6.802  542.2\n    smoothness (standard error):          0.002  0.031\n    compactness (standard error):         0.002  0.135\n    concavity (standard error):           0.0    0.396\n    concave points (standard error):      0.0    0.053\n    symmetry (standard error):            0.008  0.079\n    fractal dimension (standard error):   0.001  0.03\n    radius (worst):                       7.93   36.04\n    texture (worst):                      12.02  49.54\n    perimeter (worst):                    50.41  251.2\n    area (worst):                         185.2  4254.0\n    smoothness (worst):                   0.071  0.223\n    compactness (worst):                  0.027  1.058\n    concavity (worst):                    0.0    1.252\n    concave points (worst):               0.0    0.291\n    symmetry (worst):                     0.156  0.664\n    fractal dimension (worst):            0.055  0.208\n    ===================================== ====== ======\n\n    :Missing Attribute Values: None\n\n    :Class Distribution: 212 - Malignant, 357 - Benign\n\n    :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian\n\n    :Donor: Nick Street\n\n    :Date: November, 1995\n\nThis is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.\nhttps://goo.gl/U2Uwz2\n\nFeatures are computed from a digitized image of a fine needle\naspirate (FNA) of a breast mass.  They describe\ncharacteristics of the cell nuclei present in the image.\n\nSeparating plane described above was obtained using\nMultisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree\nConstruction Via Linear Programming." Proceedings of the 4th\nMidwest Artificial Intelligence and Cognitive Science Society,\npp. 97-101, 1992], a classification method which uses linear\nprogramming to construct a decision tree.  Relevant features\nwere selected using an exhaustive search in the space of 1-4\nfeatures and 1-3 separating planes.\n\nThe actual linear program used to obtain the separating plane\nin the 3-dimensional space is that described in:\n[K. P. Bennett and O. L. Mangasarian: "Robust Linear\nProgramming Discrimination of Two Linearly Inseparable Sets",\nOptimization Methods and Software 1, 1992, 23-34].\n\nThis database is also available through the UW CS ftp server:\n\nftp ftp.cs.wisc.edu\ncd math-prog/cpo-dataset/machine-learn/WDBC/\n\n.. topic:: References\n\n   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction \n     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on \n     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,\n     San Jose, CA, 1993.\n   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and \n     prognosis via linear programming. Operations Research, 43(4), pages 570-577, \n     July-August 1995.\n   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques\n     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) \n     163-171.',
     'feature_names': array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error',
            'fractal dimension error', 'worst radius', 'worst texture',
            'worst perimeter', 'worst area', 'worst smoothness',
            'worst compactness', 'worst concavity', 'worst concave points',
            'worst symmetry', 'worst fractal dimension'], dtype='<U23'),
     'filename': 'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\datasets\\data\\breast_cancer.csv'}




```python
type(cancer_dataset)
```




    sklearn.utils.Bunch




```python
# keys in dataset
cancer_dataset.keys()
```




    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])




```python
# featurs of each cells in numeric format
cancer_dataset['data']
```




    array([[1.799e+01, 1.038e+01, 1.228e+02, ..., 2.654e-01, 4.601e-01,
            1.189e-01],
           [2.057e+01, 1.777e+01, 1.329e+02, ..., 1.860e-01, 2.750e-01,
            8.902e-02],
           [1.969e+01, 2.125e+01, 1.300e+02, ..., 2.430e-01, 3.613e-01,
            8.758e-02],
           ...,
           [1.660e+01, 2.808e+01, 1.083e+02, ..., 1.418e-01, 2.218e-01,
            7.820e-02],
           [2.060e+01, 2.933e+01, 1.401e+02, ..., 2.650e-01, 4.087e-01,
            1.240e-01],
           [7.760e+00, 2.454e+01, 4.792e+01, ..., 0.000e+00, 2.871e-01,
            7.039e-02]])




```python
type(cancer_dataset['data'])
```




    numpy.ndarray




```python
# malignant or benign value
cancer_dataset['target']
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
           1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
           1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
           0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
           1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
           0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
           0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
           1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
           1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])




```python
# target value name malignant or benign tumor
cancer_dataset['target_names']
```




    array(['malignant', 'benign'], dtype='<U9')




```python
# description of data
print(cancer_dataset['DESCR'])
```

    .. _breast_cancer_dataset:
    
    Breast cancer wisconsin (diagnostic) dataset
    --------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    .. topic:: References
    
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.
    


```python
# name of features
print(cancer_dataset['feature_names'])
```

    ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness' 'mean compactness' 'mean concavity'
     'mean concave points' 'mean symmetry' 'mean fractal dimension'
     'radius error' 'texture error' 'perimeter error' 'area error'
     'smoothness error' 'compactness error' 'concavity error'
     'concave points error' 'symmetry error' 'fractal dimension error'
     'worst radius' 'worst texture' 'worst perimeter' 'worst area'
     'worst smoothness' 'worst compactness' 'worst concavity'
     'worst concave points' 'worst symmetry' 'worst fractal dimension']
    


```python
# location/path of data file
print(cancer_dataset['filename'])
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\datasets\data\breast_cancer.csv
    

## Create DataFrame


```python
# create datafrmae
cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
             columns = np.append(cancer_dataset['feature_names'], ['target']))
```


```python
# DataFrame to CSV file
cancer_df.to_csv('BREAST CANCER.csv')
```


```python
# Head of cancer DataFrame
cancer_df.head(6) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.45</td>
      <td>15.70</td>
      <td>82.57</td>
      <td>477.1</td>
      <td>0.12780</td>
      <td>0.17000</td>
      <td>0.1578</td>
      <td>0.08089</td>
      <td>0.2087</td>
      <td>0.07613</td>
      <td>...</td>
      <td>23.75</td>
      <td>103.40</td>
      <td>741.6</td>
      <td>0.1791</td>
      <td>0.5249</td>
      <td>0.5355</td>
      <td>0.1741</td>
      <td>0.3985</td>
      <td>0.12440</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 31 columns</p>
</div>




```python
# Tail of cancer DataFrame
cancer_df.tail(6) 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>563</th>
      <td>20.92</td>
      <td>25.09</td>
      <td>143.00</td>
      <td>1347.0</td>
      <td>0.10990</td>
      <td>0.22360</td>
      <td>0.31740</td>
      <td>0.14740</td>
      <td>0.2149</td>
      <td>0.06879</td>
      <td>...</td>
      <td>29.41</td>
      <td>179.10</td>
      <td>1819.0</td>
      <td>0.14070</td>
      <td>0.41860</td>
      <td>0.6599</td>
      <td>0.2542</td>
      <td>0.2929</td>
      <td>0.09873</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>564</th>
      <td>21.56</td>
      <td>22.39</td>
      <td>142.00</td>
      <td>1479.0</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.24390</td>
      <td>0.13890</td>
      <td>0.1726</td>
      <td>0.05623</td>
      <td>...</td>
      <td>26.40</td>
      <td>166.10</td>
      <td>2027.0</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.4107</td>
      <td>0.2216</td>
      <td>0.2060</td>
      <td>0.07115</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>565</th>
      <td>20.13</td>
      <td>28.25</td>
      <td>131.20</td>
      <td>1261.0</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.14400</td>
      <td>0.09791</td>
      <td>0.1752</td>
      <td>0.05533</td>
      <td>...</td>
      <td>38.25</td>
      <td>155.00</td>
      <td>1731.0</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.3215</td>
      <td>0.1628</td>
      <td>0.2572</td>
      <td>0.06637</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>566</th>
      <td>16.60</td>
      <td>28.08</td>
      <td>108.30</td>
      <td>858.1</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.09251</td>
      <td>0.05302</td>
      <td>0.1590</td>
      <td>0.05648</td>
      <td>...</td>
      <td>34.12</td>
      <td>126.70</td>
      <td>1124.0</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.3403</td>
      <td>0.1418</td>
      <td>0.2218</td>
      <td>0.07820</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>567</th>
      <td>20.60</td>
      <td>29.33</td>
      <td>140.10</td>
      <td>1265.0</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.35140</td>
      <td>0.15200</td>
      <td>0.2397</td>
      <td>0.07016</td>
      <td>...</td>
      <td>39.42</td>
      <td>184.60</td>
      <td>1821.0</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.9387</td>
      <td>0.2650</td>
      <td>0.4087</td>
      <td>0.12400</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>568</th>
      <td>7.76</td>
      <td>24.54</td>
      <td>47.92</td>
      <td>181.0</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.1587</td>
      <td>0.05884</td>
      <td>...</td>
      <td>30.37</td>
      <td>59.16</td>
      <td>268.6</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2871</td>
      <td>0.07039</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 31 columns</p>
</div>




```python
# Information of cancer Dataframe
cancer_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 31 columns):
    mean radius                569 non-null float64
    mean texture               569 non-null float64
    mean perimeter             569 non-null float64
    mean area                  569 non-null float64
    mean smoothness            569 non-null float64
    mean compactness           569 non-null float64
    mean concavity             569 non-null float64
    mean concave points        569 non-null float64
    mean symmetry              569 non-null float64
    mean fractal dimension     569 non-null float64
    radius error               569 non-null float64
    texture error              569 non-null float64
    perimeter error            569 non-null float64
    area error                 569 non-null float64
    smoothness error           569 non-null float64
    compactness error          569 non-null float64
    concavity error            569 non-null float64
    concave points error       569 non-null float64
    symmetry error             569 non-null float64
    fractal dimension error    569 non-null float64
    worst radius               569 non-null float64
    worst texture              569 non-null float64
    worst perimeter            569 non-null float64
    worst area                 569 non-null float64
    worst smoothness           569 non-null float64
    worst compactness          569 non-null float64
    worst concavity            569 non-null float64
    worst concave points       569 non-null float64
    worst symmetry             569 non-null float64
    worst fractal dimension    569 non-null float64
    target                     569 non-null float64
    dtypes: float64(31)
    memory usage: 137.9 KB
    


```python
# Numerical distribution of data
cancer_df.describe() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>...</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
      <td>569.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>0.181162</td>
      <td>0.062798</td>
      <td>...</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
      <td>0.627417</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>0.027414</td>
      <td>0.007060</td>
      <td>...</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
      <td>0.483918</td>
    </tr>
    <tr>
      <th>min</th>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.106000</td>
      <td>0.049960</td>
      <td>...</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>0.161900</td>
      <td>0.057700</td>
      <td>...</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>0.179200</td>
      <td>0.061540</td>
      <td>...</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>0.195700</td>
      <td>0.066120</td>
      <td>...</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>0.304000</td>
      <td>0.097440</td>
      <td>...</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
cancer_df.isnull().sum()
```




    mean radius                0
    mean texture               0
    mean perimeter             0
    mean area                  0
    mean smoothness            0
    mean compactness           0
    mean concavity             0
    mean concave points        0
    mean symmetry              0
    mean fractal dimension     0
    radius error               0
    texture error              0
    perimeter error            0
    area error                 0
    smoothness error           0
    compactness error          0
    concavity error            0
    concave points error       0
    symmetry error             0
    fractal dimension error    0
    worst radius               0
    worst texture              0
    worst perimeter            0
    worst area                 0
    worst smoothness           0
    worst compactness          0
    worst concavity            0
    worst concave points       0
    worst symmetry             0
    worst fractal dimension    0
    target                     0
    dtype: int64



# Data Visualization


```python
# Paiplot of cancer dataframe
sns.pairplot(cancer_df, hue = 'target') 
```

    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\nonparametric\kde.py:488: RuntimeWarning: invalid value encountered in true_divide
      binned = fast_linbin(X, a, b, gridsize) / (delta * nobs)
    C:\ProgramData\Anaconda3\lib\site-packages\statsmodels\nonparametric\kdetools.py:34: RuntimeWarning: invalid value encountered in double_scalars
      FAC1 = 2*(np.pi*bw/RANGE)**2
    




    <seaborn.axisgrid.PairGrid at 0x2723ae3c2e8>




![png](output_25_2.png)



```python
# Count the target class
sns.countplot(cancer_df['target']) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27272ef3898>




![png](output_26_1.png)



```python
# counter plot of feature mean radius
plt.figure(figsize = (20,8))
sns.countplot(cancer_df['mean radius']) # *** img 7 ****
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27272ebdb00>




![png](output_27_1.png)


# Heatmap


```python
# heatmap of DataFrame
plt.figure(figsize=(16,9))
sns.heatmap(cancer_df) # **** img 8 ****
```




    <matplotlib.axes._subplots.AxesSubplot at 0x272733bcd30>




![png](output_29_1.png)


##  Heatmap of a correlation matrix 


```python
cancer_df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean radius</th>
      <td>1.000000</td>
      <td>0.323782</td>
      <td>0.997855</td>
      <td>0.987357</td>
      <td>0.170581</td>
      <td>0.506124</td>
      <td>0.676764</td>
      <td>0.822529</td>
      <td>0.147741</td>
      <td>-0.311631</td>
      <td>...</td>
      <td>0.297008</td>
      <td>0.965137</td>
      <td>0.941082</td>
      <td>0.119616</td>
      <td>0.413463</td>
      <td>0.526911</td>
      <td>0.744214</td>
      <td>0.163953</td>
      <td>0.007066</td>
      <td>-0.730029</td>
    </tr>
    <tr>
      <th>mean texture</th>
      <td>0.323782</td>
      <td>1.000000</td>
      <td>0.329533</td>
      <td>0.321086</td>
      <td>-0.023389</td>
      <td>0.236702</td>
      <td>0.302418</td>
      <td>0.293464</td>
      <td>0.071401</td>
      <td>-0.076437</td>
      <td>...</td>
      <td>0.912045</td>
      <td>0.358040</td>
      <td>0.343546</td>
      <td>0.077503</td>
      <td>0.277830</td>
      <td>0.301025</td>
      <td>0.295316</td>
      <td>0.105008</td>
      <td>0.119205</td>
      <td>-0.415185</td>
    </tr>
    <tr>
      <th>mean perimeter</th>
      <td>0.997855</td>
      <td>0.329533</td>
      <td>1.000000</td>
      <td>0.986507</td>
      <td>0.207278</td>
      <td>0.556936</td>
      <td>0.716136</td>
      <td>0.850977</td>
      <td>0.183027</td>
      <td>-0.261477</td>
      <td>...</td>
      <td>0.303038</td>
      <td>0.970387</td>
      <td>0.941550</td>
      <td>0.150549</td>
      <td>0.455774</td>
      <td>0.563879</td>
      <td>0.771241</td>
      <td>0.189115</td>
      <td>0.051019</td>
      <td>-0.742636</td>
    </tr>
    <tr>
      <th>mean area</th>
      <td>0.987357</td>
      <td>0.321086</td>
      <td>0.986507</td>
      <td>1.000000</td>
      <td>0.177028</td>
      <td>0.498502</td>
      <td>0.685983</td>
      <td>0.823269</td>
      <td>0.151293</td>
      <td>-0.283110</td>
      <td>...</td>
      <td>0.287489</td>
      <td>0.959120</td>
      <td>0.959213</td>
      <td>0.123523</td>
      <td>0.390410</td>
      <td>0.512606</td>
      <td>0.722017</td>
      <td>0.143570</td>
      <td>0.003738</td>
      <td>-0.708984</td>
    </tr>
    <tr>
      <th>mean smoothness</th>
      <td>0.170581</td>
      <td>-0.023389</td>
      <td>0.207278</td>
      <td>0.177028</td>
      <td>1.000000</td>
      <td>0.659123</td>
      <td>0.521984</td>
      <td>0.553695</td>
      <td>0.557775</td>
      <td>0.584792</td>
      <td>...</td>
      <td>0.036072</td>
      <td>0.238853</td>
      <td>0.206718</td>
      <td>0.805324</td>
      <td>0.472468</td>
      <td>0.434926</td>
      <td>0.503053</td>
      <td>0.394309</td>
      <td>0.499316</td>
      <td>-0.358560</td>
    </tr>
    <tr>
      <th>mean compactness</th>
      <td>0.506124</td>
      <td>0.236702</td>
      <td>0.556936</td>
      <td>0.498502</td>
      <td>0.659123</td>
      <td>1.000000</td>
      <td>0.883121</td>
      <td>0.831135</td>
      <td>0.602641</td>
      <td>0.565369</td>
      <td>...</td>
      <td>0.248133</td>
      <td>0.590210</td>
      <td>0.509604</td>
      <td>0.565541</td>
      <td>0.865809</td>
      <td>0.816275</td>
      <td>0.815573</td>
      <td>0.510223</td>
      <td>0.687382</td>
      <td>-0.596534</td>
    </tr>
    <tr>
      <th>mean concavity</th>
      <td>0.676764</td>
      <td>0.302418</td>
      <td>0.716136</td>
      <td>0.685983</td>
      <td>0.521984</td>
      <td>0.883121</td>
      <td>1.000000</td>
      <td>0.921391</td>
      <td>0.500667</td>
      <td>0.336783</td>
      <td>...</td>
      <td>0.299879</td>
      <td>0.729565</td>
      <td>0.675987</td>
      <td>0.448822</td>
      <td>0.754968</td>
      <td>0.884103</td>
      <td>0.861323</td>
      <td>0.409464</td>
      <td>0.514930</td>
      <td>-0.696360</td>
    </tr>
    <tr>
      <th>mean concave points</th>
      <td>0.822529</td>
      <td>0.293464</td>
      <td>0.850977</td>
      <td>0.823269</td>
      <td>0.553695</td>
      <td>0.831135</td>
      <td>0.921391</td>
      <td>1.000000</td>
      <td>0.462497</td>
      <td>0.166917</td>
      <td>...</td>
      <td>0.292752</td>
      <td>0.855923</td>
      <td>0.809630</td>
      <td>0.452753</td>
      <td>0.667454</td>
      <td>0.752399</td>
      <td>0.910155</td>
      <td>0.375744</td>
      <td>0.368661</td>
      <td>-0.776614</td>
    </tr>
    <tr>
      <th>mean symmetry</th>
      <td>0.147741</td>
      <td>0.071401</td>
      <td>0.183027</td>
      <td>0.151293</td>
      <td>0.557775</td>
      <td>0.602641</td>
      <td>0.500667</td>
      <td>0.462497</td>
      <td>1.000000</td>
      <td>0.479921</td>
      <td>...</td>
      <td>0.090651</td>
      <td>0.219169</td>
      <td>0.177193</td>
      <td>0.426675</td>
      <td>0.473200</td>
      <td>0.433721</td>
      <td>0.430297</td>
      <td>0.699826</td>
      <td>0.438413</td>
      <td>-0.330499</td>
    </tr>
    <tr>
      <th>mean fractal dimension</th>
      <td>-0.311631</td>
      <td>-0.076437</td>
      <td>-0.261477</td>
      <td>-0.283110</td>
      <td>0.584792</td>
      <td>0.565369</td>
      <td>0.336783</td>
      <td>0.166917</td>
      <td>0.479921</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.051269</td>
      <td>-0.205151</td>
      <td>-0.231854</td>
      <td>0.504942</td>
      <td>0.458798</td>
      <td>0.346234</td>
      <td>0.175325</td>
      <td>0.334019</td>
      <td>0.767297</td>
      <td>0.012838</td>
    </tr>
    <tr>
      <th>radius error</th>
      <td>0.679090</td>
      <td>0.275869</td>
      <td>0.691765</td>
      <td>0.732562</td>
      <td>0.301467</td>
      <td>0.497473</td>
      <td>0.631925</td>
      <td>0.698050</td>
      <td>0.303379</td>
      <td>0.000111</td>
      <td>...</td>
      <td>0.194799</td>
      <td>0.719684</td>
      <td>0.751548</td>
      <td>0.141919</td>
      <td>0.287103</td>
      <td>0.380585</td>
      <td>0.531062</td>
      <td>0.094543</td>
      <td>0.049559</td>
      <td>-0.567134</td>
    </tr>
    <tr>
      <th>texture error</th>
      <td>-0.097317</td>
      <td>0.386358</td>
      <td>-0.086761</td>
      <td>-0.066280</td>
      <td>0.068406</td>
      <td>0.046205</td>
      <td>0.076218</td>
      <td>0.021480</td>
      <td>0.128053</td>
      <td>0.164174</td>
      <td>...</td>
      <td>0.409003</td>
      <td>-0.102242</td>
      <td>-0.083195</td>
      <td>-0.073658</td>
      <td>-0.092439</td>
      <td>-0.068956</td>
      <td>-0.119638</td>
      <td>-0.128215</td>
      <td>-0.045655</td>
      <td>0.008303</td>
    </tr>
    <tr>
      <th>perimeter error</th>
      <td>0.674172</td>
      <td>0.281673</td>
      <td>0.693135</td>
      <td>0.726628</td>
      <td>0.296092</td>
      <td>0.548905</td>
      <td>0.660391</td>
      <td>0.710650</td>
      <td>0.313893</td>
      <td>0.039830</td>
      <td>...</td>
      <td>0.200371</td>
      <td>0.721031</td>
      <td>0.730713</td>
      <td>0.130054</td>
      <td>0.341919</td>
      <td>0.418899</td>
      <td>0.554897</td>
      <td>0.109930</td>
      <td>0.085433</td>
      <td>-0.556141</td>
    </tr>
    <tr>
      <th>area error</th>
      <td>0.735864</td>
      <td>0.259845</td>
      <td>0.744983</td>
      <td>0.800086</td>
      <td>0.246552</td>
      <td>0.455653</td>
      <td>0.617427</td>
      <td>0.690299</td>
      <td>0.223970</td>
      <td>-0.090170</td>
      <td>...</td>
      <td>0.196497</td>
      <td>0.761213</td>
      <td>0.811408</td>
      <td>0.125389</td>
      <td>0.283257</td>
      <td>0.385100</td>
      <td>0.538166</td>
      <td>0.074126</td>
      <td>0.017539</td>
      <td>-0.548236</td>
    </tr>
    <tr>
      <th>smoothness error</th>
      <td>-0.222600</td>
      <td>0.006614</td>
      <td>-0.202694</td>
      <td>-0.166777</td>
      <td>0.332375</td>
      <td>0.135299</td>
      <td>0.098564</td>
      <td>0.027653</td>
      <td>0.187321</td>
      <td>0.401964</td>
      <td>...</td>
      <td>-0.074743</td>
      <td>-0.217304</td>
      <td>-0.182195</td>
      <td>0.314457</td>
      <td>-0.055558</td>
      <td>-0.058298</td>
      <td>-0.102007</td>
      <td>-0.107342</td>
      <td>0.101480</td>
      <td>0.067016</td>
    </tr>
    <tr>
      <th>compactness error</th>
      <td>0.206000</td>
      <td>0.191975</td>
      <td>0.250744</td>
      <td>0.212583</td>
      <td>0.318943</td>
      <td>0.738722</td>
      <td>0.670279</td>
      <td>0.490424</td>
      <td>0.421659</td>
      <td>0.559837</td>
      <td>...</td>
      <td>0.143003</td>
      <td>0.260516</td>
      <td>0.199371</td>
      <td>0.227394</td>
      <td>0.678780</td>
      <td>0.639147</td>
      <td>0.483208</td>
      <td>0.277878</td>
      <td>0.590973</td>
      <td>-0.292999</td>
    </tr>
    <tr>
      <th>concavity error</th>
      <td>0.194204</td>
      <td>0.143293</td>
      <td>0.228082</td>
      <td>0.207660</td>
      <td>0.248396</td>
      <td>0.570517</td>
      <td>0.691270</td>
      <td>0.439167</td>
      <td>0.342627</td>
      <td>0.446630</td>
      <td>...</td>
      <td>0.100241</td>
      <td>0.226680</td>
      <td>0.188353</td>
      <td>0.168481</td>
      <td>0.484858</td>
      <td>0.662564</td>
      <td>0.440472</td>
      <td>0.197788</td>
      <td>0.439329</td>
      <td>-0.253730</td>
    </tr>
    <tr>
      <th>concave points error</th>
      <td>0.376169</td>
      <td>0.163851</td>
      <td>0.407217</td>
      <td>0.372320</td>
      <td>0.380676</td>
      <td>0.642262</td>
      <td>0.683260</td>
      <td>0.615634</td>
      <td>0.393298</td>
      <td>0.341198</td>
      <td>...</td>
      <td>0.086741</td>
      <td>0.394999</td>
      <td>0.342271</td>
      <td>0.215351</td>
      <td>0.452888</td>
      <td>0.549592</td>
      <td>0.602450</td>
      <td>0.143116</td>
      <td>0.310655</td>
      <td>-0.408042</td>
    </tr>
    <tr>
      <th>symmetry error</th>
      <td>-0.104321</td>
      <td>0.009127</td>
      <td>-0.081629</td>
      <td>-0.072497</td>
      <td>0.200774</td>
      <td>0.229977</td>
      <td>0.178009</td>
      <td>0.095351</td>
      <td>0.449137</td>
      <td>0.345007</td>
      <td>...</td>
      <td>-0.077473</td>
      <td>-0.103753</td>
      <td>-0.110343</td>
      <td>-0.012662</td>
      <td>0.060255</td>
      <td>0.037119</td>
      <td>-0.030413</td>
      <td>0.389402</td>
      <td>0.078079</td>
      <td>0.006522</td>
    </tr>
    <tr>
      <th>fractal dimension error</th>
      <td>-0.042641</td>
      <td>0.054458</td>
      <td>-0.005523</td>
      <td>-0.019887</td>
      <td>0.283607</td>
      <td>0.507318</td>
      <td>0.449301</td>
      <td>0.257584</td>
      <td>0.331786</td>
      <td>0.688132</td>
      <td>...</td>
      <td>-0.003195</td>
      <td>-0.001000</td>
      <td>-0.022736</td>
      <td>0.170568</td>
      <td>0.390159</td>
      <td>0.379975</td>
      <td>0.215204</td>
      <td>0.111094</td>
      <td>0.591328</td>
      <td>-0.077972</td>
    </tr>
    <tr>
      <th>worst radius</th>
      <td>0.969539</td>
      <td>0.352573</td>
      <td>0.969476</td>
      <td>0.962746</td>
      <td>0.213120</td>
      <td>0.535315</td>
      <td>0.688236</td>
      <td>0.830318</td>
      <td>0.185728</td>
      <td>-0.253691</td>
      <td>...</td>
      <td>0.359921</td>
      <td>0.993708</td>
      <td>0.984015</td>
      <td>0.216574</td>
      <td>0.475820</td>
      <td>0.573975</td>
      <td>0.787424</td>
      <td>0.243529</td>
      <td>0.093492</td>
      <td>-0.776454</td>
    </tr>
    <tr>
      <th>worst texture</th>
      <td>0.297008</td>
      <td>0.912045</td>
      <td>0.303038</td>
      <td>0.287489</td>
      <td>0.036072</td>
      <td>0.248133</td>
      <td>0.299879</td>
      <td>0.292752</td>
      <td>0.090651</td>
      <td>-0.051269</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.365098</td>
      <td>0.345842</td>
      <td>0.225429</td>
      <td>0.360832</td>
      <td>0.368366</td>
      <td>0.359755</td>
      <td>0.233027</td>
      <td>0.219122</td>
      <td>-0.456903</td>
    </tr>
    <tr>
      <th>worst perimeter</th>
      <td>0.965137</td>
      <td>0.358040</td>
      <td>0.970387</td>
      <td>0.959120</td>
      <td>0.238853</td>
      <td>0.590210</td>
      <td>0.729565</td>
      <td>0.855923</td>
      <td>0.219169</td>
      <td>-0.205151</td>
      <td>...</td>
      <td>0.365098</td>
      <td>1.000000</td>
      <td>0.977578</td>
      <td>0.236775</td>
      <td>0.529408</td>
      <td>0.618344</td>
      <td>0.816322</td>
      <td>0.269493</td>
      <td>0.138957</td>
      <td>-0.782914</td>
    </tr>
    <tr>
      <th>worst area</th>
      <td>0.941082</td>
      <td>0.343546</td>
      <td>0.941550</td>
      <td>0.959213</td>
      <td>0.206718</td>
      <td>0.509604</td>
      <td>0.675987</td>
      <td>0.809630</td>
      <td>0.177193</td>
      <td>-0.231854</td>
      <td>...</td>
      <td>0.345842</td>
      <td>0.977578</td>
      <td>1.000000</td>
      <td>0.209145</td>
      <td>0.438296</td>
      <td>0.543331</td>
      <td>0.747419</td>
      <td>0.209146</td>
      <td>0.079647</td>
      <td>-0.733825</td>
    </tr>
    <tr>
      <th>worst smoothness</th>
      <td>0.119616</td>
      <td>0.077503</td>
      <td>0.150549</td>
      <td>0.123523</td>
      <td>0.805324</td>
      <td>0.565541</td>
      <td>0.448822</td>
      <td>0.452753</td>
      <td>0.426675</td>
      <td>0.504942</td>
      <td>...</td>
      <td>0.225429</td>
      <td>0.236775</td>
      <td>0.209145</td>
      <td>1.000000</td>
      <td>0.568187</td>
      <td>0.518523</td>
      <td>0.547691</td>
      <td>0.493838</td>
      <td>0.617624</td>
      <td>-0.421465</td>
    </tr>
    <tr>
      <th>worst compactness</th>
      <td>0.413463</td>
      <td>0.277830</td>
      <td>0.455774</td>
      <td>0.390410</td>
      <td>0.472468</td>
      <td>0.865809</td>
      <td>0.754968</td>
      <td>0.667454</td>
      <td>0.473200</td>
      <td>0.458798</td>
      <td>...</td>
      <td>0.360832</td>
      <td>0.529408</td>
      <td>0.438296</td>
      <td>0.568187</td>
      <td>1.000000</td>
      <td>0.892261</td>
      <td>0.801080</td>
      <td>0.614441</td>
      <td>0.810455</td>
      <td>-0.590998</td>
    </tr>
    <tr>
      <th>worst concavity</th>
      <td>0.526911</td>
      <td>0.301025</td>
      <td>0.563879</td>
      <td>0.512606</td>
      <td>0.434926</td>
      <td>0.816275</td>
      <td>0.884103</td>
      <td>0.752399</td>
      <td>0.433721</td>
      <td>0.346234</td>
      <td>...</td>
      <td>0.368366</td>
      <td>0.618344</td>
      <td>0.543331</td>
      <td>0.518523</td>
      <td>0.892261</td>
      <td>1.000000</td>
      <td>0.855434</td>
      <td>0.532520</td>
      <td>0.686511</td>
      <td>-0.659610</td>
    </tr>
    <tr>
      <th>worst concave points</th>
      <td>0.744214</td>
      <td>0.295316</td>
      <td>0.771241</td>
      <td>0.722017</td>
      <td>0.503053</td>
      <td>0.815573</td>
      <td>0.861323</td>
      <td>0.910155</td>
      <td>0.430297</td>
      <td>0.175325</td>
      <td>...</td>
      <td>0.359755</td>
      <td>0.816322</td>
      <td>0.747419</td>
      <td>0.547691</td>
      <td>0.801080</td>
      <td>0.855434</td>
      <td>1.000000</td>
      <td>0.502528</td>
      <td>0.511114</td>
      <td>-0.793566</td>
    </tr>
    <tr>
      <th>worst symmetry</th>
      <td>0.163953</td>
      <td>0.105008</td>
      <td>0.189115</td>
      <td>0.143570</td>
      <td>0.394309</td>
      <td>0.510223</td>
      <td>0.409464</td>
      <td>0.375744</td>
      <td>0.699826</td>
      <td>0.334019</td>
      <td>...</td>
      <td>0.233027</td>
      <td>0.269493</td>
      <td>0.209146</td>
      <td>0.493838</td>
      <td>0.614441</td>
      <td>0.532520</td>
      <td>0.502528</td>
      <td>1.000000</td>
      <td>0.537848</td>
      <td>-0.416294</td>
    </tr>
    <tr>
      <th>worst fractal dimension</th>
      <td>0.007066</td>
      <td>0.119205</td>
      <td>0.051019</td>
      <td>0.003738</td>
      <td>0.499316</td>
      <td>0.687382</td>
      <td>0.514930</td>
      <td>0.368661</td>
      <td>0.438413</td>
      <td>0.767297</td>
      <td>...</td>
      <td>0.219122</td>
      <td>0.138957</td>
      <td>0.079647</td>
      <td>0.617624</td>
      <td>0.810455</td>
      <td>0.686511</td>
      <td>0.511114</td>
      <td>0.537848</td>
      <td>1.000000</td>
      <td>-0.323872</td>
    </tr>
    <tr>
      <th>target</th>
      <td>-0.730029</td>
      <td>-0.415185</td>
      <td>-0.742636</td>
      <td>-0.708984</td>
      <td>-0.358560</td>
      <td>-0.596534</td>
      <td>-0.696360</td>
      <td>-0.776614</td>
      <td>-0.330499</td>
      <td>0.012838</td>
      <td>...</td>
      <td>-0.456903</td>
      <td>-0.782914</td>
      <td>-0.733825</td>
      <td>-0.421465</td>
      <td>-0.590998</td>
      <td>-0.659610</td>
      <td>-0.793566</td>
      <td>-0.416294</td>
      <td>-0.323872</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>31 rows × 31 columns</p>
</div>




```python
# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(20,20))
sns.heatmap(cancer_df.corr(), annot = True, cmap ='coolwarm', linewidths=2) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x27274bb2748>




![png](output_32_1.png)


# Correlation Barplot


```python
# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis = 1)
print("The shape of 'cancer_df2' is : ", cancer_df2.shape)
```

    The shape of 'cancer_df2' is :  (569, 30)
    


```python
cancer_df2.corrwith(cancer_df.target).index
```




    Index(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness', 'mean compactness', 'mean concavity',
           'mean concave points', 'mean symmetry', 'mean fractal dimension',
           'radius error', 'texture error', 'perimeter error', 'area error',
           'smoothness error', 'compactness error', 'concavity error',
           'concave points error', 'symmetry error', 'fractal dimension error',
           'worst radius', 'worst texture', 'worst perimeter', 'worst area',
           'worst smoothness', 'worst compactness', 'worst concavity',
           'worst concave points', 'worst symmetry', 'worst fractal dimension'],
          dtype='object')



# Split DatFrame in Train and Test


```python
# input variable
X = cancer_df.drop(['target'], axis = 1) 
X.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.80</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.60</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.90</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.80</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.00</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.50</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.42</td>
      <td>20.38</td>
      <td>77.58</td>
      <td>386.1</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>0.2597</td>
      <td>0.09744</td>
      <td>...</td>
      <td>14.91</td>
      <td>26.50</td>
      <td>98.87</td>
      <td>567.7</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20.29</td>
      <td>14.34</td>
      <td>135.10</td>
      <td>1297.0</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>0.1809</td>
      <td>0.05883</td>
      <td>...</td>
      <td>22.54</td>
      <td>16.67</td>
      <td>152.20</td>
      <td>1575.0</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
    <tr>
      <th>5</th>
      <td>12.45</td>
      <td>15.70</td>
      <td>82.57</td>
      <td>477.1</td>
      <td>0.12780</td>
      <td>0.17000</td>
      <td>0.1578</td>
      <td>0.08089</td>
      <td>0.2087</td>
      <td>0.07613</td>
      <td>...</td>
      <td>15.47</td>
      <td>23.75</td>
      <td>103.40</td>
      <td>741.6</td>
      <td>0.1791</td>
      <td>0.5249</td>
      <td>0.5355</td>
      <td>0.1741</td>
      <td>0.3985</td>
      <td>0.12440</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 30 columns</p>
</div>




```python
# output variable
y = cancer_df['target'] 
y.head(6)
```




    0    0.0
    1    0.0
    2    0.0
    3    0.0
    4    0.0
    5    0.0
    Name: target, dtype: float64




```python
# split dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 5)
```

# Feature scaling 


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```

# Machine Learning Model Building


```python
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
```

## Suppor vector Classifier


```python
# Support vector classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(y_test, y_pred_scv)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    




    0.5789473684210527




```python
# Train with Standard scaled Data
svc_classifier2 = SVC()
svc_classifier2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_svc_sc)
```




    0.9649122807017544



# Logistic Regression


```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 51, penalty = 'l1')
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)
accuracy_score(y_test, y_pred_lr)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\svm\base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)
    




    0.9736842105263158




```python
# Train with Standard scaled Data
lr_classifier2 = LogisticRegression(random_state = 51, penalty = 'l1')
lr_classifier2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_lr_sc)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    




    0.5526315789473685



# K – Nearest Neighbor Classifier


```python
# K – Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(y_test, y_pred_knn)
```




    0.9385964912280702




```python
# Train with Standard scaled Data
knn_classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_knn_sc)
```




    0.5789473684210527



# Naive Bayes Classifier


```python
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_nb)
```




    0.9473684210526315




```python
# Train with Standard scaled Data
nb_classifier2 = GaussianNB()
nb_classifier2.fit(X_train_sc, y_train)
y_pred_nb_sc = nb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_nb_sc)
```




    0.9385964912280702



# Decision Tree Classifier


```python
# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(y_test, y_pred_dt)
```




    0.9473684210526315




```python
# Train with Standard scaled Data
dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)
```




    0.7543859649122807



 # Random Forest Classifier


```python
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
accuracy_score(y_test, y_pred_rf)
```




    0.9736842105263158




```python
# Train with Standard scaled Data
rf_classifier2 = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_classifier.predict(X_test_sc)
accuracy_score(y_test, y_pred_rf_sc)
```




    0.7543859649122807



# AdaBoost Classifier


```python
# Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier.fit(X_train, y_train)
y_pred_adb = adb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_adb)
```




    0.9473684210526315




```python
# Train with Standard scaled Data
adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier2.fit(X_train_sc, y_train)
y_pred_adb_sc = adb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_adb_sc)
```




    0.9473684210526315



# XGBoost Classifier


```python
# XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(y_test, y_pred_xgb)
```




    0.9824561403508771




```python
# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(y_test, y_pred_xgb_sc)
```




    0.9824561403508771



 # XGBoost Parameter Tuning Randomized Search 


```python
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] 
}
```


```python
# Randomized Search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_classifier, param_distributions=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
random_search.fit(X_train, y_train)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    

    Fitting 3 folds for each of 10 candidates, totalling 30 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed:  3.9min finished
    




    RandomizedSearchCV(cv='warn', error_score='raise-deprecating',
              estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
           subsample=1, verbosity=1),
              fit_params=None, iid='warn', n_iter=10, n_jobs=-1,
              param_distributions={'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], 'max_depth': [3, 4, 5, 6, 8, 10, 12, 15], 'min_child_weight': [1, 3, 5, 7], 'gamma': [0.0, 0.1, 0.2, 0.3, 0.4], 'colsample_bytree': [0.3, 0.4, 0.5, 0.7]},
              pre_dispatch='2*n_jobs', random_state=None, refit=True,
              return_train_score='warn', scoring='roc_auc', verbose=3)




```python
random_search.best_params_
```




    {'min_child_weight': 3,
     'max_depth': 5,
     'learning_rate': 0.25,
     'gamma': 0.0,
     'colsample_bytree': 0.3}




```python
random_search.best_estimator_
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=0.3, gamma=0.0,
           learning_rate=0.25, max_delta_step=0, max_depth=5,
           min_child_weight=3, missing=None, n_estimators=100, n_jobs=1,
           nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=None, subsample=1, verbosity=1)




```python
# training XGBoost classifier with best parameters
xgb_classifier_pt = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.4, gamma=0.2,
       learning_rate=0.1, max_delta_step=0, max_depth=15,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)

xgb_classifier_pt.fit(X_train, y_train)
y_pred_xgb_pt = xgb_classifier_pt.predict(X_test)
```


```python
accuracy_score(y_test, y_pred_xgb_pt)
```




    0.9824561403508771



# Grid Search


```python
from sklearn.model_selection import GridSearchCV 
grid_search = GridSearchCV(xgb_classifier, param_grid=params, scoring= 'roc_auc', n_jobs= -1, verbose= 3)
grid_search.fit(X_train, y_train)
```

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\model_selection\_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    

    Fitting 3 folds for each of 3840 candidates, totalling 11520 fits
    

    [Parallel(n_jobs=-1)]: Done  24 tasks      | elapsed:    0.9s
    [Parallel(n_jobs=-1)]: Done 120 tasks      | elapsed:    4.6s
    [Parallel(n_jobs=-1)]: Done 280 tasks      | elapsed:    9.9s
    [Parallel(n_jobs=-1)]: Done 664 tasks      | elapsed:   22.4s
    [Parallel(n_jobs=-1)]: Done 1240 tasks      | elapsed:   41.6s
    [Parallel(n_jobs=-1)]: Done 1944 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 2776 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 3730 tasks      | elapsed:  2.4min
    [Parallel(n_jobs=-1)]: Done 4641 tasks      | elapsed:  3.0min
    [Parallel(n_jobs=-1)]: Done 5857 tasks      | elapsed:  3.9min
    [Parallel(n_jobs=-1)]: Done 7201 tasks      | elapsed:  4.7min
    [Parallel(n_jobs=-1)]: Done 8673 tasks      | elapsed:  5.4min
    [Parallel(n_jobs=-1)]: Done 9770 tasks      | elapsed:  6.2min
    [Parallel(n_jobs=-1)]: Done 10917 tasks      | elapsed:  6.9min
    [Parallel(n_jobs=-1)]: Done 11513 out of 11520 | elapsed:  7.3min remaining:    0.2s
    [Parallel(n_jobs=-1)]: Done 11520 out of 11520 | elapsed:  7.3min finished
    




    GridSearchCV(cv='warn', error_score='raise-deprecating',
           estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
           subsample=1, verbosity=1),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], 'max_depth': [3, 4, 5, 6, 8, 10, 12, 15], 'min_child_weight': [1, 3, 5, 7], 'gamma': [0.0, 0.1, 0.2, 0.3, 0.4], 'colsample_bytree': [0.3, 0.4, 0.5, 0.7]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='roc_auc', verbose=3)




```python
grid_search.best_estimator_
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=0.3, gamma=0.0,
           learning_rate=0.3, max_delta_step=0, max_depth=3,
           min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
           nthread=None, objective='binary:logistic', random_state=0,
           reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
           silent=None, subsample=1, verbosity=1)




```python
xgb_classifier_pt_gs = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.3, gamma=0.0,
       learning_rate=0.3, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
xgb_classifier_pt_gs.fit(X_train, y_train)
y_pred_xgb_pt_gs = xgb_classifier_pt_gs.predict(X_test)
accuracy_score(y_test, y_pred_xgb_pt_gs)
```




    0.9824561403508771



# Confusion Matrix


```python
cm = confusion_matrix(y_test, y_pred_xgb_pt)
plt.title('Heatmap of Confusion Matrix', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()
```


![png](output_80_0.png)

The model is giving 0 type II error and it is best
# Classification Report Of model


```python
print(classification_report(y_test, y_pred_xgb_pt))
```

                  precision    recall  f1-score   support
    
             0.0       1.00      0.96      0.98        48
             1.0       0.97      1.00      0.99        66
    
       micro avg       0.98      0.98      0.98       114
       macro avg       0.99      0.98      0.98       114
    weighted avg       0.98      0.98      0.98       114
    
    

# Cross-validation of the ML model


```python
# Cross validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_classifier_pt, X = X_train_sc,y = y_train, cv = 10)
print("Cross validation accuracy of XGBoost model = ", cross_validation)
print("\nCross validation mean accuracy of XGBoost model = ", cross_validation.mean())
```

    Cross validation accuracy of XGBoost model =  [0.9787234  0.97826087 0.97826087 0.97826087 0.93333333 0.91111111
     1.         1.         0.97777778 0.88888889]
    
    Cross validation mean accuracy of XGBoost model =  0.9624617124062083
    

# Save XGBoost Classifier model using Pickel


```python
## Pickle
import pickle

# save model
pickle.dump(xgb_classifier_pt, open('breast_cancer_detector.pickle', 'wb'))

# load model
breast_cancer_detector_model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

# predict the output
y_pred = breast_cancer_detector_model.predict(X_test)

# confusion matrix
print('Confusion matrix of XGBoost model: \n',confusion_matrix(y_test, y_pred),'\n')

# show the accuracy
print('Accuracy of XGBoost model = ',accuracy_score(y_test, y_pred))
```

    Confusion matrix of XGBoost model: 
     [[46  2]
     [ 0 66]] 
    
    Accuracy of XGBoost model =  0.9824561403508771
    
