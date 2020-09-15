```python
import tensorflow as tf
import pandas as pd
import numpy as np

```


```python
dataframe = pd.read_csv('cancerdataset.csv')
```


```python
print(dataframe)
```

               id  diagnosis  radius_0ean  texture_0ean  peri0eter_0ean  \
    0      842302          0        17.99         10.38          122.80   
    1      842517          0        20.57         17.77          132.90   
    2    84300903          0        19.69         21.25          130.00   
    3    84348301          0        11.42         20.38           77.58   
    4    84358402          0        20.29         14.34          135.10   
    ..        ...        ...          ...           ...             ...   
    564    926424          0        21.56         22.39          142.00   
    565    926682          0        20.13         28.25          131.20   
    566    926954          0        16.60         28.08          108.30   
    567    927241          0        20.60         29.33          140.10   
    568     92751          1         7.76         24.54           47.92   
    
         area_0ean  s0oothness_0ean  co0pactness_0ean  concavity_0ean  \
    0       1001.0          0.11840           0.27760         0.30010   
    1       1326.0          0.08474           0.07864         0.08690   
    2       1203.0          0.10960           0.15990         0.19740   
    3        386.1          0.14250           0.28390         0.24140   
    4       1297.0          0.10030           0.13280         0.19800   
    ..         ...              ...               ...             ...   
    564     1479.0          0.11100           0.11590         0.24390   
    565     1261.0          0.09780           0.10340         0.14400   
    566      858.1          0.08455           0.10230         0.09251   
    567     1265.0          0.11780           0.27700         0.35140   
    568      181.0          0.05263           0.04362         0.00000   
    
         concave points_0ean  ...  radius_worst  texture_worst  peri0eter_worst  \
    0                0.14710  ...        25.380          17.33           184.60   
    1                0.07017  ...        24.990          23.41           158.80   
    2                0.12790  ...        23.570          25.53           152.50   
    3                0.10520  ...        14.910          26.50            98.87   
    4                0.10430  ...        22.540          16.67           152.20   
    ..                   ...  ...           ...            ...              ...   
    564              0.13890  ...        25.450          26.40           166.10   
    565              0.09791  ...        23.690          38.25           155.00   
    566              0.05302  ...        18.980          34.12           126.70   
    567              0.15200  ...        25.740          39.42           184.60   
    568              0.00000  ...         9.456          30.37            59.16   
    
         area_worst  s0oothness_worst  co0pactness_worst  concavity_worst  \
    0        2019.0           0.16220            0.66560           0.7119   
    1        1956.0           0.12380            0.18660           0.2416   
    2        1709.0           0.14440            0.42450           0.4504   
    3         567.7           0.20980            0.86630           0.6869   
    4        1575.0           0.13740            0.20500           0.4000   
    ..          ...               ...                ...              ...   
    564      2027.0           0.14100            0.21130           0.4107   
    565      1731.0           0.11660            0.19220           0.3215   
    566      1124.0           0.11390            0.30940           0.3403   
    567      1821.0           0.16500            0.86810           0.9387   
    568       268.6           0.08996            0.06444           0.0000   
    
         concave points_worst  sy00etry_worst  fractal_di0ension_worst  
    0                  0.2654          0.4601                  0.11890  
    1                  0.1860          0.2750                  0.08902  
    2                  0.2430          0.3613                  0.08758  
    3                  0.2575          0.6638                  0.17300  
    4                  0.1625          0.2364                  0.07678  
    ..                    ...             ...                      ...  
    564                0.2216          0.2060                  0.07115  
    565                0.1628          0.2572                  0.06637  
    566                0.1418          0.2218                  0.07820  
    567                0.2650          0.4087                  0.12400  
    568                0.0000          0.2871                  0.07039  
    
    [569 rows x 32 columns]
    


```python
dataframe.describe()
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
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_0ean</th>
      <th>texture_0ean</th>
      <th>peri0eter_0ean</th>
      <th>area_0ean</th>
      <th>s0oothness_0ean</th>
      <th>co0pactness_0ean</th>
      <th>concavity_0ean</th>
      <th>concave points_0ean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>peri0eter_worst</th>
      <th>area_worst</th>
      <th>s0oothness_worst</th>
      <th>co0pactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>sy00etry_worst</th>
      <th>fractal_di0ension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.690000e+02</td>
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
      <td>3.037183e+07</td>
      <td>0.627417</td>
      <td>14.127292</td>
      <td>19.289649</td>
      <td>91.969033</td>
      <td>654.889104</td>
      <td>0.096360</td>
      <td>0.104341</td>
      <td>0.088799</td>
      <td>0.048919</td>
      <td>...</td>
      <td>16.269190</td>
      <td>25.677223</td>
      <td>107.261213</td>
      <td>880.583128</td>
      <td>0.132369</td>
      <td>0.254265</td>
      <td>0.272188</td>
      <td>0.114606</td>
      <td>0.290076</td>
      <td>0.083946</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.250206e+08</td>
      <td>0.483918</td>
      <td>3.524049</td>
      <td>4.301036</td>
      <td>24.298981</td>
      <td>351.914129</td>
      <td>0.014064</td>
      <td>0.052813</td>
      <td>0.079720</td>
      <td>0.038803</td>
      <td>...</td>
      <td>4.833242</td>
      <td>6.146258</td>
      <td>33.602542</td>
      <td>569.356993</td>
      <td>0.022832</td>
      <td>0.157336</td>
      <td>0.208624</td>
      <td>0.065732</td>
      <td>0.061867</td>
      <td>0.018061</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.670000e+03</td>
      <td>0.000000</td>
      <td>6.981000</td>
      <td>9.710000</td>
      <td>43.790000</td>
      <td>143.500000</td>
      <td>0.052630</td>
      <td>0.019380</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>7.930000</td>
      <td>12.020000</td>
      <td>50.410000</td>
      <td>185.200000</td>
      <td>0.071170</td>
      <td>0.027290</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.156500</td>
      <td>0.055040</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.692180e+05</td>
      <td>0.000000</td>
      <td>11.700000</td>
      <td>16.170000</td>
      <td>75.170000</td>
      <td>420.300000</td>
      <td>0.086370</td>
      <td>0.064920</td>
      <td>0.029560</td>
      <td>0.020310</td>
      <td>...</td>
      <td>13.010000</td>
      <td>21.080000</td>
      <td>84.110000</td>
      <td>515.300000</td>
      <td>0.116600</td>
      <td>0.147200</td>
      <td>0.114500</td>
      <td>0.064930</td>
      <td>0.250400</td>
      <td>0.071460</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9.060240e+05</td>
      <td>1.000000</td>
      <td>13.370000</td>
      <td>18.840000</td>
      <td>86.240000</td>
      <td>551.100000</td>
      <td>0.095870</td>
      <td>0.092630</td>
      <td>0.061540</td>
      <td>0.033500</td>
      <td>...</td>
      <td>14.970000</td>
      <td>25.410000</td>
      <td>97.660000</td>
      <td>686.500000</td>
      <td>0.131300</td>
      <td>0.211900</td>
      <td>0.226700</td>
      <td>0.099930</td>
      <td>0.282200</td>
      <td>0.080040</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>8.813129e+06</td>
      <td>1.000000</td>
      <td>15.780000</td>
      <td>21.800000</td>
      <td>104.100000</td>
      <td>782.700000</td>
      <td>0.105300</td>
      <td>0.130400</td>
      <td>0.130700</td>
      <td>0.074000</td>
      <td>...</td>
      <td>18.790000</td>
      <td>29.720000</td>
      <td>125.400000</td>
      <td>1084.000000</td>
      <td>0.146000</td>
      <td>0.339100</td>
      <td>0.382900</td>
      <td>0.161400</td>
      <td>0.317900</td>
      <td>0.092080</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.113205e+08</td>
      <td>1.000000</td>
      <td>28.110000</td>
      <td>39.280000</td>
      <td>188.500000</td>
      <td>2501.000000</td>
      <td>0.163400</td>
      <td>0.345400</td>
      <td>0.426800</td>
      <td>0.201200</td>
      <td>...</td>
      <td>36.040000</td>
      <td>49.540000</td>
      <td>251.200000</td>
      <td>4254.000000</td>
      <td>0.222600</td>
      <td>1.058000</td>
      <td>1.252000</td>
      <td>0.291000</td>
      <td>0.663800</td>
      <td>0.207500</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 32 columns</p>
</div>




```python
dataframe = dataframe.astype('float32')
```


```python
dataframe.head()
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
      <th>id</th>
      <th>diagnosis</th>
      <th>radius_0ean</th>
      <th>texture_0ean</th>
      <th>peri0eter_0ean</th>
      <th>area_0ean</th>
      <th>s0oothness_0ean</th>
      <th>co0pactness_0ean</th>
      <th>concavity_0ean</th>
      <th>concave points_0ean</th>
      <th>...</th>
      <th>radius_worst</th>
      <th>texture_worst</th>
      <th>peri0eter_worst</th>
      <th>area_worst</th>
      <th>s0oothness_worst</th>
      <th>co0pactness_worst</th>
      <th>concavity_worst</th>
      <th>concave points_worst</th>
      <th>sy00etry_worst</th>
      <th>fractal_di0ension_worst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842302.0</td>
      <td>0.0</td>
      <td>17.990000</td>
      <td>10.380000</td>
      <td>122.800003</td>
      <td>1001.000000</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>...</td>
      <td>25.379999</td>
      <td>17.330000</td>
      <td>184.600006</td>
      <td>2019.000000</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>842517.0</td>
      <td>0.0</td>
      <td>20.570000</td>
      <td>17.770000</td>
      <td>132.899994</td>
      <td>1326.000000</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>...</td>
      <td>24.990000</td>
      <td>23.410000</td>
      <td>158.800003</td>
      <td>1956.000000</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84300904.0</td>
      <td>0.0</td>
      <td>19.690001</td>
      <td>21.250000</td>
      <td>130.000000</td>
      <td>1203.000000</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>...</td>
      <td>23.570000</td>
      <td>25.530001</td>
      <td>152.500000</td>
      <td>1709.000000</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84348304.0</td>
      <td>0.0</td>
      <td>11.420000</td>
      <td>20.379999</td>
      <td>77.580002</td>
      <td>386.100006</td>
      <td>0.14250</td>
      <td>0.28390</td>
      <td>0.2414</td>
      <td>0.10520</td>
      <td>...</td>
      <td>14.910000</td>
      <td>26.500000</td>
      <td>98.870003</td>
      <td>567.700012</td>
      <td>0.2098</td>
      <td>0.8663</td>
      <td>0.6869</td>
      <td>0.2575</td>
      <td>0.6638</td>
      <td>0.17300</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84358400.0</td>
      <td>0.0</td>
      <td>20.290001</td>
      <td>14.340000</td>
      <td>135.100006</td>
      <td>1297.000000</td>
      <td>0.10030</td>
      <td>0.13280</td>
      <td>0.1980</td>
      <td>0.10430</td>
      <td>...</td>
      <td>22.540001</td>
      <td>16.670000</td>
      <td>152.199997</td>
      <td>1575.000000</td>
      <td>0.1374</td>
      <td>0.2050</td>
      <td>0.4000</td>
      <td>0.1625</td>
      <td>0.2364</td>
      <td>0.07678</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
dataframe=dataframe.to_numpy()
```


```python
dataframe.shape
```




    (569, 32)




```python
training = dataframe[0:500]
testing = dataframe[500:]
```


```python
print(training.shape)
print(testing.shape)
```

    (500, 32)
    (69, 32)
    


```python
training_features = training[:, 1:]
training_labels = training[:, 0]
testing_features = testing[:, 1:]
testing_labels = testing[:, 0]
```


```python
print(training_features.shape)
print(training_labels.shape)
```

    (500, 31)
    (500,)
    


```python
training_features[0].shape
```




    (31,)




```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Dropout
```


```python
model = Sequential()
model.add(Input(shape=(31,)))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(31,))
model.add(Dropout(0.2))
model.add(Dense(1))
```


```python
model.compile(optimizer='sgd', loss ='binary_crossentropy', metrics=['accuracy'])
```


```python
model.fit(training_features,training_labels,epochs=5000,validation_data=(testing_features,testing_labels))
```

    Train on 500 samples, validate on 69 samples
    Epoch 1/5000
    500/500 [==============================] - 0s 381us/sample - loss: -22894881.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2/5000
    500/500 [==============================] - 0s 90us/sample - loss: 44332924.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3/5000
    500/500 [==============================] - 0s 85us/sample - loss: -94487358.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4/5000
    500/500 [==============================] - 0s 88us/sample - loss: -91694975.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 5/5000
    500/500 [==============================] - 0s 76us/sample - loss: -54389006.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 6/5000
    500/500 [==============================] - 0s 90us/sample - loss: -38579118.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 7/5000
    500/500 [==============================] - 0s 90us/sample - loss: -174655241.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 8/5000
    500/500 [==============================] - 0s 78us/sample - loss: 145670836.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 9/5000
    500/500 [==============================] - 0s 90us/sample - loss: 156239309.7915 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 10/5000
    500/500 [==============================] - 0s 84us/sample - loss: 65336679.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 11/5000
    500/500 [==============================] - 0s 88us/sample - loss: 5788803.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 12/5000
    500/500 [==============================] - 0s 90us/sample - loss: 187620094.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 13/5000
    500/500 [==============================] - 0s 76us/sample - loss: 39237585.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 14/5000
    500/500 [==============================] - 0s 86us/sample - loss: -55763097.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 15/5000
    500/500 [==============================] - 0s 90us/sample - loss: 49317954.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 16/5000
    500/500 [==============================] - 0s 84us/sample - loss: 121851967.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 17/5000
    500/500 [==============================] - 0s 88us/sample - loss: -32739337.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 18/5000
    500/500 [==============================] - 0s 88us/sample - loss: 151982728.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 19/5000
    500/500 [==============================] - 0s 80us/sample - loss: -29014353.2570 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 20/5000
    500/500 [==============================] - 0s 84us/sample - loss: -1891560.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 21/5000
    500/500 [==============================] - 0s 78us/sample - loss: 128281274.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 22/5000
    500/500 [==============================] - 0s 86us/sample - loss: 200521895.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 23/5000
    500/500 [==============================] - 0s 90us/sample - loss: 105850712.0620 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 24/5000
    500/500 [==============================] - 0s 80us/sample - loss: 73009562.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 25/5000
    500/500 [==============================] - 0s 88us/sample - loss: -32603730.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 26/5000
    500/500 [==============================] - 0s 100us/sample - loss: 151580669.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 27/5000
    500/500 [==============================] - 0s 96us/sample - loss: -8573434.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 28/5000
    500/500 [==============================] - 0s 94us/sample - loss: 21921269.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 29/5000
    500/500 [==============================] - 0s 82us/sample - loss: -22941769.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 30/5000
    500/500 [==============================] - 0s 76us/sample - loss: -46321298.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 31/5000
    500/500 [==============================] - 0s 76us/sample - loss: 21443312.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 32/5000
    500/500 [==============================] - 0s 88us/sample - loss: -42287627.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 33/5000
    500/500 [==============================] - 0s 88us/sample - loss: -54031047.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 34/5000
    500/500 [==============================] - 0s 78us/sample - loss: 122827438.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 35/5000
    500/500 [==============================] - 0s 86us/sample - loss: 14440203.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 36/5000
    500/500 [==============================] - 0s 80us/sample - loss: 74127067.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 37/5000
    500/500 [==============================] - 0s 86us/sample - loss: 128673760.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 38/5000
    500/500 [==============================] - 0s 82us/sample - loss: 163743026.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 39/5000
    500/500 [==============================] - 0s 86us/sample - loss: -88903235.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 40/5000
    500/500 [==============================] - 0s 74us/sample - loss: 50040065.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 41/5000
    500/500 [==============================] - 0s 82us/sample - loss: -61803739.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 42/5000
    500/500 [==============================] - 0s 90us/sample - loss: 24340752.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 43/5000
    500/500 [==============================] - 0s 96us/sample - loss: 34090958.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 44/5000
    500/500 [==============================] - 0s 84us/sample - loss: 33954680.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 45/5000
    500/500 [==============================] - 0s 88us/sample - loss: 145522981.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 46/5000
    500/500 [==============================] - 0s 94us/sample - loss: -241898659.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 47/5000
    500/500 [==============================] - 0s 90us/sample - loss: -36440749.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 48/5000
    500/500 [==============================] - 0s 100us/sample - loss: 347209.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 49/5000
    500/500 [==============================] - 0s 84us/sample - loss: -95706602.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 50/5000
    500/500 [==============================] - 0s 82us/sample - loss: -90967330.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 51/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32710756.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 52/5000
    500/500 [==============================] - 0s 80us/sample - loss: 78357736.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 53/5000
    500/500 [==============================] - 0s 80us/sample - loss: 260481504.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 54/5000
    500/500 [==============================] - 0s 90us/sample - loss: -171409220.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 55/5000
    500/500 [==============================] - 0s 77us/sample - loss: -26546488.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 56/5000
    500/500 [==============================] - 0s 78us/sample - loss: 157583878.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 57/5000
    500/500 [==============================] - 0s 75us/sample - loss: 23503025.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 58/5000
    500/500 [==============================] - 0s 78us/sample - loss: 70419578.5515 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 59/5000
    500/500 [==============================] - 0s 78us/sample - loss: -81446378.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 60/5000
    500/500 [==============================] - 0s 78us/sample - loss: 35137918.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 61/5000
    500/500 [==============================] - 0s 78us/sample - loss: -61910323.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 62/5000
    500/500 [==============================] - 0s 80us/sample - loss: -80865124.2280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 63/5000
    500/500 [==============================] - 0s 78us/sample - loss: -45136322.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 64/5000
    500/500 [==============================] - 0s 80us/sample - loss: -9121507.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 65/5000
    500/500 [==============================] - 0s 76us/sample - loss: 120419765.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 66/5000
    500/500 [==============================] - 0s 82us/sample - loss: 12530147.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 67/5000
    500/500 [==============================] - 0s 78us/sample - loss: 83544732.6040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 68/5000
    500/500 [==============================] - 0s 82us/sample - loss: 20089657.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 69/5000
    500/500 [==============================] - 0s 78us/sample - loss: -40402666.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 70/5000
    500/500 [==============================] - 0s 80us/sample - loss: -61858139.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 71/5000
    500/500 [==============================] - 0s 80us/sample - loss: -174740194.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 72/5000
    500/500 [==============================] - 0s 84us/sample - loss: 167659411.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 73/5000
    500/500 [==============================] - 0s 78us/sample - loss: 102887422.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 74/5000
    500/500 [==============================] - 0s 84us/sample - loss: -92747873.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 75/5000
    500/500 [==============================] - 0s 78us/sample - loss: -134508103.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 76/5000
    500/500 [==============================] - 0s 80us/sample - loss: -137812651.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 77/5000
    500/500 [==============================] - 0s 80us/sample - loss: 21862057.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 78/5000
    500/500 [==============================] - 0s 80us/sample - loss: 94789581.5670 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 79/5000
    500/500 [==============================] - 0s 78us/sample - loss: -138700614.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 80/5000
    500/500 [==============================] - 0s 82us/sample - loss: -72296814.6220 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 81/5000
    500/500 [==============================] - 0s 80us/sample - loss: 76991722.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 82/5000
    500/500 [==============================] - 0s 78us/sample - loss: 131392678.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 83/5000
    500/500 [==============================] - 0s 82us/sample - loss: 115876768.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 84/5000
    500/500 [==============================] - 0s 82us/sample - loss: 153919975.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 85/5000
    500/500 [==============================] - 0s 78us/sample - loss: -64927396.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 86/5000
    500/500 [==============================] - 0s 78us/sample - loss: -184004562.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 87/5000
    500/500 [==============================] - 0s 82us/sample - loss: -101417911.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 88/5000
    500/500 [==============================] - 0s 80us/sample - loss: 96051938.7240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 89/5000
    500/500 [==============================] - 0s 84us/sample - loss: 47655380.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 90/5000
    500/500 [==============================] - 0s 80us/sample - loss: -33170596.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 91/5000
    500/500 [==============================] - 0s 82us/sample - loss: -152835504.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 92/5000
    500/500 [==============================] - 0s 82us/sample - loss: 70388618.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 93/5000
    500/500 [==============================] - 0s 84us/sample - loss: -67297453.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 94/5000
    500/500 [==============================] - 0s 80us/sample - loss: -55440179.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 95/5000
    500/500 [==============================] - 0s 80us/sample - loss: -23814840.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 96/5000
    500/500 [==============================] - 0s 84us/sample - loss: 133115284.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 97/5000
    500/500 [==============================] - 0s 78us/sample - loss: -52643505.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 98/5000
    500/500 [==============================] - 0s 78us/sample - loss: -136487127.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 99/5000
    500/500 [==============================] - 0s 78us/sample - loss: -122329867.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 100/5000
    500/500 [==============================] - 0s 76us/sample - loss: -108448773.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 101/5000
    500/500 [==============================] - 0s 80us/sample - loss: -91758713.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 102/5000
    500/500 [==============================] - 0s 80us/sample - loss: 36876467.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 103/5000
    500/500 [==============================] - 0s 75us/sample - loss: 16276579.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 104/5000
    500/500 [==============================] - 0s 78us/sample - loss: -37831002.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 105/5000
    500/500 [==============================] - 0s 78us/sample - loss: 13779517.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 106/5000
    500/500 [==============================] - 0s 80us/sample - loss: 74695908.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 107/5000
    500/500 [==============================] - 0s 78us/sample - loss: 101515797.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 108/5000
    500/500 [==============================] - 0s 78us/sample - loss: 95858532.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 109/5000
    500/500 [==============================] - 0s 76us/sample - loss: -112202452.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 110/5000
    500/500 [==============================] - 0s 80us/sample - loss: 76430431.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 111/5000
    500/500 [==============================] - 0s 80us/sample - loss: 20784818.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 112/5000
    500/500 [==============================] - 0s 73us/sample - loss: -27310638.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 113/5000
    500/500 [==============================] - 0s 78us/sample - loss: -170378215.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 114/5000
    500/500 [==============================] - 0s 82us/sample - loss: 139335543.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 115/5000
    500/500 [==============================] - 0s 82us/sample - loss: 70952443.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 116/5000
    500/500 [==============================] - 0s 82us/sample - loss: 81521681.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 117/5000
    500/500 [==============================] - 0s 78us/sample - loss: -52826405.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 118/5000
    500/500 [==============================] - 0s 82us/sample - loss: 78441307.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 119/5000
    500/500 [==============================] - 0s 82us/sample - loss: 51966076.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 120/5000
    500/500 [==============================] - 0s 86us/sample - loss: -35630230.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 121/5000
    500/500 [==============================] - 0s 80us/sample - loss: -75419701.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 122/5000
    500/500 [==============================] - 0s 84us/sample - loss: 119928104.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 123/5000
    500/500 [==============================] - 0s 80us/sample - loss: -98422909.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 124/5000
    500/500 [==============================] - 0s 80us/sample - loss: 76135142.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 125/5000
    500/500 [==============================] - 0s 82us/sample - loss: 111987834.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 126/5000
    500/500 [==============================] - 0s 80us/sample - loss: 39821595.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 127/5000
    500/500 [==============================] - 0s 86us/sample - loss: 142609798.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 128/5000
    500/500 [==============================] - 0s 80us/sample - loss: -165994565.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 129/5000
    500/500 [==============================] - 0s 82us/sample - loss: 83655837.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 130/5000
    500/500 [==============================] - 0s 80us/sample - loss: 44876668.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 131/5000
    500/500 [==============================] - 0s 80us/sample - loss: 128328270.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 132/5000
    500/500 [==============================] - 0s 80us/sample - loss: 123214067.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 133/5000
    500/500 [==============================] - 0s 80us/sample - loss: -31568393.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 134/5000
    500/500 [==============================] - 0s 80us/sample - loss: 35168249.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 135/5000
    500/500 [==============================] - 0s 84us/sample - loss: 85290429.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 136/5000
    500/500 [==============================] - 0s 82us/sample - loss: 14103635.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 137/5000
    500/500 [==============================] - 0s 80us/sample - loss: 137068019.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 138/5000
    500/500 [==============================] - 0s 80us/sample - loss: 183606356.9960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 139/5000
    500/500 [==============================] - 0s 82us/sample - loss: 113881352.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 140/5000
    500/500 [==============================] - 0s 82us/sample - loss: -64635695.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 141/5000
    500/500 [==============================] - 0s 82us/sample - loss: -199027244.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 142/5000
    500/500 [==============================] - 0s 80us/sample - loss: 101803311.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 143/5000
    500/500 [==============================] - 0s 82us/sample - loss: -8949049.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 144/5000
    500/500 [==============================] - 0s 78us/sample - loss: 16018677.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 145/5000
    500/500 [==============================] - 0s 78us/sample - loss: -76171173.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 146/5000
    500/500 [==============================] - 0s 76us/sample - loss: 61482246.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 147/5000
    500/500 [==============================] - 0s 74us/sample - loss: -33425468.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 148/5000
    500/500 [==============================] - 0s 76us/sample - loss: -21687218.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 149/5000
    500/500 [==============================] - 0s 76us/sample - loss: -65705918.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 150/5000
    500/500 [==============================] - 0s 80us/sample - loss: 132750688.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 151/5000
    500/500 [==============================] - 0s 80us/sample - loss: 91616668.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 152/5000
    500/500 [==============================] - 0s 80us/sample - loss: -82968899.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 153/5000
    500/500 [==============================] - 0s 76us/sample - loss: -122114558.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 154/5000
    500/500 [==============================] - 0s 82us/sample - loss: -201500254.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 155/5000
    500/500 [==============================] - 0s 80us/sample - loss: -104046910.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 156/5000
    500/500 [==============================] - 0s 76us/sample - loss: -4738964.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 157/5000
    500/500 [==============================] - 0s 78us/sample - loss: 36316802.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 158/5000
    500/500 [==============================] - 0s 80us/sample - loss: -159786256.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 159/5000
    500/500 [==============================] - 0s 78us/sample - loss: 88541569.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 160/5000
    500/500 [==============================] - 0s 81us/sample - loss: 37359756.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 161/5000
    500/500 [==============================] - 0s 88us/sample - loss: -57975391.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 162/5000
    500/500 [==============================] - 0s 92us/sample - loss: 41359704.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 163/5000
    500/500 [==============================] - 0s 86us/sample - loss: 31028589.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 164/5000
    500/500 [==============================] - 0s 82us/sample - loss: 54327034.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 165/5000
    500/500 [==============================] - 0s 84us/sample - loss: -39217637.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 166/5000
    500/500 [==============================] - 0s 90us/sample - loss: 89706788.7357 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 167/5000
    500/500 [==============================] - 0s 90us/sample - loss: -138954186.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 168/5000
    500/500 [==============================] - 0s 92us/sample - loss: -58148171.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 169/5000
    500/500 [==============================] - 0s 100us/sample - loss: -148245473.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 170/5000
    500/500 [==============================] - 0s 86us/sample - loss: -2414276.9290 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 171/5000
    500/500 [==============================] - 0s 84us/sample - loss: 52703041.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 172/5000
    500/500 [==============================] - 0s 90us/sample - loss: 25582667.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 173/5000
    500/500 [==============================] - 0s 90us/sample - loss: 27139067.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 174/5000
    500/500 [==============================] - 0s 82us/sample - loss: 39322916.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 175/5000
    500/500 [==============================] - 0s 90us/sample - loss: 102454207.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 176/5000
    500/500 [==============================] - 0s 90us/sample - loss: 104329865.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 177/5000
    500/500 [==============================] - 0s 92us/sample - loss: 72592346.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 178/5000
    500/500 [==============================] - 0s 90us/sample - loss: -69418097.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 179/5000
    500/500 [==============================] - 0s 82us/sample - loss: 49243448.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 180/5000
    500/500 [==============================] - 0s 86us/sample - loss: -46384875.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 181/5000
    500/500 [==============================] - 0s 88us/sample - loss: 127913323.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 182/5000
    500/500 [==============================] - 0s 86us/sample - loss: 212642772.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 183/5000
    500/500 [==============================] - 0s 86us/sample - loss: -130005503.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 184/5000
    500/500 [==============================] - 0s 90us/sample - loss: 92922138.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 185/5000
    500/500 [==============================] - 0s 90us/sample - loss: 114695480.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 186/5000
    500/500 [==============================] - 0s 84us/sample - loss: -54113506.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 187/5000
    500/500 [==============================] - 0s 92us/sample - loss: -225055422.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 188/5000
    500/500 [==============================] - 0s 92us/sample - loss: 94157962.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 189/5000
    500/500 [==============================] - 0s 92us/sample - loss: 60749957.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 190/5000
    500/500 [==============================] - 0s 88us/sample - loss: -48963659.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 191/5000
    500/500 [==============================] - 0s 82us/sample - loss: 3020268.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 192/5000
    500/500 [==============================] - 0s 98us/sample - loss: -5931658.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 193/5000
    500/500 [==============================] - 0s 92us/sample - loss: -123413346.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 194/5000
    500/500 [==============================] - 0s 84us/sample - loss: 3308645.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 195/5000
    500/500 [==============================] - 0s 78us/sample - loss: 53341363.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 196/5000
    500/500 [==============================] - 0s 86us/sample - loss: -10686031.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 197/5000
    500/500 [==============================] - 0s 78us/sample - loss: 78114353.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 198/5000
    500/500 [==============================] - 0s 80us/sample - loss: 50663483.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 199/5000
    500/500 [==============================] - 0s 78us/sample - loss: 39604087.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 200/5000
    500/500 [==============================] - 0s 98us/sample - loss: 221245788.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 201/5000
    500/500 [==============================] - 0s 92us/sample - loss: 70670500.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 202/5000
    500/500 [==============================] - 0s 88us/sample - loss: 174143957.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 203/5000
    500/500 [==============================] - 0s 80us/sample - loss: -190377947.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 204/5000
    500/500 [==============================] - 0s 78us/sample - loss: 92398942.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 205/5000
    500/500 [==============================] - 0s 76us/sample - loss: 40512157.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 206/5000
    500/500 [==============================] - 0s 84us/sample - loss: 10946088.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 207/5000
    500/500 [==============================] - 0s 80us/sample - loss: 69720200.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 208/5000
    500/500 [==============================] - 0s 86us/sample - loss: -54054226.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 209/5000
    500/500 [==============================] - 0s 80us/sample - loss: 127014189.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 210/5000
    500/500 [==============================] - 0s 86us/sample - loss: -95460631.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 211/5000
    500/500 [==============================] - 0s 76us/sample - loss: -6569632.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 212/5000
    500/500 [==============================] - 0s 76us/sample - loss: -204518168.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 213/5000
    500/500 [==============================] - 0s 76us/sample - loss: -215327729.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 214/5000
    500/500 [==============================] - 0s 90us/sample - loss: -103491975.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 215/5000
    500/500 [==============================] - 0s 94us/sample - loss: -27013151.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 216/5000
    500/500 [==============================] - 0s 92us/sample - loss: 4082911.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 217/5000
    500/500 [==============================] - 0s 84us/sample - loss: -78836171.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 218/5000
    500/500 [==============================] - 0s 80us/sample - loss: 63046527.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 219/5000
    500/500 [==============================] - 0s 76us/sample - loss: -173615580.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 220/5000
    500/500 [==============================] - 0s 86us/sample - loss: -107507067.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 221/5000
    500/500 [==============================] - 0s 76us/sample - loss: -133662485.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 222/5000
    500/500 [==============================] - 0s 78us/sample - loss: 115289274.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 223/5000
    500/500 [==============================] - 0s 80us/sample - loss: 125446232.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 224/5000
    500/500 [==============================] - 0s 90us/sample - loss: -16847352.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 225/5000
    500/500 [==============================] - 0s 88us/sample - loss: 187089589.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 226/5000
    500/500 [==============================] - 0s 78us/sample - loss: -169969213.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 227/5000
    500/500 [==============================] - 0s 88us/sample - loss: -109693869.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 228/5000
    500/500 [==============================] - 0s 78us/sample - loss: -1208218.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 229/5000
    500/500 [==============================] - 0s 78us/sample - loss: -20292831.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 230/5000
    500/500 [==============================] - 0s 76us/sample - loss: 94840496.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 231/5000
    500/500 [==============================] - 0s 82us/sample - loss: 38144969.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 232/5000
    500/500 [==============================] - 0s 78us/sample - loss: -170195091.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 233/5000
    500/500 [==============================] - 0s 86us/sample - loss: 14550954.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 234/5000
    500/500 [==============================] - 0s 78us/sample - loss: -148544931.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 235/5000
    500/500 [==============================] - 0s 76us/sample - loss: 2640071.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 236/5000
    500/500 [==============================] - 0s 78us/sample - loss: -152577212.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 237/5000
    500/500 [==============================] - 0s 84us/sample - loss: 90114069.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 238/5000
    500/500 [==============================] - 0s 78us/sample - loss: -117065668.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 239/5000
    500/500 [==============================] - 0s 82us/sample - loss: 33550443.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 240/5000
    500/500 [==============================] - 0s 74us/sample - loss: 20875269.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 241/5000
    500/500 [==============================] - 0s 80us/sample - loss: -5683410.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 242/5000
    500/500 [==============================] - 0s 76us/sample - loss: 74649511.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 243/5000
    500/500 [==============================] - 0s 84us/sample - loss: -36177776.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 244/5000
    500/500 [==============================] - 0s 74us/sample - loss: 69398006.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 245/5000
    500/500 [==============================] - 0s 82us/sample - loss: -56868655.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 246/5000
    500/500 [==============================] - 0s 72us/sample - loss: 69991411.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 247/5000
    500/500 [==============================] - 0s 74us/sample - loss: 85595905.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 248/5000
    500/500 [==============================] - 0s 84us/sample - loss: -115933562.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 249/5000
    500/500 [==============================] - 0s 76us/sample - loss: 136637214.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 250/5000
    500/500 [==============================] - 0s 82us/sample - loss: -104250742.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 251/5000
    500/500 [==============================] - 0s 74us/sample - loss: -91740028.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 252/5000
    500/500 [==============================] - 0s 74us/sample - loss: -117049757.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 253/5000
    500/500 [==============================] - 0s 76us/sample - loss: 75538147.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 254/5000
    500/500 [==============================] - 0s 76us/sample - loss: -49590586.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 255/5000
    500/500 [==============================] - 0s 88us/sample - loss: 35117001.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 256/5000
    500/500 [==============================] - 0s 86us/sample - loss: 9945885.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 257/5000
    500/500 [==============================] - 0s 86us/sample - loss: -15328010.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 258/5000
    500/500 [==============================] - 0s 91us/sample - loss: 115067359.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 259/5000
    500/500 [==============================] - 0s 84us/sample - loss: 35193363.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 260/5000
    500/500 [==============================] - 0s 80us/sample - loss: -190890391.4260 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 261/5000
    500/500 [==============================] - 0s 82us/sample - loss: 113328748.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 262/5000
    500/500 [==============================] - 0s 84us/sample - loss: -28376824.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 263/5000
    500/500 [==============================] - 0s 90us/sample - loss: -86969091.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 264/5000
    500/500 [==============================] - 0s 88us/sample - loss: 88063913.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 265/5000
    500/500 [==============================] - 0s 84us/sample - loss: -74483523.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 266/5000
    500/500 [==============================] - 0s 88us/sample - loss: -50151303.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 267/5000
    500/500 [==============================] - 0s 88us/sample - loss: -3551205.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 268/5000
    500/500 [==============================] - 0s 80us/sample - loss: 27820772.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 269/5000
    500/500 [==============================] - 0s 86us/sample - loss: 52274607.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 270/5000
    500/500 [==============================] - 0s 84us/sample - loss: 45304020.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 271/5000
    500/500 [==============================] - 0s 84us/sample - loss: 52657813.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 272/5000
    500/500 [==============================] - 0s 86us/sample - loss: -56383464.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 273/5000
    500/500 [==============================] - 0s 88us/sample - loss: 54843315.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 274/5000
    500/500 [==============================] - 0s 82us/sample - loss: -40629492.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 275/5000
    500/500 [==============================] - 0s 90us/sample - loss: 16896763.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 276/5000
    500/500 [==============================] - 0s 88us/sample - loss: 37756809.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 277/5000
    500/500 [==============================] - 0s 84us/sample - loss: -178737996.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 278/5000
    500/500 [==============================] - 0s 82us/sample - loss: 101910090.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 279/5000
    500/500 [==============================] - 0s 88us/sample - loss: -83006893.8563 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 280/5000
    500/500 [==============================] - 0s 90us/sample - loss: 17635718.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 281/5000
    500/500 [==============================] - 0s 82us/sample - loss: 99361969.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 282/5000
    500/500 [==============================] - 0s 90us/sample - loss: -27231374.9720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 283/5000
    500/500 [==============================] - 0s 92us/sample - loss: -122486428.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 284/5000
    500/500 [==============================] - 0s 88us/sample - loss: 36553593.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 285/5000
    500/500 [==============================] - 0s 82us/sample - loss: -82609294.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 286/5000
    500/500 [==============================] - 0s 88us/sample - loss: 160314712.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 287/5000
    500/500 [==============================] - 0s 78us/sample - loss: -166310244.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 288/5000
    500/500 [==============================] - 0s 88us/sample - loss: -22939877.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 289/5000
    500/500 [==============================] - 0s 88us/sample - loss: 26694919.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 290/5000
    500/500 [==============================] - 0s 78us/sample - loss: 132915115.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 291/5000
    500/500 [==============================] - 0s 84us/sample - loss: -88816311.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 292/5000
    500/500 [==============================] - 0s 80us/sample - loss: 27312554.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 293/5000
    500/500 [==============================] - 0s 86us/sample - loss: 127524481.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 294/5000
    500/500 [==============================] - 0s 82us/sample - loss: -63385342.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 295/5000
    500/500 [==============================] - 0s 94us/sample - loss: -63600216.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 296/5000
    500/500 [==============================] - 0s 94us/sample - loss: -141571137.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 297/5000
    500/500 [==============================] - 0s 94us/sample - loss: -97893241.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 298/5000
    500/500 [==============================] - 0s 94us/sample - loss: -31320482.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 299/5000
    500/500 [==============================] - 0s 106us/sample - loss: -19732431.8700 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 300/5000
    500/500 [==============================] - 0s 122us/sample - loss: -91997657.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 301/5000
    500/500 [==============================] - 0s 100us/sample - loss: -116688367.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 302/5000
    500/500 [==============================] - 0s 90us/sample - loss: 19110500.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 303/5000
    500/500 [==============================] - 0s 88us/sample - loss: -55567875.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 304/5000
    500/500 [==============================] - 0s 86us/sample - loss: 174752030.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 305/5000
    500/500 [==============================] - 0s 122us/sample - loss: -39307339.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 306/5000
    500/500 [==============================] - 0s 114us/sample - loss: -165016999.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 307/5000
    500/500 [==============================] - 0s 98us/sample - loss: 156089610.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 308/5000
    500/500 [==============================] - 0s 100us/sample - loss: -83848449.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 309/5000
    500/500 [==============================] - 0s 96us/sample - loss: -93210267.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 310/5000
    500/500 [==============================] - 0s 96us/sample - loss: 3473848.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 311/5000
    500/500 [==============================] - 0s 100us/sample - loss: 82062734.2760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 312/5000
    500/500 [==============================] - 0s 106us/sample - loss: -32625342.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 313/5000
    500/500 [==============================] - 0s 126us/sample - loss: -238761689.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 314/5000
    500/500 [==============================] - 0s 102us/sample - loss: 156361192.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 315/5000
    500/500 [==============================] - 0s 94us/sample - loss: 140926545.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 316/5000
    500/500 [==============================] - 0s 90us/sample - loss: -12347719.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 317/5000
    500/500 [==============================] - 0s 94us/sample - loss: 53006222.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 318/5000
    500/500 [==============================] - 0s 96us/sample - loss: 124174738.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 319/5000
    500/500 [==============================] - 0s 102us/sample - loss: -39371425.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 320/5000
    500/500 [==============================] - 0s 102us/sample - loss: -42362015.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 321/5000
    500/500 [==============================] - 0s 104us/sample - loss: -176735445.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 322/5000
    500/500 [==============================] - 0s 112us/sample - loss: 150068656.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 323/5000
    500/500 [==============================] - 0s 90us/sample - loss: 89582890.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 324/5000
    500/500 [==============================] - 0s 94us/sample - loss: 96545415.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 325/5000
    500/500 [==============================] - 0s 90us/sample - loss: 82288121.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 326/5000
    500/500 [==============================] - 0s 92us/sample - loss: 22776547.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 327/5000
    500/500 [==============================] - 0s 100us/sample - loss: -65713111.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 328/5000
    500/500 [==============================] - 0s 94us/sample - loss: 204981809.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 329/5000
    500/500 [==============================] - 0s 102us/sample - loss: 38501323.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 330/5000
    500/500 [==============================] - 0s 94us/sample - loss: 118145512.2520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 331/5000
    500/500 [==============================] - 0s 86us/sample - loss: 139002857.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 332/5000
    500/500 [==============================] - 0s 106us/sample - loss: 19188139.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 333/5000
    500/500 [==============================] - 0s 92us/sample - loss: -11181712.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 334/5000
    500/500 [==============================] - 0s 94us/sample - loss: -131356264.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 335/5000
    500/500 [==============================] - 0s 76us/sample - loss: -742965.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 336/5000
    500/500 [==============================] - 0s 72us/sample - loss: -63932176.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 337/5000
    500/500 [==============================] - 0s 84us/sample - loss: -58572946.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 338/5000
    500/500 [==============================] - 0s 96us/sample - loss: 28784026.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 339/5000
    500/500 [==============================] - 0s 76us/sample - loss: -87529000.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 340/5000
    500/500 [==============================] - 0s 82us/sample - loss: 29465701.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 341/5000
    500/500 [==============================] - 0s 78us/sample - loss: 220088709.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 342/5000
    500/500 [==============================] - 0s 82us/sample - loss: -76957231.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 343/5000
    500/500 [==============================] - 0s 76us/sample - loss: -13837660.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 344/5000
    500/500 [==============================] - 0s 82us/sample - loss: 95786680.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 345/5000
    500/500 [==============================] - 0s 76us/sample - loss: -5150168.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 346/5000
    500/500 [==============================] - 0s 84us/sample - loss: 104343566.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 347/5000
    500/500 [==============================] - 0s 74us/sample - loss: 35069135.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 348/5000
    500/500 [==============================] - 0s 84us/sample - loss: -78890065.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 349/5000
    500/500 [==============================] - 0s 76us/sample - loss: -1806552.6360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 350/5000
    500/500 [==============================] - 0s 84us/sample - loss: 47865303.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 351/5000
    500/500 [==============================] - 0s 80us/sample - loss: 126259926.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 352/5000
    500/500 [==============================] - 0s 82us/sample - loss: 97835828.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 353/5000
    500/500 [==============================] - 0s 76us/sample - loss: 11829132.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 354/5000
    500/500 [==============================] - 0s 84us/sample - loss: 5101105.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 355/5000
    500/500 [==============================] - 0s 76us/sample - loss: 14576416.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 356/5000
    500/500 [==============================] - 0s 84us/sample - loss: -37564471.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 357/5000
    500/500 [==============================] - 0s 78us/sample - loss: 78075586.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 358/5000
    500/500 [==============================] - 0s 74us/sample - loss: 104473659.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 359/5000
    500/500 [==============================] - 0s 86us/sample - loss: -126665711.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 360/5000
    500/500 [==============================] - 0s 78us/sample - loss: 65919188.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 361/5000
    500/500 [==============================] - 0s 80us/sample - loss: 8984350.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 362/5000
    500/500 [==============================] - 0s 74us/sample - loss: -86607746.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 363/5000
    500/500 [==============================] - 0s 80us/sample - loss: -100774702.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 364/5000
    500/500 [==============================] - 0s 87us/sample - loss: 110549854.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 365/5000
    500/500 [==============================] - 0s 80us/sample - loss: -41083645.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 366/5000
    500/500 [==============================] - 0s 86us/sample - loss: 95532742.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 367/5000
    500/500 [==============================] - 0s 86us/sample - loss: 127275348.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 368/5000
    500/500 [==============================] - 0s 74us/sample - loss: 93970044.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 369/5000
    500/500 [==============================] - 0s 82us/sample - loss: -161477133.5980 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 370/5000
    500/500 [==============================] - 0s 80us/sample - loss: -56498199.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 371/5000
    500/500 [==============================] - 0s 76us/sample - loss: -96252319.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 372/5000
    500/500 [==============================] - 0s 82us/sample - loss: -142891298.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 373/5000
    500/500 [==============================] - 0s 78us/sample - loss: 215924458.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 374/5000
    500/500 [==============================] - 0s 84us/sample - loss: -56197725.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 375/5000
    500/500 [==============================] - 0s 78us/sample - loss: -60467793.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 376/5000
    500/500 [==============================] - 0s 80us/sample - loss: 18087670.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 377/5000
    500/500 [==============================] - 0s 82us/sample - loss: -112680435.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 378/5000
    500/500 [==============================] - 0s 80us/sample - loss: -207162336.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 379/5000
    500/500 [==============================] - 0s 82us/sample - loss: 30284235.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 380/5000
    500/500 [==============================] - 0s 80us/sample - loss: 143869667.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 381/5000
    500/500 [==============================] - 0s 78us/sample - loss: -67284774.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 382/5000
    500/500 [==============================] - 0s 80us/sample - loss: -18415958.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 383/5000
    500/500 [==============================] - 0s 78us/sample - loss: 5827579.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 384/5000
    500/500 [==============================] - 0s 80us/sample - loss: -82499932.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 385/5000
    500/500 [==============================] - 0s 78us/sample - loss: -22461011.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 386/5000
    500/500 [==============================] - 0s 78us/sample - loss: -149083035.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 387/5000
    500/500 [==============================] - 0s 76us/sample - loss: 37046244.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 388/5000
    500/500 [==============================] - 0s 82us/sample - loss: 2896984.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 389/5000
    500/500 [==============================] - 0s 78us/sample - loss: -30814699.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 390/5000
    500/500 [==============================] - 0s 82us/sample - loss: -214091856.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 391/5000
    500/500 [==============================] - 0s 88us/sample - loss: -16267195.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 392/5000
    500/500 [==============================] - 0s 78us/sample - loss: -243750503.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 393/5000
    500/500 [==============================] - 0s 86us/sample - loss: 1000162.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 394/5000
    500/500 [==============================] - 0s 80us/sample - loss: 118869848.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 395/5000
    500/500 [==============================] - 0s 86us/sample - loss: -3127774.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 396/5000
    500/500 [==============================] - 0s 80us/sample - loss: 8987718.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 397/5000
    500/500 [==============================] - 0s 78us/sample - loss: -2158713.6020 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 398/5000
    500/500 [==============================] - 0s 76us/sample - loss: -5342633.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 399/5000
    500/500 [==============================] - 0s 86us/sample - loss: 156502824.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 400/5000
    500/500 [==============================] - 0s 56us/sample - loss: 50963684.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 401/5000
    500/500 [==============================] - 0s 105us/sample - loss: -81043468.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 402/5000
    500/500 [==============================] - 0s 80us/sample - loss: 110203809.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 403/5000
    500/500 [==============================] - 0s 82us/sample - loss: 50031465.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 404/5000
    500/500 [==============================] - 0s 80us/sample - loss: -85922996.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 405/5000
    500/500 [==============================] - 0s 98us/sample - loss: -129249211.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 406/5000
    500/500 [==============================] - 0s 92us/sample - loss: 22874281.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 407/5000
    500/500 [==============================] - 0s 90us/sample - loss: 73451332.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 408/5000
    500/500 [==============================] - 0s 82us/sample - loss: 162799583.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 409/5000
    500/500 [==============================] - 0s 94us/sample - loss: 984025.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 410/5000
    500/500 [==============================] - 0s 94us/sample - loss: -73363003.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 411/5000
    500/500 [==============================] - 0s 94us/sample - loss: 56310178.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 412/5000
    500/500 [==============================] - 0s 96us/sample - loss: -67207948.7080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 413/5000
    500/500 [==============================] - 0s 90us/sample - loss: 163986518.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 414/5000
    500/500 [==============================] - 0s 90us/sample - loss: 1780460.1130 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 415/5000
    500/500 [==============================] - 0s 84us/sample - loss: 30134844.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 416/5000
    500/500 [==============================] - 0s 92us/sample - loss: 27358634.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 417/5000
    500/500 [==============================] - 0s 94us/sample - loss: -34066.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 418/5000
    500/500 [==============================] - 0s 90us/sample - loss: -52044275.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 419/5000
    500/500 [==============================] - 0s 86us/sample - loss: 31156878.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 420/5000
    500/500 [==============================] - 0s 88us/sample - loss: 59255814.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 421/5000
    500/500 [==============================] - 0s 94us/sample - loss: -18014541.6140 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 422/5000
    500/500 [==============================] - 0s 92us/sample - loss: -231235701.6360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 423/5000
    500/500 [==============================] - 0s 90us/sample - loss: 125012734.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 424/5000
    500/500 [==============================] - 0s 88us/sample - loss: 39113359.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 425/5000
    500/500 [==============================] - 0s 86us/sample - loss: -34167777.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 426/5000
    500/500 [==============================] - 0s 92us/sample - loss: 99097079.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 427/5000
    500/500 [==============================] - 0s 94us/sample - loss: -66405067.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 428/5000
    500/500 [==============================] - 0s 90us/sample - loss: 182638870.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 429/5000
    500/500 [==============================] - 0s 82us/sample - loss: -39887449.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 430/5000
    500/500 [==============================] - 0s 88us/sample - loss: 92032184.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 431/5000
    500/500 [==============================] - 0s 92us/sample - loss: 127466937.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 432/5000
    500/500 [==============================] - 0s 92us/sample - loss: 100621098.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 433/5000
    500/500 [==============================] - 0s 84us/sample - loss: -62435142.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 434/5000
    500/500 [==============================] - 0s 82us/sample - loss: 2548382.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 435/5000
    500/500 [==============================] - 0s 94us/sample - loss: 47302461.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 436/5000
    500/500 [==============================] - 0s 94us/sample - loss: -91034303.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 437/5000
    500/500 [==============================] - 0s 80us/sample - loss: -148350635.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 438/5000
    500/500 [==============================] - 0s 86us/sample - loss: 87409807.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 439/5000
    500/500 [==============================] - 0s 86us/sample - loss: -45994644.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 440/5000
    500/500 [==============================] - 0s 82us/sample - loss: -152671241.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 441/5000
    500/500 [==============================] - 0s 90us/sample - loss: 117785895.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 442/5000
    500/500 [==============================] - 0s 90us/sample - loss: 117328771.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 443/5000
    500/500 [==============================] - 0s 82us/sample - loss: 20790047.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 444/5000
    500/500 [==============================] - 0s 88us/sample - loss: 8196463.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 445/5000
    500/500 [==============================] - 0s 88us/sample - loss: 93246212.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 446/5000
    500/500 [==============================] - 0s 86us/sample - loss: -158929084.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 447/5000
    500/500 [==============================] - 0s 82us/sample - loss: 71004214.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 448/5000
    500/500 [==============================] - 0s 94us/sample - loss: -62272284.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 449/5000
    500/500 [==============================] - 0s 92us/sample - loss: -113058470.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 450/5000
    500/500 [==============================] - 0s 88us/sample - loss: 170242810.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 451/5000
    500/500 [==============================] - 0s 80us/sample - loss: 49154448.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 452/5000
    500/500 [==============================] - ETA: 0s - loss: -474650112.0000 - accuracy: 0.0000e+0 - 0s 88us/sample - loss: -16707999.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 453/5000
    500/500 [==============================] - 0s 78us/sample - loss: 43249074.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 454/5000
    500/500 [==============================] - 0s 80us/sample - loss: -61510822.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 455/5000
    500/500 [==============================] - 0s 88us/sample - loss: -16124721.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 456/5000
    500/500 [==============================] - 0s 84us/sample - loss: -99867741.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 457/5000
    500/500 [==============================] - 0s 88us/sample - loss: -140145100.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 458/5000
    500/500 [==============================] - 0s 92us/sample - loss: -69914143.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 459/5000
    500/500 [==============================] - 0s 88us/sample - loss: 47324044.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 460/5000
    500/500 [==============================] - 0s 84us/sample - loss: 70642481.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 461/5000
    500/500 [==============================] - 0s 94us/sample - loss: 40858324.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 462/5000
    500/500 [==============================] - 0s 90us/sample - loss: 210108794.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 463/5000
    500/500 [==============================] - 0s 78us/sample - loss: 92087210.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 464/5000
    500/500 [==============================] - 0s 80us/sample - loss: -13792660.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 465/5000
    500/500 [==============================] - 0s 82us/sample - loss: 90004684.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 466/5000
    500/500 [==============================] - 0s 78us/sample - loss: 82993048.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 467/5000
    500/500 [==============================] - 0s 82us/sample - loss: -34078355.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 468/5000
    500/500 [==============================] - 0s 84us/sample - loss: 22332186.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 469/5000
    500/500 [==============================] - 0s 82us/sample - loss: 84730916.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 470/5000
    500/500 [==============================] - 0s 82us/sample - loss: 97254422.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 471/5000
    500/500 [==============================] - 0s 86us/sample - loss: 24518858.8780 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 472/5000
    500/500 [==============================] - 0s 84us/sample - loss: -40670644.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 473/5000
    500/500 [==============================] - 0s 88us/sample - loss: -96228292.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 474/5000
    500/500 [==============================] - 0s 96us/sample - loss: -30450743.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 475/5000
    500/500 [==============================] - 0s 90us/sample - loss: 116247229.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 476/5000
    500/500 [==============================] - 0s 76us/sample - loss: 75883489.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 477/5000
    500/500 [==============================] - 0s 86us/sample - loss: -6108373.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 478/5000
    500/500 [==============================] - 0s 76us/sample - loss: -16235736.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 479/5000
    500/500 [==============================] - 0s 86us/sample - loss: 98397134.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 480/5000
    500/500 [==============================] - 0s 80us/sample - loss: 42464027.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 481/5000
    500/500 [==============================] - 0s 84us/sample - loss: -70199342.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 482/5000
    500/500 [==============================] - 0s 78us/sample - loss: 23666901.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 483/5000
    500/500 [==============================] - 0s 76us/sample - loss: -89835331.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 484/5000
    500/500 [==============================] - 0s 76us/sample - loss: -75545463.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 485/5000
    500/500 [==============================] - 0s 88us/sample - loss: 28727193.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 486/5000
    500/500 [==============================] - 0s 76us/sample - loss: 160831160.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 487/5000
    500/500 [==============================] - 0s 92us/sample - loss: 197673778.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 488/5000
    500/500 [==============================] - 0s 92us/sample - loss: -142137226.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 489/5000
    500/500 [==============================] - 0s 80us/sample - loss: 87216033.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 490/5000
    500/500 [==============================] - 0s 86us/sample - loss: 41505213.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 491/5000
    500/500 [==============================] - 0s 84us/sample - loss: -101540288.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 492/5000
    500/500 [==============================] - 0s 82us/sample - loss: -4862721.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 493/5000
    500/500 [==============================] - 0s 90us/sample - loss: -45502539.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 494/5000
    500/500 [==============================] - 0s 86us/sample - loss: 238432455.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 495/5000
    500/500 [==============================] - 0s 82us/sample - loss: 46455915.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 496/5000
    500/500 [==============================] - 0s 88us/sample - loss: 18654286.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 497/5000
    500/500 [==============================] - 0s 90us/sample - loss: 28249737.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 498/5000
    500/500 [==============================] - 0s 82us/sample - loss: 52777588.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 499/5000
    500/500 [==============================] - 0s 88us/sample - loss: 10990841.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 500/5000
    500/500 [==============================] - 0s 90us/sample - loss: -116601960.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 501/5000
    500/500 [==============================] - 0s 84us/sample - loss: -167519388.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 502/5000
    500/500 [==============================] - 0s 86us/sample - loss: -63641165.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 503/5000
    500/500 [==============================] - 0s 98us/sample - loss: -53810966.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 504/5000
    500/500 [==============================] - 0s 96us/sample - loss: -43459340.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 505/5000
    500/500 [==============================] - 0s 88us/sample - loss: -109673198.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 506/5000
    500/500 [==============================] - 0s 98us/sample - loss: -3701616.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 507/5000
    500/500 [==============================] - 0s 84us/sample - loss: -64348253.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 508/5000
    500/500 [==============================] - 0s 96us/sample - loss: -24285927.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 509/5000
    500/500 [==============================] - 0s 76us/sample - loss: 73718002.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 510/5000
    500/500 [==============================] - 0s 96us/sample - loss: 29729988.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 511/5000
    500/500 [==============================] - 0s 84us/sample - loss: -11806388.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 512/5000
    500/500 [==============================] - 0s 96us/sample - loss: 72033863.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 513/5000
    500/500 [==============================] - 0s 76us/sample - loss: -102340659.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 514/5000
    500/500 [==============================] - 0s 86us/sample - loss: -16362799.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 515/5000
    500/500 [==============================] - 0s 94us/sample - loss: -63681989.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 516/5000
    500/500 [==============================] - 0s 80us/sample - loss: -32221146.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 517/5000
    500/500 [==============================] - 0s 86us/sample - loss: -24026934.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 518/5000
    500/500 [==============================] - 0s 96us/sample - loss: 19355815.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 519/5000
    500/500 [==============================] - 0s 78us/sample - loss: 84920873.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 520/5000
    500/500 [==============================] - 0s 86us/sample - loss: 32888736.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 521/5000
    500/500 [==============================] - 0s 74us/sample - loss: 68010920.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 522/5000
    500/500 [==============================] - 0s 80us/sample - loss: -33016541.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 523/5000
    500/500 [==============================] - 0s 76us/sample - loss: 31781583.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 524/5000
    500/500 [==============================] - 0s 80us/sample - loss: 54072782.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 525/5000
    500/500 [==============================] - 0s 80us/sample - loss: -97195375.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 526/5000
    500/500 [==============================] - 0s 88us/sample - loss: -23530763.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 527/5000
    500/500 [==============================] - 0s 74us/sample - loss: -53081948.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 528/5000
    500/500 [==============================] - 0s 78us/sample - loss: 73710264.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 529/5000
    500/500 [==============================] - 0s 80us/sample - loss: 24367866.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 530/5000
    500/500 [==============================] - 0s 82us/sample - loss: 948297.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 531/5000
    500/500 [==============================] - 0s 80us/sample - loss: -35263881.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 532/5000
    500/500 [==============================] - 0s 74us/sample - loss: 125848252.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 533/5000
    500/500 [==============================] - 0s 82us/sample - loss: 4134386.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 534/5000
    500/500 [==============================] - 0s 80us/sample - loss: -14221872.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 535/5000
    500/500 [==============================] - 0s 84us/sample - loss: -8988054.8170 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 536/5000
    500/500 [==============================] - 0s 86us/sample - loss: 44165593.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 537/5000
    500/500 [==============================] - 0s 80us/sample - loss: 65337004.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 538/5000
    500/500 [==============================] - 0s 80us/sample - loss: 37202992.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 539/5000
    500/500 [==============================] - 0s 80us/sample - loss: 10522793.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 540/5000
    500/500 [==============================] - 0s 78us/sample - loss: -52626051.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 541/5000
    500/500 [==============================] - 0s 90us/sample - loss: -59747917.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 542/5000
    500/500 [==============================] - 0s 82us/sample - loss: -98435608.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 543/5000
    500/500 [==============================] - 0s 88us/sample - loss: 140587990.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 544/5000
    500/500 [==============================] - 0s 80us/sample - loss: 141657270.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 545/5000
    500/500 [==============================] - 0s 82us/sample - loss: 109386164.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 546/5000
    500/500 [==============================] - 0s 78us/sample - loss: -23252714.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 547/5000
    500/500 [==============================] - 0s 136us/sample - loss: 102688248.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 548/5000
    500/500 [==============================] - 0s 114us/sample - loss: -23855752.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 549/5000
    500/500 [==============================] - 0s 90us/sample - loss: 124226370.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 550/5000
    500/500 [==============================] - 0s 80us/sample - loss: 26324993.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 551/5000
    500/500 [==============================] - 0s 80us/sample - loss: 91963745.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 552/5000
    500/500 [==============================] - 0s 90us/sample - loss: -76660690.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 553/5000
    500/500 [==============================] - 0s 74us/sample - loss: 103009618.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 554/5000
    500/500 [==============================] - 0s 82us/sample - loss: 97259591.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 555/5000
    500/500 [==============================] - 0s 90us/sample - loss: -130029644.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 556/5000
    500/500 [==============================] - 0s 78us/sample - loss: -171799665.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 557/5000
    500/500 [==============================] - 0s 94us/sample - loss: -45578788.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 558/5000
    500/500 [==============================] - 0s 104us/sample - loss: 125780727.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 559/5000
    500/500 [==============================] - 0s 90us/sample - loss: 119493727.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 560/5000
    500/500 [==============================] - 0s 106us/sample - loss: 185851672.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 561/5000
    500/500 [==============================] - 0s 94us/sample - loss: -24868819.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 562/5000
    500/500 [==============================] - 0s 96us/sample - loss: -21853087.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 563/5000
    500/500 [==============================] - 0s 94us/sample - loss: 175175376.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 564/5000
    500/500 [==============================] - 0s 92us/sample - loss: -27801742.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 565/5000
    500/500 [==============================] - 0s 96us/sample - loss: 14099557.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 566/5000
    500/500 [==============================] - 0s 82us/sample - loss: -16271926.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 567/5000
    500/500 [==============================] - 0s 78us/sample - loss: -29585828.2920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 568/5000
    500/500 [==============================] - 0s 78us/sample - loss: -59747073.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 569/5000
    500/500 [==============================] - 0s 86us/sample - loss: 20117990.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 570/5000
    500/500 [==============================] - 0s 76us/sample - loss: -197256750.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 571/5000
    500/500 [==============================] - 0s 72us/sample - loss: -24858866.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 572/5000
    500/500 [==============================] - 0s 78us/sample - loss: 85386882.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 573/5000
    500/500 [==============================] - 0s 78us/sample - loss: -64439708.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 574/5000
    500/500 [==============================] - 0s 82us/sample - loss: 15433204.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 575/5000
    500/500 [==============================] - 0s 76us/sample - loss: 55999343.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 576/5000
    500/500 [==============================] - 0s 78us/sample - loss: -32494164.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 577/5000
    500/500 [==============================] - 0s 78us/sample - loss: 156085496.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 578/5000
    500/500 [==============================] - 0s 86us/sample - loss: -19812762.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 579/5000
    500/500 [==============================] - 0s 78us/sample - loss: 111475417.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 580/5000
    500/500 [==============================] - 0s 80us/sample - loss: -35554295.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 581/5000
    500/500 [==============================] - 0s 82us/sample - loss: -95258795.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 582/5000
    500/500 [==============================] - 0s 82us/sample - loss: -96040677.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 583/5000
    500/500 [==============================] - 0s 84us/sample - loss: -121441674.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 584/5000
    500/500 [==============================] - 0s 82us/sample - loss: 43728717.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 585/5000
    500/500 [==============================] - 0s 84us/sample - loss: 49206689.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 586/5000
    500/500 [==============================] - 0s 80us/sample - loss: -38978820.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 587/5000
    500/500 [==============================] - 0s 84us/sample - loss: 146940942.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 588/5000
    500/500 [==============================] - 0s 92us/sample - loss: -87745286.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 589/5000
    500/500 [==============================] - 0s 78us/sample - loss: -57907836.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 590/5000
    500/500 [==============================] - 0s 84us/sample - loss: -103180776.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 591/5000
    500/500 [==============================] - 0s 82us/sample - loss: -111377180.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 592/5000
    500/500 [==============================] - 0s 92us/sample - loss: 20349266.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 593/5000
    500/500 [==============================] - 0s 88us/sample - loss: 10176641.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 594/5000
    500/500 [==============================] - 0s 92us/sample - loss: -13631731.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 595/5000
    500/500 [==============================] - 0s 78us/sample - loss: 98708084.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 596/5000
    500/500 [==============================] - 0s 84us/sample - loss: 18519836.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 597/5000
    500/500 [==============================] - 0s 88us/sample - loss: 17600161.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 598/5000
    500/500 [==============================] - 0s 84us/sample - loss: -20836762.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 599/5000
    500/500 [==============================] - 0s 94us/sample - loss: 29914492.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 600/5000
    500/500 [==============================] - 0s 76us/sample - loss: 104826103.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 601/5000
    500/500 [==============================] - 0s 80us/sample - loss: 39674806.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 602/5000
    500/500 [==============================] - 0s 78us/sample - loss: -28383512.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 603/5000
    500/500 [==============================] - 0s 86us/sample - loss: 86693615.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 604/5000
    500/500 [==============================] - 0s 80us/sample - loss: 9593476.6085 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 605/5000
    500/500 [==============================] - 0s 82us/sample - loss: 33016958.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 606/5000
    500/500 [==============================] - 0s 86us/sample - loss: 81627586.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 607/5000
    500/500 [==============================] - 0s 82us/sample - loss: -106124993.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 608/5000
    500/500 [==============================] - 0s 82us/sample - loss: -125410818.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 609/5000
    500/500 [==============================] - 0s 82us/sample - loss: 34183846.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 610/5000
    500/500 [==============================] - 0s 84us/sample - loss: 18015001.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 611/5000
    500/500 [==============================] - 0s 96us/sample - loss: 78396736.7679 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 612/5000
    500/500 [==============================] - 0s 78us/sample - loss: 21261105.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 613/5000
    500/500 [==============================] - 0s 80us/sample - loss: -86932280.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 614/5000
    500/500 [==============================] - 0s 80us/sample - loss: -32982394.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 615/5000
    500/500 [==============================] - 0s 88us/sample - loss: 179121817.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 616/5000
    500/500 [==============================] - 0s 74us/sample - loss: 24584502.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 617/5000
    500/500 [==============================] - 0s 76us/sample - loss: 40532862.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 618/5000
    500/500 [==============================] - 0s 78us/sample - loss: 19084089.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 619/5000
    500/500 [==============================] - 0s 82us/sample - loss: 42199100.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 620/5000
    500/500 [==============================] - 0s 80us/sample - loss: -47753852.8040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 621/5000
    500/500 [==============================] - 0s 84us/sample - loss: -141134382.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 622/5000
    500/500 [==============================] - 0s 78us/sample - loss: -79882149.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 623/5000
    500/500 [==============================] - 0s 82us/sample - loss: -134249134.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 624/5000
    500/500 [==============================] - 0s 80us/sample - loss: -42001637.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 625/5000
    500/500 [==============================] - 0s 78us/sample - loss: 18924279.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 626/5000
    500/500 [==============================] - 0s 82us/sample - loss: 94464534.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 627/5000
    500/500 [==============================] - 0s 75us/sample - loss: -125951513.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 628/5000
    500/500 [==============================] - 0s 78us/sample - loss: 6526365.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 629/5000
    500/500 [==============================] - 0s 84us/sample - loss: 27760063.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 630/5000
    500/500 [==============================] - 0s 80us/sample - loss: 152782395.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 631/5000
    500/500 [==============================] - 0s 82us/sample - loss: 13295548.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 632/5000
    500/500 [==============================] - 0s 82us/sample - loss: -28861906.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 633/5000
    500/500 [==============================] - 0s 84us/sample - loss: -144693779.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 634/5000
    500/500 [==============================] - 0s 80us/sample - loss: 88168348.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 635/5000
    500/500 [==============================] - 0s 86us/sample - loss: 35521050.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 636/5000
    500/500 [==============================] - 0s 78us/sample - loss: -42539896.3160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 637/5000
    500/500 [==============================] - 0s 82us/sample - loss: -182001765.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 638/5000
    500/500 [==============================] - 0s 80us/sample - loss: -77719161.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 639/5000
    500/500 [==============================] - 0s 80us/sample - loss: -91566676.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 640/5000
    500/500 [==============================] - 0s 86us/sample - loss: 30274743.2190 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 641/5000
    500/500 [==============================] - 0s 84us/sample - loss: -231920531.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 642/5000
    500/500 [==============================] - 0s 84us/sample - loss: 48586408.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 643/5000
    500/500 [==============================] - 0s 86us/sample - loss: 72088393.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 644/5000
    500/500 [==============================] - 0s 82us/sample - loss: 74658619.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 645/5000
    500/500 [==============================] - 0s 94us/sample - loss: 146335180.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 646/5000
    500/500 [==============================] - 0s 86us/sample - loss: 95258310.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 647/5000
    500/500 [==============================] - 0s 92us/sample - loss: 157746512.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 648/5000
    500/500 [==============================] - 0s 74us/sample - loss: 66609826.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 649/5000
    500/500 [==============================] - 0s 88us/sample - loss: 139557337.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 650/5000
    500/500 [==============================] - 0s 78us/sample - loss: -27309326.6600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 651/5000
    500/500 [==============================] - 0s 82us/sample - loss: -168006303.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 652/5000
    500/500 [==============================] - 0s 84us/sample - loss: 39926555.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 653/5000
    500/500 [==============================] - 0s 74us/sample - loss: -343675.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 654/5000
    500/500 [==============================] - 0s 110us/sample - loss: 246326826.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 655/5000
    500/500 [==============================] - 0s 102us/sample - loss: 38813814.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 656/5000
    500/500 [==============================] - 0s 95us/sample - loss: -68062008.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 657/5000
    500/500 [==============================] - 0s 104us/sample - loss: -43982231.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 658/5000
    500/500 [==============================] - 0s 96us/sample - loss: 15674339.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 659/5000
    500/500 [==============================] - 0s 87us/sample - loss: 161070693.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 660/5000
    500/500 [==============================] - 0s 104us/sample - loss: 109759723.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 661/5000
    500/500 [==============================] - 0s 94us/sample - loss: 56029752.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 662/5000
    500/500 [==============================] - 0s 105us/sample - loss: -20821255.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 663/5000
    500/500 [==============================] - 0s 88us/sample - loss: 27863998.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 664/5000
    500/500 [==============================] - 0s 94us/sample - loss: -137484218.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 665/5000
    500/500 [==============================] - 0s 94us/sample - loss: 165741872.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 666/5000
    500/500 [==============================] - 0s 88us/sample - loss: 6107676.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 667/5000
    500/500 [==============================] - 0s 112us/sample - loss: -85257138.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 668/5000
    500/500 [==============================] - 0s 102us/sample - loss: -56878998.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 669/5000
    500/500 [==============================] - 0s 89us/sample - loss: -47464375.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 670/5000
    500/500 [==============================] - 0s 92us/sample - loss: 206612358.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 671/5000
    500/500 [==============================] - 0s 90us/sample - loss: 140497275.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 672/5000
    500/500 [==============================] - 0s 92us/sample - loss: -148347964.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 673/5000
    500/500 [==============================] - 0s 104us/sample - loss: -65651317.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 674/5000
    500/500 [==============================] - 0s 94us/sample - loss: -41291507.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 675/5000
    500/500 [==============================] - 0s 92us/sample - loss: 7792994.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 676/5000
    500/500 [==============================] - 0s 92us/sample - loss: 99313844.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 677/5000
    500/500 [==============================] - 0s 106us/sample - loss: -68378550.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 678/5000
    500/500 [==============================] - 0s 98us/sample - loss: -48334116.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 679/5000
    500/500 [==============================] - 0s 97us/sample - loss: 195714954.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 680/5000
    500/500 [==============================] - 0s 96us/sample - loss: -237388689.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 681/5000
    500/500 [==============================] - 0s 94us/sample - loss: -71564850.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 682/5000
    500/500 [==============================] - 0s 90us/sample - loss: -29648941.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 683/5000
    500/500 [==============================] - 0s 90us/sample - loss: -96005842.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 684/5000
    500/500 [==============================] - 0s 106us/sample - loss: 53957439.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 685/5000
    500/500 [==============================] - 0s 96us/sample - loss: 10767069.3140 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 686/5000
    500/500 [==============================] - 0s 91us/sample - loss: 39762769.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 687/5000
    500/500 [==============================] - 0s 94us/sample - loss: 151229187.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 688/5000
    500/500 [==============================] - 0s 90us/sample - loss: -30139158.9100 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 689/5000
    500/500 [==============================] - 0s 110us/sample - loss: 98326074.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 690/5000
    500/500 [==============================] - 0s 96us/sample - loss: 216028834.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 691/5000
    500/500 [==============================] - 0s 118us/sample - loss: -16234415.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 692/5000
    500/500 [==============================] - 0s 100us/sample - loss: 73319063.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 693/5000
    500/500 [==============================] - 0s 97us/sample - loss: -65143128.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 694/5000
    500/500 [==============================] - 0s 80us/sample - loss: 165110949.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 695/5000
    500/500 [==============================] - 0s 82us/sample - loss: 117554970.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 696/5000
    500/500 [==============================] - 0s 88us/sample - loss: -42215994.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 697/5000
    500/500 [==============================] - 0s 96us/sample - loss: -38128644.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 698/5000
    500/500 [==============================] - 0s 82us/sample - loss: -34198423.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 699/5000
    500/500 [==============================] - 0s 90us/sample - loss: 125658043.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 700/5000
    500/500 [==============================] - 0s 78us/sample - loss: 161018529.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 701/5000
    500/500 [==============================] - 0s 82us/sample - loss: -204030856.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 702/5000
    500/500 [==============================] - 0s 78us/sample - loss: 120813018.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 703/5000
    500/500 [==============================] - 0s 86us/sample - loss: -36413534.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 704/5000
    500/500 [==============================] - 0s 80us/sample - loss: -3181811.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 705/5000
    500/500 [==============================] - 0s 84us/sample - loss: -239901653.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 706/5000
    500/500 [==============================] - 0s 96us/sample - loss: -32877969.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 707/5000
    500/500 [==============================] - 0s 84us/sample - loss: -129536606.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 708/5000
    500/500 [==============================] - 0s 80us/sample - loss: 12254327.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 709/5000
    500/500 [==============================] - 0s 88us/sample - loss: 76788612.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 710/5000
    500/500 [==============================] - 0s 74us/sample - loss: 57244229.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 711/5000
    500/500 [==============================] - 0s 76us/sample - loss: 115945101.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 712/5000
    500/500 [==============================] - 0s 78us/sample - loss: 150336391.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 713/5000
    500/500 [==============================] - 0s 78us/sample - loss: -9891621.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 714/5000
    500/500 [==============================] - 0s 80us/sample - loss: 10308350.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 715/5000
    500/500 [==============================] - 0s 82us/sample - loss: 32500816.5080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 716/5000
    500/500 [==============================] - 0s 80us/sample - loss: 49963301.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 717/5000
    500/500 [==============================] - 0s 84us/sample - loss: 86924604.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 718/5000
    500/500 [==============================] - 0s 80us/sample - loss: 84802113.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 719/5000
    500/500 [==============================] - 0s 82us/sample - loss: -77322324.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 720/5000
    500/500 [==============================] - 0s 84us/sample - loss: 75599734.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 721/5000
    500/500 [==============================] - 0s 84us/sample - loss: -125091282.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 722/5000
    500/500 [==============================] - 0s 80us/sample - loss: -36141913.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 723/5000
    500/500 [==============================] - 0s 77us/sample - loss: 10160788.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 724/5000
    500/500 [==============================] - 0s 78us/sample - loss: 43589179.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 725/5000
    500/500 [==============================] - 0s 84us/sample - loss: 37497106.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 726/5000
    500/500 [==============================] - 0s 82us/sample - loss: 40671306.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 727/5000
    500/500 [==============================] - 0s 76us/sample - loss: 146511102.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 728/5000
    500/500 [==============================] - 0s 80us/sample - loss: 33723566.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 729/5000
    500/500 [==============================] - 0s 82us/sample - loss: 109704309.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 730/5000
    500/500 [==============================] - 0s 80us/sample - loss: 52458255.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 731/5000
    500/500 [==============================] - 0s 84us/sample - loss: 50021005.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 732/5000
    500/500 [==============================] - 0s 82us/sample - loss: 154176115.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 733/5000
    500/500 [==============================] - 0s 82us/sample - loss: -164550645.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 734/5000
    500/500 [==============================] - 0s 80us/sample - loss: 9900109.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 735/5000
    500/500 [==============================] - 0s 84us/sample - loss: 153434656.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 736/5000
    500/500 [==============================] - 0s 88us/sample - loss: 14434179.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 737/5000
    500/500 [==============================] - 0s 82us/sample - loss: 15645291.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 738/5000
    500/500 [==============================] - 0s 73us/sample - loss: 29896352.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 739/5000
    500/500 [==============================] - 0s 80us/sample - loss: 84936804.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 740/5000
    500/500 [==============================] - 0s 82us/sample - loss: -54047919.3481 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 741/5000
    500/500 [==============================] - 0s 84us/sample - loss: 115967394.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 742/5000
    500/500 [==============================] - 0s 75us/sample - loss: 123202284.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 743/5000
    500/500 [==============================] - 0s 82us/sample - loss: -108852135.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 744/5000
    500/500 [==============================] - 0s 86us/sample - loss: 39505789.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 745/5000
    500/500 [==============================] - 0s 82us/sample - loss: 91101836.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 746/5000
    500/500 [==============================] - 0s 80us/sample - loss: 48572913.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 747/5000
    500/500 [==============================] - 0s 80us/sample - loss: 144610807.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 748/5000
    500/500 [==============================] - 0s 80us/sample - loss: -20160200.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 749/5000
    500/500 [==============================] - 0s 84us/sample - loss: 14124729.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 750/5000
    500/500 [==============================] - 0s 96us/sample - loss: -48682991.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 751/5000
    500/500 [==============================] - 0s 76us/sample - loss: -97171224.0635 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 752/5000
    500/500 [==============================] - 0s 88us/sample - loss: 24014476.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 753/5000
    500/500 [==============================] - 0s 82us/sample - loss: 52050132.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 754/5000
    500/500 [==============================] - 0s 84us/sample - loss: 27595011.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 755/5000
    500/500 [==============================] - 0s 82us/sample - loss: 114153678.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 756/5000
    500/500 [==============================] - 0s 74us/sample - loss: 115598545.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 757/5000
    500/500 [==============================] - 0s 78us/sample - loss: 43690715.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 758/5000
    500/500 [==============================] - 0s 82us/sample - loss: 113431050.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 759/5000
    500/500 [==============================] - 0s 80us/sample - loss: -42020420.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 760/5000
    500/500 [==============================] - 0s 76us/sample - loss: 137076977.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 761/5000
    500/500 [==============================] - 0s 80us/sample - loss: -2136824.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 762/5000
    500/500 [==============================] - 0s 82us/sample - loss: -156815808.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 763/5000
    500/500 [==============================] - 0s 82us/sample - loss: -139356853.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 764/5000
    500/500 [==============================] - 0s 82us/sample - loss: 22666861.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 765/5000
    500/500 [==============================] - 0s 80us/sample - loss: -94256525.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 766/5000
    500/500 [==============================] - 0s 82us/sample - loss: 26828237.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 767/5000
    500/500 [==============================] - 0s 78us/sample - loss: 173332373.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 768/5000
    500/500 [==============================] - 0s 82us/sample - loss: 105110859.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 769/5000
    500/500 [==============================] - 0s 78us/sample - loss: 11674360.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 770/5000
    500/500 [==============================] - 0s 82us/sample - loss: 43790766.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 771/5000
    500/500 [==============================] - 0s 78us/sample - loss: -29493531.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 772/5000
    500/500 [==============================] - 0s 82us/sample - loss: -83629823.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 773/5000
    500/500 [==============================] - 0s 78us/sample - loss: 925772.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 774/5000
    500/500 [==============================] - 0s 86us/sample - loss: 54815295.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 775/5000
    500/500 [==============================] - 0s 82us/sample - loss: -141745431.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 776/5000
    500/500 [==============================] - 0s 80us/sample - loss: 165375587.9640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 777/5000
    500/500 [==============================] - 0s 92us/sample - loss: 104420458.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 778/5000
    500/500 [==============================] - 0s 78us/sample - loss: -81062249.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 779/5000
    500/500 [==============================] - 0s 74us/sample - loss: 96799894.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 780/5000
    500/500 [==============================] - 0s 80us/sample - loss: -7326881.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 781/5000
    500/500 [==============================] - 0s 84us/sample - loss: 8010961.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 782/5000
    500/500 [==============================] - 0s 82us/sample - loss: 104730519.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 783/5000
    500/500 [==============================] - 0s 84us/sample - loss: 79719027.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 784/5000
    500/500 [==============================] - 0s 92us/sample - loss: 45090761.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 785/5000
    500/500 [==============================] - 0s 78us/sample - loss: -107572006.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 786/5000
    500/500 [==============================] - 0s 82us/sample - loss: -22113204.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 787/5000
    500/500 [==============================] - 0s 80us/sample - loss: -37584610.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 788/5000
    500/500 [==============================] - 0s 92us/sample - loss: 22101788.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 789/5000
    500/500 [==============================] - 0s 84us/sample - loss: -58047602.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 790/5000
    500/500 [==============================] - 0s 82us/sample - loss: -129816567.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 791/5000
    500/500 [==============================] - 0s 77us/sample - loss: 20492994.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 792/5000
    500/500 [==============================] - 0s 88us/sample - loss: 107290149.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 793/5000
    500/500 [==============================] - 0s 85us/sample - loss: 84992467.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 794/5000
    500/500 [==============================] - 0s 94us/sample - loss: 109549509.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 795/5000
    500/500 [==============================] - 0s 86us/sample - loss: 13714984.4520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 796/5000
    500/500 [==============================] - 0s 88us/sample - loss: 133594611.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 797/5000
    500/500 [==============================] - 0s 96us/sample - loss: -64235905.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 798/5000
    500/500 [==============================] - 0s 76us/sample - loss: 115474986.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 799/5000
    500/500 [==============================] - 0s 82us/sample - loss: 48645156.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 800/5000
    500/500 [==============================] - 0s 84us/sample - loss: 150891184.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 801/5000
    500/500 [==============================] - 0s 90us/sample - loss: -164566417.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 802/5000
    500/500 [==============================] - 0s 76us/sample - loss: 34473820.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 803/5000
    500/500 [==============================] - 0s 80us/sample - loss: 22555137.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 804/5000
    500/500 [==============================] - 0s 80us/sample - loss: -64076710.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 805/5000
    500/500 [==============================] - 0s 88us/sample - loss: 27804273.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 806/5000
    500/500 [==============================] - 0s 80us/sample - loss: -4159566.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 807/5000
    500/500 [==============================] - 0s 75us/sample - loss: -29188227.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 808/5000
    500/500 [==============================] - 0s 78us/sample - loss: 80433731.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 809/5000
    500/500 [==============================] - 0s 84us/sample - loss: -139601476.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 810/5000
    500/500 [==============================] - 0s 80us/sample - loss: -54193724.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 811/5000
    500/500 [==============================] - 0s 82us/sample - loss: -17899314.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 812/5000
    500/500 [==============================] - 0s 80us/sample - loss: 197801968.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 813/5000
    500/500 [==============================] - 0s 77us/sample - loss: -26719522.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 814/5000
    500/500 [==============================] - 0s 78us/sample - loss: -120558249.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 815/5000
    500/500 [==============================] - 0s 84us/sample - loss: 68333933.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 816/5000
    500/500 [==============================] - 0s 80us/sample - loss: 25660509.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 817/5000
    500/500 [==============================] - 0s 82us/sample - loss: 19723202.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 818/5000
    500/500 [==============================] - 0s 80us/sample - loss: -195889909.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 819/5000
    500/500 [==============================] - 0s 84us/sample - loss: -72332854.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 820/5000
    500/500 [==============================] - 0s 86us/sample - loss: 80622279.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 821/5000
    500/500 [==============================] - 0s 82us/sample - loss: 136539264.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 822/5000
    500/500 [==============================] - 0s 82us/sample - loss: 111213634.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 823/5000
    500/500 [==============================] - 0s 80us/sample - loss: 100032292.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 824/5000
    500/500 [==============================] - 0s 84us/sample - loss: 107525140.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 825/5000
    500/500 [==============================] - 0s 84us/sample - loss: -117399625.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 826/5000
    500/500 [==============================] - 0s 75us/sample - loss: -26404974.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 827/5000
    500/500 [==============================] - 0s 80us/sample - loss: -72107331.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 828/5000
    500/500 [==============================] - 0s 82us/sample - loss: 119809475.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 829/5000
    500/500 [==============================] - 0s 82us/sample - loss: -96016820.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 830/5000
    500/500 [==============================] - 0s 82us/sample - loss: 28678521.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 831/5000
    500/500 [==============================] - 0s 82us/sample - loss: -82762696.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 832/5000
    500/500 [==============================] - 0s 80us/sample - loss: 153093084.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 833/5000
    500/500 [==============================] - 0s 80us/sample - loss: 80383269.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 834/5000
    500/500 [==============================] - 0s 82us/sample - loss: -149419139.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 835/5000
    500/500 [==============================] - 0s 86us/sample - loss: -134753637.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 836/5000
    500/500 [==============================] - 0s 82us/sample - loss: 84888486.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 837/5000
    500/500 [==============================] - 0s 78us/sample - loss: 7603613.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 838/5000
    500/500 [==============================] - 0s 82us/sample - loss: -138380728.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 839/5000
    500/500 [==============================] - 0s 84us/sample - loss: 34029580.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 840/5000
    500/500 [==============================] - 0s 82us/sample - loss: 79508165.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 841/5000
    500/500 [==============================] - 0s 84us/sample - loss: 77173745.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 842/5000
    500/500 [==============================] - 0s 86us/sample - loss: 122208849.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 843/5000
    500/500 [==============================] - 0s 88us/sample - loss: 30782427.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 844/5000
    500/500 [==============================] - 0s 80us/sample - loss: 9252462.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 845/5000
    500/500 [==============================] - 0s 80us/sample - loss: 157162697.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 846/5000
    500/500 [==============================] - 0s 80us/sample - loss: -120589454.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 847/5000
    500/500 [==============================] - 0s 84us/sample - loss: -157519185.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 848/5000
    500/500 [==============================] - 0s 86us/sample - loss: -36638400.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 849/5000
    500/500 [==============================] - 0s 78us/sample - loss: 53400556.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 850/5000
    500/500 [==============================] - 0s 80us/sample - loss: -23058324.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 851/5000
    500/500 [==============================] - 0s 82us/sample - loss: -100255525.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 852/5000
    500/500 [==============================] - 0s 80us/sample - loss: -175235010.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 853/5000
    500/500 [==============================] - 0s 84us/sample - loss: 64471447.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 854/5000
    500/500 [==============================] - 0s 80us/sample - loss: 5366559.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 855/5000
    500/500 [==============================] - 0s 80us/sample - loss: -119067140.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 856/5000
    500/500 [==============================] - 0s 78us/sample - loss: 189323336.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 857/5000
    500/500 [==============================] - 0s 80us/sample - loss: -160709844.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 858/5000
    500/500 [==============================] - 0s 80us/sample - loss: 6692343.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 859/5000
    500/500 [==============================] - 0s 80us/sample - loss: -149735401.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 860/5000
    500/500 [==============================] - 0s 80us/sample - loss: 183196014.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 861/5000
    500/500 [==============================] - 0s 82us/sample - loss: 88648974.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 862/5000
    500/500 [==============================] - 0s 82us/sample - loss: 92629128.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 863/5000
    500/500 [==============================] - 0s 80us/sample - loss: 26080893.8230 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 864/5000
    500/500 [==============================] - 0s 82us/sample - loss: 36340823.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 865/5000
    500/500 [==============================] - 0s 80us/sample - loss: -66289357.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 866/5000
    500/500 [==============================] - 0s 84us/sample - loss: -6664088.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 867/5000
    500/500 [==============================] - 0s 82us/sample - loss: -50836217.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 868/5000
    500/500 [==============================] - 0s 84us/sample - loss: -79820757.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 869/5000
    500/500 [==============================] - 0s 84us/sample - loss: 68600118.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 870/5000
    500/500 [==============================] - 0s 82us/sample - loss: 36280044.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 871/5000
    500/500 [==============================] - 0s 84us/sample - loss: 72526519.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 872/5000
    500/500 [==============================] - 0s 82us/sample - loss: -88569796.8840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 873/5000
    500/500 [==============================] - 0s 82us/sample - loss: 49501382.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 874/5000
    500/500 [==============================] - 0s 80us/sample - loss: -13964457.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 875/5000
    500/500 [==============================] - 0s 82us/sample - loss: -1451510.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 876/5000
    500/500 [==============================] - 0s 82us/sample - loss: -53196248.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 877/5000
    500/500 [==============================] - 0s 90us/sample - loss: -77470748.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 878/5000
    500/500 [==============================] - 0s 76us/sample - loss: 185837101.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 879/5000
    500/500 [==============================] - 0s 84us/sample - loss: -37226989.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 880/5000
    500/500 [==============================] - 0s 82us/sample - loss: -13186461.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 881/5000
    500/500 [==============================] - 0s 86us/sample - loss: -18477554.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 882/5000
    500/500 [==============================] - 0s 84us/sample - loss: 69173338.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 883/5000
    500/500 [==============================] - 0s 80us/sample - loss: -116929010.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 884/5000
    500/500 [==============================] - 0s 86us/sample - loss: 17532897.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 885/5000
    500/500 [==============================] - 0s 88us/sample - loss: 35328429.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 886/5000
    500/500 [==============================] - 0s 84us/sample - loss: -81711636.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 887/5000
    500/500 [==============================] - 0s 94us/sample - loss: -76168359.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 888/5000
    500/500 [==============================] - 0s 74us/sample - loss: -51176426.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 889/5000
    500/500 [==============================] - 0s 88us/sample - loss: -111421494.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 890/5000
    500/500 [==============================] - 0s 94us/sample - loss: 180657591.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 891/5000
    500/500 [==============================] - 0s 78us/sample - loss: -23697577.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 892/5000
    500/500 [==============================] - 0s 79us/sample - loss: 127643582.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 893/5000
    500/500 [==============================] - 0s 86us/sample - loss: -35862155.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 894/5000
    500/500 [==============================] - 0s 80us/sample - loss: 111023402.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 895/5000
    500/500 [==============================] - 0s 88us/sample - loss: 145297204.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 896/5000
    500/500 [==============================] - 0s 80us/sample - loss: 73884441.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 897/5000
    500/500 [==============================] - 0s 88us/sample - loss: -58171208.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 898/5000
    500/500 [==============================] - 0s 76us/sample - loss: 19854203.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 899/5000
    500/500 [==============================] - 0s 76us/sample - loss: 48452598.9763 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 900/5000
    500/500 [==============================] - 0s 80us/sample - loss: 164382241.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 901/5000
    500/500 [==============================] - 0s 82us/sample - loss: 88990042.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 902/5000
    500/500 [==============================] - 0s 80us/sample - loss: 68150423.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 903/5000
    500/500 [==============================] - 0s 84us/sample - loss: 37977103.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 904/5000
    500/500 [==============================] - 0s 78us/sample - loss: -17182159.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 905/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32097325.8220 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 906/5000
    500/500 [==============================] - 0s 80us/sample - loss: 158366247.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 907/5000
    500/500 [==============================] - 0s 82us/sample - loss: -105208680.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 908/5000
    500/500 [==============================] - 0s 78us/sample - loss: -85427072.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 909/5000
    500/500 [==============================] - 0s 82us/sample - loss: -188637084.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 910/5000
    500/500 [==============================] - 0s 80us/sample - loss: 160476172.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 911/5000
    500/500 [==============================] - 0s 82us/sample - loss: 93848611.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 912/5000
    500/500 [==============================] - 0s 78us/sample - loss: -75622175.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 913/5000
    500/500 [==============================] - 0s 84us/sample - loss: 47567743.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 914/5000
    500/500 [==============================] - 0s 86us/sample - loss: -106260748.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 915/5000
    500/500 [==============================] - 0s 74us/sample - loss: 90218014.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 916/5000
    500/500 [==============================] - 0s 80us/sample - loss: 61659419.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 917/5000
    500/500 [==============================] - 0s 76us/sample - loss: -100166369.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 918/5000
    500/500 [==============================] - 0s 80us/sample - loss: -72570628.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 919/5000
    500/500 [==============================] - 0s 84us/sample - loss: 130194771.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 920/5000
    500/500 [==============================] - 0s 82us/sample - loss: -127542950.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 921/5000
    500/500 [==============================] - 0s 86us/sample - loss: -94785636.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 922/5000
    500/500 [==============================] - 0s 88us/sample - loss: -45144636.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 923/5000
    500/500 [==============================] - 0s 78us/sample - loss: 246385472.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 924/5000
    500/500 [==============================] - 0s 82us/sample - loss: -160768037.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 925/5000
    500/500 [==============================] - 0s 86us/sample - loss: 26148888.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 926/5000
    500/500 [==============================] - 0s 76us/sample - loss: -13464908.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 927/5000
    500/500 [==============================] - 0s 86us/sample - loss: -107945174.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 928/5000
    500/500 [==============================] - 0s 82us/sample - loss: -125751172.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 929/5000
    500/500 [==============================] - 0s 84us/sample - loss: -16436977.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 930/5000
    500/500 [==============================] - 0s 84us/sample - loss: -34487795.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 931/5000
    500/500 [==============================] - 0s 86us/sample - loss: 4995143.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 932/5000
    500/500 [==============================] - 0s 84us/sample - loss: 200682931.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 933/5000
    500/500 [==============================] - 0s 82us/sample - loss: -71485956.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 934/5000
    500/500 [==============================] - 0s 82us/sample - loss: -2299273.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 935/5000
    500/500 [==============================] - 0s 80us/sample - loss: -44545015.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 936/5000
    500/500 [==============================] - 0s 86us/sample - loss: -59487273.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 937/5000
    500/500 [==============================] - 0s 84us/sample - loss: -29335048.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 938/5000
    500/500 [==============================] - 0s 82us/sample - loss: 11727722.2420 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 939/5000
    500/500 [==============================] - 0s 92us/sample - loss: -74479624.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 940/5000
    500/500 [==============================] - 0s 76us/sample - loss: 20832794.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 941/5000
    500/500 [==============================] - 0s 73us/sample - loss: -89643890.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 942/5000
    500/500 [==============================] - 0s 84us/sample - loss: -35841003.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 943/5000
    500/500 [==============================] - 0s 78us/sample - loss: -191545452.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 944/5000
    500/500 [==============================] - 0s 78us/sample - loss: 58200937.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 945/5000
    500/500 [==============================] - 0s 80us/sample - loss: 181897614.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 946/5000
    500/500 [==============================] - 0s 80us/sample - loss: 140129049.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 947/5000
    500/500 [==============================] - 0s 80us/sample - loss: -14830996.1100 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 948/5000
    500/500 [==============================] - 0s 78us/sample - loss: -4426090.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 949/5000
    500/500 [==============================] - 0s 82us/sample - loss: 176173576.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 950/5000
    500/500 [==============================] - 0s 78us/sample - loss: 91308481.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 951/5000
    500/500 [==============================] - 0s 82us/sample - loss: -91581806.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 952/5000
    500/500 [==============================] - 0s 80us/sample - loss: 10586427.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 953/5000
    500/500 [==============================] - 0s 82us/sample - loss: 25062190.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 954/5000
    500/500 [==============================] - 0s 78us/sample - loss: 145150211.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 955/5000
    500/500 [==============================] - 0s 84us/sample - loss: 34235945.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 956/5000
    500/500 [==============================] - 0s 78us/sample - loss: 119650821.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 957/5000
    500/500 [==============================] - 0s 84us/sample - loss: -38159109.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 958/5000
    500/500 [==============================] - 0s 92us/sample - loss: -5677853.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 959/5000
    500/500 [==============================] - 0s 76us/sample - loss: -53286571.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 960/5000
    500/500 [==============================] - 0s 84us/sample - loss: -132387982.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 961/5000
    500/500 [==============================] - 0s 80us/sample - loss: 163178018.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 962/5000
    500/500 [==============================] - 0s 88us/sample - loss: 16628889.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 963/5000
    500/500 [==============================] - 0s 88us/sample - loss: 15841238.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 964/5000
    500/500 [==============================] - 0s 80us/sample - loss: -22512151.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 965/5000
    500/500 [==============================] - 0s 84us/sample - loss: 18994329.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 966/5000
    500/500 [==============================] - 0s 86us/sample - loss: 55718807.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 967/5000
    500/500 [==============================] - 0s 78us/sample - loss: 31878467.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 968/5000
    500/500 [==============================] - 0s 84us/sample - loss: 22120476.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 969/5000
    500/500 [==============================] - 0s 80us/sample - loss: 70168446.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 970/5000
    500/500 [==============================] - 0s 84us/sample - loss: -21192549.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 971/5000
    500/500 [==============================] - 0s 90us/sample - loss: -17392501.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 972/5000
    500/500 [==============================] - 0s 74us/sample - loss: -124140985.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 973/5000
    500/500 [==============================] - 0s 80us/sample - loss: -155531513.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 974/5000
    500/500 [==============================] - 0s 80us/sample - loss: 42660983.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 975/5000
    500/500 [==============================] - 0s 92us/sample - loss: -142847027.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 976/5000
    500/500 [==============================] - 0s 76us/sample - loss: 41643187.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 977/5000
    500/500 [==============================] - 0s 84us/sample - loss: -50613042.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 978/5000
    500/500 [==============================] - 0s 84us/sample - loss: 14107970.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 979/5000
    500/500 [==============================] - 0s 86us/sample - loss: -109130319.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 980/5000
    500/500 [==============================] - 0s 92us/sample - loss: 62285815.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 981/5000
    500/500 [==============================] - 0s 78us/sample - loss: 88305722.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 982/5000
    500/500 [==============================] - 0s 75us/sample - loss: 92227550.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 983/5000
    500/500 [==============================] - 0s 86us/sample - loss: -65200766.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 984/5000
    500/500 [==============================] - 0s 80us/sample - loss: -52285805.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 985/5000
    500/500 [==============================] - 0s 86us/sample - loss: 47553137.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 986/5000
    500/500 [==============================] - 0s 82us/sample - loss: 22038583.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 987/5000
    500/500 [==============================] - 0s 84us/sample - loss: 96812237.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 988/5000
    500/500 [==============================] - 0s 92us/sample - loss: -93878739.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 989/5000
    500/500 [==============================] - 0s 76us/sample - loss: 75704688.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 990/5000
    500/500 [==============================] - 0s 80us/sample - loss: -148565021.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 991/5000
    500/500 [==============================] - 0s 286us/sample - loss: -26830521.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 992/5000
    500/500 [==============================] - 0s 84us/sample - loss: -248999755.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 993/5000
    500/500 [==============================] - 0s 80us/sample - loss: -60270258.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 994/5000
    500/500 [==============================] - 0s 78us/sample - loss: 16647312.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 995/5000
    500/500 [==============================] - 0s 81us/sample - loss: -67163345.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 996/5000
    500/500 [==============================] - 0s 84us/sample - loss: -60094482.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 997/5000
    500/500 [==============================] - 0s 83us/sample - loss: -37605110.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 998/5000
    500/500 [==============================] - 0s 84us/sample - loss: -121167327.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 999/5000
    500/500 [==============================] - 0s 83us/sample - loss: -8259755.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1000/5000
    500/500 [==============================] - 0s 84us/sample - loss: -60417546.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1001/5000
    500/500 [==============================] - 0s 86us/sample - loss: 52978890.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1002/5000
    500/500 [==============================] - 0s 80us/sample - loss: -55330095.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1003/5000
    500/500 [==============================] - 0s 84us/sample - loss: 114721927.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1004/5000
    500/500 [==============================] - 0s 82us/sample - loss: -52433968.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1005/5000
    500/500 [==============================] - 0s 84us/sample - loss: -12545711.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1006/5000
    500/500 [==============================] - 0s 80us/sample - loss: 82765938.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1007/5000
    500/500 [==============================] - 0s 86us/sample - loss: -44598850.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1008/5000
    500/500 [==============================] - 0s 80us/sample - loss: -46723196.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1009/5000
    500/500 [==============================] - 0s 84us/sample - loss: 143995082.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1010/5000
    500/500 [==============================] - 0s 84us/sample - loss: -72836670.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1011/5000
    500/500 [==============================] - 0s 78us/sample - loss: 154445327.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1012/5000
    500/500 [==============================] - 0s 92us/sample - loss: 32504402.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1013/5000
    500/500 [==============================] - 0s 76us/sample - loss: -198368864.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1014/5000
    500/500 [==============================] - 0s 74us/sample - loss: -77462810.8840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1015/5000
    500/500 [==============================] - 0s 78us/sample - loss: 88102643.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1016/5000
    500/500 [==============================] - 0s 84us/sample - loss: -83466789.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1017/5000
    500/500 [==============================] - 0s 82us/sample - loss: -18102534.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1018/5000
    500/500 [==============================] - 0s 90us/sample - loss: -125678470.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1019/5000
    500/500 [==============================] - 0s 120us/sample - loss: 33574678.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1020/5000
    500/500 [==============================] - 0s 108us/sample - loss: -93201651.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1021/5000
    500/500 [==============================] - 0s 100us/sample - loss: -9790.8500 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1022/5000
    500/500 [==============================] - 0s 106us/sample - loss: 170650788.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1023/5000
    500/500 [==============================] - 0s 104us/sample - loss: -134518640.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1024/5000
    500/500 [==============================] - 0s 96us/sample - loss: 73407377.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1025/5000
    500/500 [==============================] - 0s 88us/sample - loss: 28259201.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1026/5000
    500/500 [==============================] - 0s 88us/sample - loss: 99563044.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1027/5000
    500/500 [==============================] - 0s 112us/sample - loss: 204266182.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1028/5000
    500/500 [==============================] - 0s 100us/sample - loss: 94323221.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1029/5000
    500/500 [==============================] - 0s 102us/sample - loss: 100371582.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1030/5000
    500/500 [==============================] - 0s 102us/sample - loss: -110874751.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1031/5000
    500/500 [==============================] - 0s 108us/sample - loss: 44424023.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1032/5000
    500/500 [==============================] - 0s 93us/sample - loss: -63798957.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1033/5000
    500/500 [==============================] - 0s 88us/sample - loss: -146773198.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1034/5000
    500/500 [==============================] - 0s 104us/sample - loss: 142561668.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1035/5000
    500/500 [==============================] - 0s 96us/sample - loss: 11326668.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1036/5000
    500/500 [==============================] - 0s 96us/sample - loss: -154657933.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1037/5000
    500/500 [==============================] - 0s 93us/sample - loss: 19300200.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1038/5000
    500/500 [==============================] - 0s 86us/sample - loss: 160553399.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1039/5000
    500/500 [==============================] - 0s 92us/sample - loss: 43730726.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1040/5000
    500/500 [==============================] - 0s 100us/sample - loss: 51280872.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1041/5000
    500/500 [==============================] - 0s 100us/sample - loss: 62780347.2440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1042/5000
    500/500 [==============================] - 0s 91us/sample - loss: 156823163.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1043/5000
    500/500 [==============================] - 0s 108us/sample - loss: 127178719.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1044/5000
    500/500 [==============================] - 0s 102us/sample - loss: -75124867.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1045/5000
    500/500 [==============================] - 0s 95us/sample - loss: 16824917.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1046/5000
    500/500 [==============================] - 0s 92us/sample - loss: -118183206.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1047/5000
    500/500 [==============================] - 0s 88us/sample - loss: -111098815.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1048/5000
    500/500 [==============================] - 0s 96us/sample - loss: 130407451.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1049/5000
    500/500 [==============================] - 0s 90us/sample - loss: 129718325.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1050/5000
    500/500 [==============================] - 0s 102us/sample - loss: -62079241.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1051/5000
    500/500 [==============================] - 0s 96us/sample - loss: 64848411.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1052/5000
    500/500 [==============================] - 0s 93us/sample - loss: 173477501.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1053/5000
    500/500 [==============================] - 0s 100us/sample - loss: -80858907.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1054/5000
    500/500 [==============================] - 0s 98us/sample - loss: 144603769.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1055/5000
    500/500 [==============================] - 0s 89us/sample - loss: 17031806.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1056/5000
    500/500 [==============================] - 0s 92us/sample - loss: -72715236.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1057/5000
    500/500 [==============================] - 0s 100us/sample - loss: 72098173.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1058/5000
    500/500 [==============================] - 0s 100us/sample - loss: -163793119.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1059/5000
    500/500 [==============================] - 0s 85us/sample - loss: 110673982.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1060/5000
    500/500 [==============================] - 0s 84us/sample - loss: -138809187.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1061/5000
    500/500 [==============================] - 0s 84us/sample - loss: 128261223.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1062/5000
    500/500 [==============================] - 0s 118us/sample - loss: -88956555.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1063/5000
    500/500 [==============================] - 0s 82us/sample - loss: -140029716.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1064/5000
    500/500 [==============================] - 0s 88us/sample - loss: 25730596.4520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1065/5000
    500/500 [==============================] - 0s 92us/sample - loss: 58928018.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1066/5000
    500/500 [==============================] - 0s 76us/sample - loss: -205787547.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1067/5000
    500/500 [==============================] - 0s 79us/sample - loss: 32981232.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1068/5000
    500/500 [==============================] - 0s 78us/sample - loss: -65033824.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1069/5000
    500/500 [==============================] - 0s 102us/sample - loss: -125682837.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1070/5000
    500/500 [==============================] - 0s 76us/sample - loss: 138137817.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1071/5000
    500/500 [==============================] - 0s 88us/sample - loss: -80190415.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1072/5000
    500/500 [==============================] - 0s 76us/sample - loss: 71070872.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1073/5000
    500/500 [==============================] - 0s 82us/sample - loss: 44669436.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1074/5000
    500/500 [==============================] - 0s 82us/sample - loss: 154523476.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1075/5000
    500/500 [==============================] - 0s 84us/sample - loss: -64902775.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1076/5000
    500/500 [==============================] - 0s 82us/sample - loss: 83600377.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1077/5000
    500/500 [==============================] - 0s 84us/sample - loss: -8528325.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1078/5000
    500/500 [==============================] - 0s 80us/sample - loss: -92132646.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1079/5000
    500/500 [==============================] - 0s 82us/sample - loss: 18412413.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1080/5000
    500/500 [==============================] - 0s 94us/sample - loss: -97867763.1700 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1081/5000
    500/500 [==============================] - 0s 78us/sample - loss: -231451760.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1082/5000
    500/500 [==============================] - 0s 74us/sample - loss: 112458574.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1083/5000
    500/500 [==============================] - 0s 80us/sample - loss: 84123706.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1084/5000
    500/500 [==============================] - 0s 82us/sample - loss: 153603163.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1085/5000
    500/500 [==============================] - 0s 80us/sample - loss: -82773932.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1086/5000
    500/500 [==============================] - 0s 82us/sample - loss: -18776643.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1087/5000
    500/500 [==============================] - 0s 82us/sample - loss: -145822909.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1088/5000
    500/500 [==============================] - 0s 74us/sample - loss: 118833064.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1089/5000
    500/500 [==============================] - 0s 78us/sample - loss: 96517920.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1090/5000
    500/500 [==============================] - 0s 86us/sample - loss: 78725693.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1091/5000
    500/500 [==============================] - 0s 82us/sample - loss: -75808151.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1092/5000
    500/500 [==============================] - 0s 82us/sample - loss: -62297870.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1093/5000
    500/500 [==============================] - 0s 80us/sample - loss: -58368195.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1094/5000
    500/500 [==============================] - 0s 82us/sample - loss: 23277033.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1095/5000
    500/500 [==============================] - 0s 80us/sample - loss: -53448329.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1096/5000
    500/500 [==============================] - 0s 80us/sample - loss: -22234048.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1097/5000
    500/500 [==============================] - 0s 80us/sample - loss: -89794494.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1098/5000
    500/500 [==============================] - 0s 82us/sample - loss: 136209635.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1099/5000
    500/500 [==============================] - 0s 84us/sample - loss: 249719873.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1100/5000
    500/500 [==============================] - 0s 82us/sample - loss: -69462818.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1101/5000
    500/500 [==============================] - 0s 82us/sample - loss: -78312454.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1102/5000
    500/500 [==============================] - 0s 84us/sample - loss: -149571213.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1103/5000
    500/500 [==============================] - 0s 76us/sample - loss: 74455331.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1104/5000
    500/500 [==============================] - 0s 78us/sample - loss: 15334450.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1105/5000
    500/500 [==============================] - 0s 82us/sample - loss: 43523644.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1106/5000
    500/500 [==============================] - 0s 80us/sample - loss: 138204781.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1107/5000
    500/500 [==============================] - 0s 82us/sample - loss: 50149416.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1108/5000
    500/500 [==============================] - 0s 82us/sample - loss: 111814850.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1109/5000
    500/500 [==============================] - 0s 80us/sample - loss: 34893093.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1110/5000
    500/500 [==============================] - 0s 90us/sample - loss: 5223502.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1111/5000
    500/500 [==============================] - 0s 76us/sample - loss: -121544276.1580 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1112/5000
    500/500 [==============================] - 0s 84us/sample - loss: 80073879.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1113/5000
    500/500 [==============================] - 0s 78us/sample - loss: 171916739.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1114/5000
    500/500 [==============================] - 0s 88us/sample - loss: 96306906.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1115/5000
    500/500 [==============================] - 0s 82us/sample - loss: 23922995.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1116/5000
    500/500 [==============================] - 0s 82us/sample - loss: 166909487.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1117/5000
    500/500 [==============================] - 0s 84us/sample - loss: -159803205.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1118/5000
    500/500 [==============================] - 0s 84us/sample - loss: -77975104.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1119/5000
    500/500 [==============================] - 0s 84us/sample - loss: 182560982.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1120/5000
    500/500 [==============================] - 0s 80us/sample - loss: 5158147.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1121/5000
    500/500 [==============================] - 0s 80us/sample - loss: 45345678.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1122/5000
    500/500 [==============================] - 0s 96us/sample - loss: 2056405.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1123/5000
    500/500 [==============================] - 0s 78us/sample - loss: 93782579.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1124/5000
    500/500 [==============================] - 0s 76us/sample - loss: 38247166.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1125/5000
    500/500 [==============================] - 0s 82us/sample - loss: -1220592.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1126/5000
    500/500 [==============================] - 0s 92us/sample - loss: 78167738.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1127/5000
    500/500 [==============================] - 0s 74us/sample - loss: -66717264.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1128/5000
    500/500 [==============================] - 0s 74us/sample - loss: -71082935.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1129/5000
    500/500 [==============================] - 0s 78us/sample - loss: -7641006.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1130/5000
    500/500 [==============================] - 0s 82us/sample - loss: -83281311.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1131/5000
    500/500 [==============================] - 0s 80us/sample - loss: -95639901.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1132/5000
    500/500 [==============================] - 0s 80us/sample - loss: 87634804.1240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1133/5000
    500/500 [==============================] - 0s 78us/sample - loss: -11795522.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1134/5000
    500/500 [==============================] - 0s 84us/sample - loss: -190780240.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1135/5000
    500/500 [==============================] - 0s 80us/sample - loss: -57780258.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1136/5000
    500/500 [==============================] - 0s 82us/sample - loss: 5076991.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1137/5000
    500/500 [==============================] - 0s 78us/sample - loss: 114395058.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1138/5000
    500/500 [==============================] - 0s 74us/sample - loss: 80288109.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1139/5000
    500/500 [==============================] - 0s 80us/sample - loss: 31534402.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1140/5000
    500/500 [==============================] - 0s 84us/sample - loss: 46850681.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1141/5000
    500/500 [==============================] - 0s 80us/sample - loss: 66617963.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1142/5000
    500/500 [==============================] - 0s 76us/sample - loss: 91634020.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1143/5000
    500/500 [==============================] - 0s 82us/sample - loss: 160598396.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1144/5000
    500/500 [==============================] - 0s 86us/sample - loss: 16214275.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1145/5000
    500/500 [==============================] - 0s 82us/sample - loss: 88631006.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1146/5000
    500/500 [==============================] - 0s 86us/sample - loss: 20890230.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1147/5000
    500/500 [==============================] - 0s 86us/sample - loss: -61449205.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1148/5000
    500/500 [==============================] - 0s 78us/sample - loss: 124028303.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1149/5000
    500/500 [==============================] - 0s 86us/sample - loss: -72421835.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1150/5000
    500/500 [==============================] - 0s 80us/sample - loss: 33445896.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1151/5000
    500/500 [==============================] - 0s 80us/sample - loss: -27507590.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1152/5000
    500/500 [==============================] - 0s 78us/sample - loss: -16312527.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1153/5000
    500/500 [==============================] - 0s 84us/sample - loss: 84914009.7960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1154/5000
    500/500 [==============================] - 0s 80us/sample - loss: 43554990.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1155/5000
    500/500 [==============================] - 0s 84us/sample - loss: 95077594.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1156/5000
    500/500 [==============================] - 0s 90us/sample - loss: -152003451.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1157/5000
    500/500 [==============================] - 0s 76us/sample - loss: 79638323.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1158/5000
    500/500 [==============================] - 0s 82us/sample - loss: -77855229.3160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1159/5000
    500/500 [==============================] - 0s 80us/sample - loss: -129046444.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1160/5000
    500/500 [==============================] - 0s 88us/sample - loss: -104310241.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1161/5000
    500/500 [==============================] - 0s 84us/sample - loss: 42250376.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1162/5000
    500/500 [==============================] - 0s 82us/sample - loss: 18602861.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1163/5000
    500/500 [==============================] - 0s 82us/sample - loss: 181484355.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1164/5000
    500/500 [==============================] - 0s 86us/sample - loss: -20256721.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1165/5000
    500/500 [==============================] - 0s 78us/sample - loss: 28325794.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1166/5000
    500/500 [==============================] - 0s 82us/sample - loss: 95628557.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1167/5000
    500/500 [==============================] - 0s 86us/sample - loss: 44601401.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1168/5000
    500/500 [==============================] - 0s 92us/sample - loss: 236149690.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1169/5000
    500/500 [==============================] - 0s 78us/sample - loss: 151313322.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1170/5000
    500/500 [==============================] - 0s 73us/sample - loss: 10429106.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1171/5000
    500/500 [==============================] - 0s 80us/sample - loss: 191594549.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1172/5000
    500/500 [==============================] - 0s 84us/sample - loss: 39757815.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1173/5000
    500/500 [==============================] - 0s 82us/sample - loss: -24479345.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1174/5000
    500/500 [==============================] - 0s 84us/sample - loss: 86085567.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1175/5000
    500/500 [==============================] - 0s 90us/sample - loss: 61762930.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1176/5000
    500/500 [==============================] - 0s 80us/sample - loss: 7846215.1020 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1177/5000
    500/500 [==============================] - 0s 78us/sample - loss: -87210010.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1178/5000
    500/500 [==============================] - 0s 80us/sample - loss: 112788954.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1179/5000
    500/500 [==============================] - 0s 86us/sample - loss: -106226439.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1180/5000
    500/500 [==============================] - 0s 74us/sample - loss: -52465344.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1181/5000
    500/500 [==============================] - 0s 82us/sample - loss: -39307983.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1182/5000
    500/500 [==============================] - 0s 78us/sample - loss: -48527997.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1183/5000
    500/500 [==============================] - 0s 82us/sample - loss: -106341358.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1184/5000
    500/500 [==============================] - 0s 80us/sample - loss: -32878234.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1185/5000
    500/500 [==============================] - 0s 79us/sample - loss: 552697.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1186/5000
    500/500 [==============================] - 0s 78us/sample - loss: 53600680.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1187/5000
    500/500 [==============================] - 0s 82us/sample - loss: 94799866.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1188/5000
    500/500 [==============================] - 0s 80us/sample - loss: 23562871.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1189/5000
    500/500 [==============================] - 0s 82us/sample - loss: 61126306.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1190/5000
    500/500 [==============================] - 0s 80us/sample - loss: -23015991.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1191/5000
    500/500 [==============================] - 0s 84us/sample - loss: 165207188.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1192/5000
    500/500 [==============================] - 0s 94us/sample - loss: -102230437.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1193/5000
    500/500 [==============================] - 0s 76us/sample - loss: -43806144.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1194/5000
    500/500 [==============================] - 0s 82us/sample - loss: 95272265.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1195/5000
    500/500 [==============================] - 0s 78us/sample - loss: -95440872.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1196/5000
    500/500 [==============================] - 0s 88us/sample - loss: -55900450.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1197/5000
    500/500 [==============================] - 0s 76us/sample - loss: 41303026.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1198/5000
    500/500 [==============================] - 0s 73us/sample - loss: -183970581.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1199/5000
    500/500 [==============================] - 0s 80us/sample - loss: -89724311.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1200/5000
    500/500 [==============================] - 0s 84us/sample - loss: 35705040.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1201/5000
    500/500 [==============================] - 0s 82us/sample - loss: -136154066.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1202/5000
    500/500 [==============================] - 0s 82us/sample - loss: -78126573.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1203/5000
    500/500 [==============================] - 0s 158us/sample - loss: 42220662.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1204/5000
    500/500 [==============================] - 0s 114us/sample - loss: -88101208.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1205/5000
    500/500 [==============================] - 0s 90us/sample - loss: -16804137.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1206/5000
    500/500 [==============================] - 0s 84us/sample - loss: -3332035.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1207/5000
    500/500 [==============================] - 0s 86us/sample - loss: 130250283.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1208/5000
    500/500 [==============================] - 0s 76us/sample - loss: 26222432.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1209/5000
    500/500 [==============================] - 0s 90us/sample - loss: 125431535.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1210/5000
    500/500 [==============================] - 0s 78us/sample - loss: 93550324.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1211/5000
    500/500 [==============================] - 0s 82us/sample - loss: -41159269.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1212/5000
    500/500 [==============================] - 0s 78us/sample - loss: 131989867.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1213/5000
    500/500 [==============================] - 0s 94us/sample - loss: -236664173.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1214/5000
    500/500 [==============================] - 0s 78us/sample - loss: 29374052.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1215/5000
    500/500 [==============================] - 0s 73us/sample - loss: 175230217.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1216/5000
    500/500 [==============================] - 0s 80us/sample - loss: -160818180.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1217/5000
    500/500 [==============================] - 0s 82us/sample - loss: -29253390.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1218/5000
    500/500 [==============================] - 0s 82us/sample - loss: 50804352.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1219/5000
    500/500 [==============================] - 0s 84us/sample - loss: 5198343.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1220/5000
    500/500 [==============================] - 0s 92us/sample - loss: -35209963.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1221/5000
    500/500 [==============================] - 0s 76us/sample - loss: -60498149.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1222/5000
    500/500 [==============================] - 0s 84us/sample - loss: -42436473.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1223/5000
    500/500 [==============================] - 0s 84us/sample - loss: 135443800.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1224/5000
    500/500 [==============================] - 0s 82us/sample - loss: 7555315.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1225/5000
    500/500 [==============================] - 0s 78us/sample - loss: 9207185.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1226/5000
    500/500 [==============================] - 0s 76us/sample - loss: -259863452.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1227/5000
    500/500 [==============================] - 0s 78us/sample - loss: 58100797.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1228/5000
    500/500 [==============================] - 0s 84us/sample - loss: -68199960.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1229/5000
    500/500 [==============================] - 0s 80us/sample - loss: 71120101.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1230/5000
    500/500 [==============================] - 0s 80us/sample - loss: 258518088.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1231/5000
    500/500 [==============================] - 0s 82us/sample - loss: -62562329.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1232/5000
    500/500 [==============================] - 0s 75us/sample - loss: -14190268.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1233/5000
    500/500 [==============================] - 0s 82us/sample - loss: -96756394.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1234/5000
    500/500 [==============================] - 0s 84us/sample - loss: 33850360.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1235/5000
    500/500 [==============================] - 0s 82us/sample - loss: 161314099.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1236/5000
    500/500 [==============================] - 0s 76us/sample - loss: -143211038.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1237/5000
    500/500 [==============================] - 0s 80us/sample - loss: 108770286.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1238/5000
    500/500 [==============================] - 0s 84us/sample - loss: 62416605.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1239/5000
    500/500 [==============================] - 0s 78us/sample - loss: 52115686.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1240/5000
    500/500 [==============================] - 0s 82us/sample - loss: 14509904.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1241/5000
    500/500 [==============================] - 0s 78us/sample - loss: 199165852.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1242/5000
    500/500 [==============================] - 0s 80us/sample - loss: 17211325.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1243/5000
    500/500 [==============================] - 0s 78us/sample - loss: -40319091.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1244/5000
    500/500 [==============================] - 0s 80us/sample - loss: 88303296.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1245/5000
    500/500 [==============================] - 0s 80us/sample - loss: -101780514.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1246/5000
    500/500 [==============================] - 0s 82us/sample - loss: -156027801.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1247/5000
    500/500 [==============================] - 0s 82us/sample - loss: 127171734.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1248/5000
    500/500 [==============================] - 0s 82us/sample - loss: -9743406.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1249/5000
    500/500 [==============================] - 0s 82us/sample - loss: -77146366.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1250/5000
    500/500 [==============================] - 0s 82us/sample - loss: 165099680.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1251/5000
    500/500 [==============================] - 0s 88us/sample - loss: -115280383.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1252/5000
    500/500 [==============================] - 0s 82us/sample - loss: 70184962.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1253/5000
    500/500 [==============================] - 0s 86us/sample - loss: -117280373.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1254/5000
    500/500 [==============================] - 0s 92us/sample - loss: 166325063.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1255/5000
    500/500 [==============================] - 0s 94us/sample - loss: -77303813.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1256/5000
    500/500 [==============================] - 0s 92us/sample - loss: -37605085.0520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1257/5000
    500/500 [==============================] - 0s 78us/sample - loss: 2017102.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1258/5000
    500/500 [==============================] - 0s 92us/sample - loss: 151420193.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1259/5000
    500/500 [==============================] - 0s 86us/sample - loss: 76543181.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1260/5000
    500/500 [==============================] - 0s 84us/sample - loss: 64295670.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1261/5000
    500/500 [==============================] - 0s 84us/sample - loss: 16465432.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1262/5000
    500/500 [==============================] - 0s 92us/sample - loss: 90532423.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1263/5000
    500/500 [==============================] - 0s 80us/sample - loss: -25350778.8820 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1264/5000
    500/500 [==============================] - 0s 92us/sample - loss: -13434937.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1265/5000
    500/500 [==============================] - 0s 84us/sample - loss: 137801714.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1266/5000
    500/500 [==============================] - 0s 82us/sample - loss: -113974561.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1267/5000
    500/500 [==============================] - 0s 84us/sample - loss: 82190708.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1268/5000
    500/500 [==============================] - 0s 78us/sample - loss: -17627854.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1269/5000
    500/500 [==============================] - 0s 86us/sample - loss: -14654124.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1270/5000
    500/500 [==============================] - 0s 82us/sample - loss: -225569147.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1271/5000
    500/500 [==============================] - 0s 78us/sample - loss: -47648785.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1272/5000
    500/500 [==============================] - 0s 78us/sample - loss: 588492.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1273/5000
    500/500 [==============================] - 0s 82us/sample - loss: -32942960.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1274/5000
    500/500 [==============================] - 0s 82us/sample - loss: 79517472.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1275/5000
    500/500 [==============================] - 0s 80us/sample - loss: -37710419.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1276/5000
    500/500 [==============================] - 0s 82us/sample - loss: -53916748.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1277/5000
    500/500 [==============================] - 0s 84us/sample - loss: -129194549.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1278/5000
    500/500 [==============================] - 0s 84us/sample - loss: -28288450.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1279/5000
    500/500 [==============================] - 0s 80us/sample - loss: 39070779.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1280/5000
    500/500 [==============================] - 0s 76us/sample - loss: -123327107.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1281/5000
    500/500 [==============================] - 0s 78us/sample - loss: -133071569.2180 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1282/5000
    500/500 [==============================] - 0s 82us/sample - loss: 130791974.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1283/5000
    500/500 [==============================] - 0s 78us/sample - loss: 2659044.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1284/5000
    500/500 [==============================] - 0s 84us/sample - loss: 6539308.1925 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1285/5000
    500/500 [==============================] - 0s 82us/sample - loss: 35371403.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1286/5000
    500/500 [==============================] - 0s 78us/sample - loss: 13053575.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1287/5000
    500/500 [==============================] - 0s 80us/sample - loss: -146706203.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1288/5000
    500/500 [==============================] - 0s 82us/sample - loss: 105128419.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1289/5000
    500/500 [==============================] - 0s 80us/sample - loss: 82636885.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1290/5000
    500/500 [==============================] - 0s 76us/sample - loss: -47448260.6070 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1291/5000
    500/500 [==============================] - 0s 80us/sample - loss: 143022306.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1292/5000
    500/500 [==============================] - 0s 84us/sample - loss: 21226554.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1293/5000
    500/500 [==============================] - 0s 80us/sample - loss: -19934206.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1294/5000
    500/500 [==============================] - 0s 76us/sample - loss: -158633171.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1295/5000
    500/500 [==============================] - 0s 80us/sample - loss: 10930271.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1296/5000
    500/500 [==============================] - 0s 84us/sample - loss: 105438853.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1297/5000
    500/500 [==============================] - 0s 82us/sample - loss: -67534976.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1298/5000
    500/500 [==============================] - 0s 86us/sample - loss: 23648489.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1299/5000
    500/500 [==============================] - 0s 82us/sample - loss: -10522391.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1300/5000
    500/500 [==============================] - 0s 80us/sample - loss: -186256467.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1301/5000
    500/500 [==============================] - 0s 88us/sample - loss: 155977879.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1302/5000
    500/500 [==============================] - 0s 94us/sample - loss: 161824001.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1303/5000
    500/500 [==============================] - 0s 76us/sample - loss: 14329444.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1304/5000
    500/500 [==============================] - 0s 82us/sample - loss: -21795148.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1305/5000
    500/500 [==============================] - 0s 86us/sample - loss: 63350748.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1306/5000
    500/500 [==============================] - 0s 82us/sample - loss: 17418622.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1307/5000
    500/500 [==============================] - 0s 82us/sample - loss: 123470458.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1308/5000
    500/500 [==============================] - 0s 80us/sample - loss: 6678150.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1309/5000
    500/500 [==============================] - 0s 84us/sample - loss: 52433813.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1310/5000
    500/500 [==============================] - 0s 80us/sample - loss: -208160203.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1311/5000
    500/500 [==============================] - 0s 90us/sample - loss: 113231882.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1312/5000
    500/500 [==============================] - 0s 88us/sample - loss: 116961809.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1313/5000
    500/500 [==============================] - 0s 92us/sample - loss: 5540947.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1314/5000
    500/500 [==============================] - 0s 78us/sample - loss: -114020266.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1315/5000
    500/500 [==============================] - 0s 82us/sample - loss: -68040955.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1316/5000
    500/500 [==============================] - 0s 80us/sample - loss: 48736242.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1317/5000
    500/500 [==============================] - 0s 90us/sample - loss: 139793438.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1318/5000
    500/500 [==============================] - 0s 80us/sample - loss: -31540793.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1319/5000
    500/500 [==============================] - 0s 80us/sample - loss: -35090360.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1320/5000
    500/500 [==============================] - 0s 80us/sample - loss: -193517152.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1321/5000
    500/500 [==============================] - 0s 82us/sample - loss: 11663229.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1322/5000
    500/500 [==============================] - 0s 78us/sample - loss: 38333260.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1323/5000
    500/500 [==============================] - 0s 82us/sample - loss: 132705333.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1324/5000
    500/500 [==============================] - 0s 84us/sample - loss: -18641304.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1325/5000
    500/500 [==============================] - 0s 78us/sample - loss: 120206613.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1326/5000
    500/500 [==============================] - 0s 82us/sample - loss: -9843273.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1327/5000
    500/500 [==============================] - 0s 80us/sample - loss: 80737552.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1328/5000
    500/500 [==============================] - 0s 82us/sample - loss: 134859951.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1329/5000
    500/500 [==============================] - 0s 78us/sample - loss: -28888024.8315 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1330/5000
    500/500 [==============================] - 0s 84us/sample - loss: 148584971.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1331/5000
    500/500 [==============================] - 0s 78us/sample - loss: -176079.6280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1332/5000
    500/500 [==============================] - 0s 82us/sample - loss: -9487843.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1333/5000
    500/500 [==============================] - 0s 80us/sample - loss: -79605186.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1334/5000
    500/500 [==============================] - 0s 84us/sample - loss: -70598016.8760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1335/5000
    500/500 [==============================] - 0s 82us/sample - loss: -17504531.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1336/5000
    500/500 [==============================] - 0s 84us/sample - loss: -89806423.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1337/5000
    500/500 [==============================] - 0s 82us/sample - loss: 93972330.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1338/5000
    500/500 [==============================] - 0s 82us/sample - loss: -79393230.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1339/5000
    500/500 [==============================] - 0s 84us/sample - loss: 20553838.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1340/5000
    500/500 [==============================] - 0s 80us/sample - loss: -5710300.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1341/5000
    500/500 [==============================] - 0s 82us/sample - loss: 5193439.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1342/5000
    500/500 [==============================] - 0s 92us/sample - loss: -146694132.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1343/5000
    500/500 [==============================] - 0s 78us/sample - loss: -9467430.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1344/5000
    500/500 [==============================] - 0s 80us/sample - loss: 56229393.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1345/5000
    500/500 [==============================] - 0s 82us/sample - loss: -900054.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1346/5000
    500/500 [==============================] - 0s 88us/sample - loss: -103092530.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1347/5000
    500/500 [==============================] - 0s 78us/sample - loss: 195104929.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1348/5000
    500/500 [==============================] - 0s 73us/sample - loss: 9138088.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1349/5000
    500/500 [==============================] - 0s 80us/sample - loss: 66863651.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1350/5000
    500/500 [==============================] - 0s 82us/sample - loss: -8754271.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1351/5000
    500/500 [==============================] - 0s 82us/sample - loss: -4309128.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1352/5000
    500/500 [==============================] - 0s 84us/sample - loss: -97010869.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1353/5000
    500/500 [==============================] - 0s 92us/sample - loss: 76313083.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1354/5000
    500/500 [==============================] - 0s 76us/sample - loss: -39012371.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1355/5000
    500/500 [==============================] - 0s 84us/sample - loss: 56915110.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1356/5000
    500/500 [==============================] - 0s 82us/sample - loss: -22844853.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1357/5000
    500/500 [==============================] - 0s 86us/sample - loss: -152182670.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1358/5000
    500/500 [==============================] - 0s 88us/sample - loss: 65357626.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1359/5000
    500/500 [==============================] - 0s 94us/sample - loss: 143422387.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1360/5000
    500/500 [==============================] - 0s 78us/sample - loss: 20717256.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1361/5000
    500/500 [==============================] - 0s 74us/sample - loss: -22019048.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1362/5000
    500/500 [==============================] - 0s 96us/sample - loss: -476695.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1363/5000
    500/500 [==============================] - 0s 80us/sample - loss: -82610290.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1364/5000
    500/500 [==============================] - 0s 82us/sample - loss: -174262529.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1365/5000
    500/500 [==============================] - 0s 78us/sample - loss: -145196744.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1366/5000
    500/500 [==============================] - 0s 82us/sample - loss: -1923683.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1367/5000
    500/500 [==============================] - 0s 80us/sample - loss: 39302860.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1368/5000
    500/500 [==============================] - 0s 84us/sample - loss: 24848870.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1369/5000
    500/500 [==============================] - 0s 82us/sample - loss: 56955533.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1370/5000
    500/500 [==============================] - 0s 77us/sample - loss: -61714971.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1371/5000
    500/500 [==============================] - 0s 80us/sample - loss: -61776208.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1372/5000
    500/500 [==============================] - 0s 82us/sample - loss: -4330894.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1373/5000
    500/500 [==============================] - 0s 80us/sample - loss: -97882951.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1374/5000
    500/500 [==============================] - 0s 84us/sample - loss: -11828223.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1375/5000
    500/500 [==============================] - 0s 82us/sample - loss: -122091025.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1376/5000
    500/500 [==============================] - 0s 73us/sample - loss: 115981938.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1377/5000
    500/500 [==============================] - 0s 82us/sample - loss: -134672786.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1378/5000
    500/500 [==============================] - 0s 82us/sample - loss: 86313594.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1379/5000
    500/500 [==============================] - 0s 78us/sample - loss: -16669104.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1380/5000
    500/500 [==============================] - 0s 86us/sample - loss: 135202107.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1381/5000
    500/500 [==============================] - 0s 82us/sample - loss: -130213356.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1382/5000
    500/500 [==============================] - 0s 86us/sample - loss: -155766826.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1383/5000
    500/500 [==============================] - 0s 80us/sample - loss: -19753664.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1384/5000
    500/500 [==============================] - 0s 100us/sample - loss: -37600876.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1385/5000
    500/500 [==============================] - 0s 88us/sample - loss: -66507908.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1386/5000
    500/500 [==============================] - 0s 100us/sample - loss: 23358828.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1387/5000
    500/500 [==============================] - 0s 98us/sample - loss: -135959231.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1388/5000
    500/500 [==============================] - 0s 96us/sample - loss: -78097543.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1389/5000
    500/500 [==============================] - 0s 90us/sample - loss: -61433600.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1390/5000
    500/500 [==============================] - 0s 95us/sample - loss: 88014760.6250 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1391/5000
    500/500 [==============================] - 0s 97us/sample - loss: 48785099.8600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1392/5000
    500/500 [==============================] - 0s 96us/sample - loss: -77570872.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1393/5000
    500/500 [==============================] - 0s 88us/sample - loss: -91233221.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1394/5000
    500/500 [==============================] - 0s 92us/sample - loss: 152548064.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1395/5000
    500/500 [==============================] - 0s 106us/sample - loss: -174101515.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1396/5000
    500/500 [==============================] - 0s 106us/sample - loss: -133969965.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1397/5000
    500/500 [==============================] - 0s 94us/sample - loss: 41975755.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1398/5000
    500/500 [==============================] - 0s 97us/sample - loss: 72885977.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1399/5000
    500/500 [==============================] - 0s 90us/sample - loss: -48639204.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1400/5000
    500/500 [==============================] - 0s 96us/sample - loss: -53444465.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1401/5000
    500/500 [==============================] - 0s 90us/sample - loss: 95415469.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1402/5000
    500/500 [==============================] - 0s 90us/sample - loss: -93245493.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1403/5000
    500/500 [==============================] - 0s 112us/sample - loss: 93428027.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1404/5000
    500/500 [==============================] - 0s 104us/sample - loss: 55862558.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1405/5000
    500/500 [==============================] - 0s 100us/sample - loss: -244828370.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1406/5000
    500/500 [==============================] - 0s 98us/sample - loss: 92062304.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1407/5000
    500/500 [==============================] - 0s 112us/sample - loss: -57320923.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1408/5000
    500/500 [==============================] - 0s 99us/sample - loss: 36031897.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1409/5000
    500/500 [==============================] - 0s 88us/sample - loss: -139220702.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1410/5000
    500/500 [==============================] - 0s 94us/sample - loss: 13702234.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1411/5000
    500/500 [==============================] - 0s 94us/sample - loss: -84702008.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1412/5000
    500/500 [==============================] - 0s 100us/sample - loss: 114254562.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1413/5000
    500/500 [==============================] - 0s 92us/sample - loss: 204590836.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1414/5000
    500/500 [==============================] - 0s 88us/sample - loss: 86806279.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1415/5000
    500/500 [==============================] - 0s 100us/sample - loss: -37671602.8340 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1416/5000
    500/500 [==============================] - 0s 100us/sample - loss: -131345996.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1417/5000
    500/500 [==============================] - 0s 89us/sample - loss: -66438013.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1418/5000
    500/500 [==============================] - 0s 104us/sample - loss: -7968775.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1419/5000
    500/500 [==============================] - 0s 98us/sample - loss: 46340735.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1420/5000
    500/500 [==============================] - 0s 91us/sample - loss: 55837024.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1421/5000
    500/500 [==============================] - 0s 104us/sample - loss: -39660995.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1422/5000
    500/500 [==============================] - 0s 100us/sample - loss: -88265386.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1423/5000
    500/500 [==============================] - 0s 91us/sample - loss: 109214722.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1424/5000
    500/500 [==============================] - 0s 88us/sample - loss: 114129056.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1425/5000
    500/500 [==============================] - 0s 88us/sample - loss: -38318835.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1426/5000
    500/500 [==============================] - 0s 90us/sample - loss: -54136167.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1427/5000
    500/500 [==============================] - 0s 76us/sample - loss: 49878129.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1428/5000
    500/500 [==============================] - 0s 106us/sample - loss: -79210951.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1429/5000
    500/500 [==============================] - 0s 84us/sample - loss: -87322836.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1430/5000
    500/500 [==============================] - 0s 92us/sample - loss: -5208307.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1431/5000
    500/500 [==============================] - 0s 78us/sample - loss: -103068700.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1432/5000
    500/500 [==============================] - 0s 80us/sample - loss: -107230333.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1433/5000
    500/500 [==============================] - 0s 84us/sample - loss: -60854988.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1434/5000
    500/500 [==============================] - 0s 74us/sample - loss: -53303308.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1435/5000
    500/500 [==============================] - 0s 80us/sample - loss: -71932465.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1436/5000
    500/500 [==============================] - 0s 84us/sample - loss: -85276486.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1437/5000
    500/500 [==============================] - 0s 80us/sample - loss: -7649318.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1438/5000
    500/500 [==============================] - 0s 88us/sample - loss: 1793488.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1439/5000
    500/500 [==============================] - 0s 86us/sample - loss: 42645516.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1440/5000
    500/500 [==============================] - 0s 82us/sample - loss: -14122617.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1441/5000
    500/500 [==============================] - 0s 90us/sample - loss: -14782382.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1442/5000
    500/500 [==============================] - 0s 94us/sample - loss: 140546613.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1443/5000
    500/500 [==============================] - 0s 86us/sample - loss: -61429529.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1444/5000
    500/500 [==============================] - 0s 82us/sample - loss: -54744376.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1445/5000
    500/500 [==============================] - 0s 84us/sample - loss: 46681493.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1446/5000
    500/500 [==============================] - 0s 92us/sample - loss: 35557475.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1447/5000
    500/500 [==============================] - 0s 78us/sample - loss: -73118699.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1448/5000
    500/500 [==============================] - 0s 75us/sample - loss: -71317665.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1449/5000
    500/500 [==============================] - 0s 80us/sample - loss: -111331818.0300 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1450/5000
    500/500 [==============================] - 0s 86us/sample - loss: 164131880.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1451/5000
    500/500 [==============================] - 0s 84us/sample - loss: 75436532.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1452/5000
    500/500 [==============================] - 0s 80us/sample - loss: 76168567.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1453/5000
    500/500 [==============================] - 0s 86us/sample - loss: -109979858.6840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1454/5000
    500/500 [==============================] - 0s 60us/sample - loss: 98318817.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1455/5000
    500/500 [==============================] - 0s 116us/sample - loss: 124913083.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1456/5000
    500/500 [==============================] - 0s 86us/sample - loss: -51018468.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1457/5000
    500/500 [==============================] - 0s 82us/sample - loss: -144916825.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1458/5000
    500/500 [==============================] - 0s 86us/sample - loss: 38941147.8390 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1459/5000
    500/500 [==============================] - 0s 90us/sample - loss: 57266475.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1460/5000
    500/500 [==============================] - 0s 76us/sample - loss: -22403573.8920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1461/5000
    500/500 [==============================] - 0s 80us/sample - loss: 25439792.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1462/5000
    500/500 [==============================] - 0s 80us/sample - loss: 160068404.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1463/5000
    500/500 [==============================] - 0s 80us/sample - loss: -124363625.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1464/5000
    500/500 [==============================] - 0s 82us/sample - loss: -8850755.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1465/5000
    500/500 [==============================] - 0s 90us/sample - loss: 67677687.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1466/5000
    500/500 [==============================] - 0s 76us/sample - loss: -180063804.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1467/5000
    500/500 [==============================] - 0s 75us/sample - loss: 137965397.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1468/5000
    500/500 [==============================] - 0s 80us/sample - loss: -164198597.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1469/5000
    500/500 [==============================] - 0s 75us/sample - loss: 53430242.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1470/5000
    500/500 [==============================] - 0s 80us/sample - loss: -33359089.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1471/5000
    500/500 [==============================] - 0s 84us/sample - loss: 12615283.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1472/5000
    500/500 [==============================] - 0s 80us/sample - loss: 36264242.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1473/5000
    500/500 [==============================] - 0s 82us/sample - loss: 104346931.1980 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1474/5000
    500/500 [==============================] - 0s 82us/sample - loss: 204708908.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1475/5000
    500/500 [==============================] - 0s 80us/sample - loss: -6460042.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1476/5000
    500/500 [==============================] - 0s 80us/sample - loss: -43770483.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1477/5000
    500/500 [==============================] - 0s 80us/sample - loss: 45805881.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1478/5000
    500/500 [==============================] - 0s 86us/sample - loss: -99797646.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1479/5000
    500/500 [==============================] - 0s 84us/sample - loss: -155396735.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1480/5000
    500/500 [==============================] - 0s 76us/sample - loss: 78328022.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1481/5000
    500/500 [==============================] - 0s 80us/sample - loss: 144655972.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1482/5000
    500/500 [==============================] - 0s 82us/sample - loss: -228687920.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1483/5000
    500/500 [==============================] - 0s 80us/sample - loss: 11625335.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1484/5000
    500/500 [==============================] - 0s 82us/sample - loss: 98987323.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1485/5000
    500/500 [==============================] - 0s 82us/sample - loss: -23423547.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1486/5000
    500/500 [==============================] - 0s 84us/sample - loss: 110113724.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1487/5000
    500/500 [==============================] - 0s 82us/sample - loss: 79433510.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1488/5000
    500/500 [==============================] - 0s 86us/sample - loss: 156553207.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1489/5000
    500/500 [==============================] - 0s 92us/sample - loss: 48820116.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1490/5000
    500/500 [==============================] - 0s 88us/sample - loss: -132526533.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1491/5000
    500/500 [==============================] - 0s 92us/sample - loss: 19133706.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1492/5000
    500/500 [==============================] - 0s 80us/sample - loss: 101692163.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1493/5000
    500/500 [==============================] - 0s 90us/sample - loss: -11386891.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1494/5000
    500/500 [==============================] - 0s 76us/sample - loss: -109501799.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1495/5000
    500/500 [==============================] - 0s 78us/sample - loss: -27779717.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1496/5000
    500/500 [==============================] - 0s 82us/sample - loss: -75088473.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1497/5000
    500/500 [==============================] - 0s 84us/sample - loss: 9523420.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1498/5000
    500/500 [==============================] - 0s 82us/sample - loss: -149977642.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1499/5000
    500/500 [==============================] - 0s 84us/sample - loss: 80673867.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1500/5000
    500/500 [==============================] - 0s 84us/sample - loss: 30050852.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1501/5000
    500/500 [==============================] - 0s 84us/sample - loss: 46950470.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1502/5000
    500/500 [==============================] - 0s 82us/sample - loss: -41470222.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1503/5000
    500/500 [==============================] - 0s 98us/sample - loss: -103303215.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1504/5000
    500/500 [==============================] - 0s 76us/sample - loss: -267871506.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1505/5000
    500/500 [==============================] - 0s 75us/sample - loss: 203275859.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1506/5000
    500/500 [==============================] - 0s 80us/sample - loss: -29096383.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1507/5000
    500/500 [==============================] - 0s 84us/sample - loss: -138830865.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1508/5000
    500/500 [==============================] - 0s 78us/sample - loss: 133962744.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1509/5000
    500/500 [==============================] - 0s 82us/sample - loss: 97288616.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1510/5000
    500/500 [==============================] - 0s 80us/sample - loss: -35475777.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1511/5000
    500/500 [==============================] - 0s 82us/sample - loss: 140167271.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1512/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32093845.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1513/5000
    500/500 [==============================] - 0s 80us/sample - loss: -23936603.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1514/5000
    500/500 [==============================] - 0s 80us/sample - loss: -49827694.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1515/5000
    500/500 [==============================] - 0s 84us/sample - loss: -142211090.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1516/5000
    500/500 [==============================] - 0s 78us/sample - loss: 5405042.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1517/5000
    500/500 [==============================] - 0s 84us/sample - loss: -102171443.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1518/5000
    500/500 [==============================] - 0s 82us/sample - loss: 70554748.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1519/5000
    500/500 [==============================] - 0s 82us/sample - loss: 60326571.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1520/5000
    500/500 [==============================] - 0s 82us/sample - loss: -97844224.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1521/5000
    500/500 [==============================] - 0s 84us/sample - loss: 6380758.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1522/5000
    500/500 [==============================] - 0s 86us/sample - loss: 76191727.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1523/5000
    500/500 [==============================] - 0s 80us/sample - loss: 164748517.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1524/5000
    500/500 [==============================] - 0s 86us/sample - loss: -159706296.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1525/5000
    500/500 [==============================] - 0s 92us/sample - loss: -42410318.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1526/5000
    500/500 [==============================] - 0s 80us/sample - loss: 93323121.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1527/5000
    500/500 [==============================] - 0s 72us/sample - loss: 33580636.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1528/5000
    500/500 [==============================] - 0s 82us/sample - loss: -125276231.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1529/5000
    500/500 [==============================] - 0s 84us/sample - loss: 41589604.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1530/5000
    500/500 [==============================] - 0s 80us/sample - loss: 132620140.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1531/5000
    500/500 [==============================] - 0s 84us/sample - loss: -57545824.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1532/5000
    500/500 [==============================] - 0s 80us/sample - loss: -25644334.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1533/5000
    500/500 [==============================] - 0s 84us/sample - loss: 42977723.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1534/5000
    500/500 [==============================] - 0s 82us/sample - loss: 34904908.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1535/5000
    500/500 [==============================] - 0s 80us/sample - loss: -184047332.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1536/5000
    500/500 [==============================] - 0s 84us/sample - loss: 7002087.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1537/5000
    500/500 [==============================] - 0s 86us/sample - loss: -120625405.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1538/5000
    500/500 [==============================] - 0s 76us/sample - loss: 152103871.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1539/5000
    500/500 [==============================] - 0s 84us/sample - loss: -29291398.8520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1540/5000
    500/500 [==============================] - 0s 80us/sample - loss: -72029210.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1541/5000
    500/500 [==============================] - 0s 88us/sample - loss: -79171960.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1542/5000
    500/500 [==============================] - 0s 80us/sample - loss: 179285634.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1543/5000
    500/500 [==============================] - 0s 75us/sample - loss: 15124510.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1544/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32072955.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1545/5000
    500/500 [==============================] - 0s 84us/sample - loss: -80268567.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1546/5000
    500/500 [==============================] - 0s 84us/sample - loss: 36645350.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1547/5000
    500/500 [==============================] - 0s 88us/sample - loss: -145481365.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1548/5000
    500/500 [==============================] - 0s 94us/sample - loss: 51911811.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1549/5000
    500/500 [==============================] - 0s 78us/sample - loss: 4754194.3380 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1550/5000
    500/500 [==============================] - 0s 75us/sample - loss: 15356656.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1551/5000
    500/500 [==============================] - 0s 80us/sample - loss: -49519290.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1552/5000
    500/500 [==============================] - 0s 94us/sample - loss: -40956186.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1553/5000
    500/500 [==============================] - 0s 84us/sample - loss: -6107953.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1554/5000
    500/500 [==============================] - 0s 92us/sample - loss: 93773777.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1555/5000
    500/500 [==============================] - 0s 80us/sample - loss: 23729036.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1556/5000
    500/500 [==============================] - 0s 90us/sample - loss: 41374946.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1557/5000
    500/500 [==============================] - 0s 76us/sample - loss: -145331040.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1558/5000
    500/500 [==============================] - 0s 76us/sample - loss: 46335769.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1559/5000
    500/500 [==============================] - 0s 80us/sample - loss: 161637665.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1560/5000
    500/500 [==============================] - 0s 75us/sample - loss: -33807360.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1561/5000
    500/500 [==============================] - 0s 80us/sample - loss: -53918554.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1562/5000
    500/500 [==============================] - 0s 84us/sample - loss: -45058046.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1563/5000
    500/500 [==============================] - 0s 82us/sample - loss: -42518575.6440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1564/5000
    500/500 [==============================] - 0s 82us/sample - loss: -138211599.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1565/5000
    500/500 [==============================] - 0s 80us/sample - loss: 2240107.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1566/5000
    500/500 [==============================] - 0s 84us/sample - loss: 42715581.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1567/5000
    500/500 [==============================] - 0s 80us/sample - loss: -74139440.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1568/5000
    500/500 [==============================] - 0s 82us/sample - loss: -20744875.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1569/5000
    500/500 [==============================] - 0s 80us/sample - loss: -11687952.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1570/5000
    500/500 [==============================] - 0s 82us/sample - loss: 152604826.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1571/5000
    500/500 [==============================] - 0s 90us/sample - loss: 43042966.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1572/5000
    500/500 [==============================] - 0s 80us/sample - loss: -130449649.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1573/5000
    500/500 [==============================] - 0s 80us/sample - loss: -67805096.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1574/5000
    500/500 [==============================] - 0s 82us/sample - loss: -79694133.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1575/5000
    500/500 [==============================] - 0s 86us/sample - loss: 14830625.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1576/5000
    500/500 [==============================] - 0s 84us/sample - loss: 73259527.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1577/5000
    500/500 [==============================] - 0s 80us/sample - loss: 210036520.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1578/5000
    500/500 [==============================] - 0s 86us/sample - loss: -16384137.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1579/5000
    500/500 [==============================] - 0s 82us/sample - loss: 162266822.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1580/5000
    500/500 [==============================] - 0s 90us/sample - loss: -23756705.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1581/5000
    500/500 [==============================] - 0s 86us/sample - loss: 10611141.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1582/5000
    500/500 [==============================] - 0s 84us/sample - loss: 85245605.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1583/5000
    500/500 [==============================] - 0s 74us/sample - loss: 106944574.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1584/5000
    500/500 [==============================] - 0s 94us/sample - loss: 58594471.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1585/5000
    500/500 [==============================] - 0s 78us/sample - loss: -140003255.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1586/5000
    500/500 [==============================] - 0s 84us/sample - loss: -5693794.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1587/5000
    500/500 [==============================] - 0s 82us/sample - loss: 44813743.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1588/5000
    500/500 [==============================] - 0s 82us/sample - loss: 85374875.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1589/5000
    500/500 [==============================] - 0s 92us/sample - loss: 125018010.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1590/5000
    500/500 [==============================] - 0s 76us/sample - loss: 67598648.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1591/5000
    500/500 [==============================] - 0s 84us/sample - loss: -9335374.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1592/5000
    500/500 [==============================] - ETA: 0s - loss: -940167360.0000 - accuracy: 0.0000e+0 - 0s 80us/sample - loss: -75449587.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1593/5000
    500/500 [==============================] - 0s 84us/sample - loss: 6549376.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1594/5000
    500/500 [==============================] - 0s 94us/sample - loss: 21665537.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1595/5000
    500/500 [==============================] - 0s 76us/sample - loss: 11127665.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1596/5000
    500/500 [==============================] - 0s 79us/sample - loss: 84458016.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1597/5000
    500/500 [==============================] - 0s 88us/sample - loss: 29069133.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1598/5000
    500/500 [==============================] - 0s 82us/sample - loss: 92633825.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1599/5000
    500/500 [==============================] - 0s 82us/sample - loss: -16885982.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1600/5000
    500/500 [==============================] - 0s 80us/sample - loss: -24548742.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1601/5000
    500/500 [==============================] - 0s 78us/sample - loss: 56201636.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1602/5000
    500/500 [==============================] - 0s 82us/sample - loss: 58204106.9780 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1603/5000
    500/500 [==============================] - 0s 84us/sample - loss: 133072310.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1604/5000
    500/500 [==============================] - 0s 78us/sample - loss: -115862030.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1605/5000
    500/500 [==============================] - 0s 86us/sample - loss: -25285488.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1606/5000
    500/500 [==============================] - 0s 86us/sample - loss: -22191741.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1607/5000
    500/500 [==============================] - 0s 74us/sample - loss: 126489166.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1608/5000
    500/500 [==============================] - 0s 78us/sample - loss: 134523117.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1609/5000
    500/500 [==============================] - 0s 80us/sample - loss: 17862217.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1610/5000
    500/500 [==============================] - 0s 80us/sample - loss: -71793675.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1611/5000
    500/500 [==============================] - 0s 82us/sample - loss: 23151189.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1612/5000
    500/500 [==============================] - 0s 78us/sample - loss: 44069460.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1613/5000
    500/500 [==============================] - 0s 82us/sample - loss: -165856584.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1614/5000
    500/500 [==============================] - 0s 80us/sample - loss: 71761402.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1615/5000
    500/500 [==============================] - 0s 80us/sample - loss: 82724235.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1616/5000
    500/500 [==============================] - 0s 82us/sample - loss: 105976407.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1617/5000
    500/500 [==============================] - 0s 76us/sample - loss: 138069183.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1618/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32179668.4481 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1619/5000
    500/500 [==============================] - 0s 86us/sample - loss: 77536580.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1620/5000
    500/500 [==============================] - 0s 82us/sample - loss: 14255918.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1621/5000
    500/500 [==============================] - 0s 82us/sample - loss: -261277954.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1622/5000
    500/500 [==============================] - 0s 86us/sample - loss: -180197744.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1623/5000
    500/500 [==============================] - 0s 82us/sample - loss: 52294770.7000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1624/5000
    500/500 [==============================] - 0s 84us/sample - loss: -86192484.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1625/5000
    500/500 [==============================] - 0s 82us/sample - loss: -30639301.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1626/5000
    500/500 [==============================] - 0s 82us/sample - loss: 104588766.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1627/5000
    500/500 [==============================] - 0s 84us/sample - loss: -53069032.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1628/5000
    500/500 [==============================] - 0s 86us/sample - loss: -74756399.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1629/5000
    500/500 [==============================] - 0s 86us/sample - loss: -11395864.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1630/5000
    500/500 [==============================] - 0s 94us/sample - loss: -137286434.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1631/5000
    500/500 [==============================] - 0s 78us/sample - loss: 1834017.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1632/5000
    500/500 [==============================] - 0s 82us/sample - loss: -22082036.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1633/5000
    500/500 [==============================] - 0s 80us/sample - loss: -56062574.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1634/5000
    500/500 [==============================] - 0s 86us/sample - loss: 108612621.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1635/5000
    500/500 [==============================] - 0s 92us/sample - loss: 136108049.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1636/5000
    500/500 [==============================] - 0s 78us/sample - loss: 205073093.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1637/5000
    500/500 [==============================] - 0s 82us/sample - loss: 109354876.9400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1638/5000
    500/500 [==============================] - 0s 82us/sample - loss: -30038464.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1639/5000
    500/500 [==============================] - 0s 86us/sample - loss: -83683663.0360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1640/5000
    500/500 [==============================] - 0s 82us/sample - loss: 3926377.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1641/5000
    500/500 [==============================] - 0s 82us/sample - loss: 14214803.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1642/5000
    500/500 [==============================] - 0s 84us/sample - loss: -150597067.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1643/5000
    500/500 [==============================] - 0s 84us/sample - loss: 69742336.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1644/5000
    500/500 [==============================] - 0s 84us/sample - loss: -113681277.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1645/5000
    500/500 [==============================] - 0s 84us/sample - loss: -73311311.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1646/5000
    500/500 [==============================] - 0s 80us/sample - loss: 216522512.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1647/5000
    500/500 [==============================] - 0s 80us/sample - loss: 109128976.8340 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1648/5000
    500/500 [==============================] - 0s 80us/sample - loss: -77982494.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1649/5000
    500/500 [==============================] - 0s 82us/sample - loss: -125096691.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1650/5000
    500/500 [==============================] - 0s 78us/sample - loss: 38537392.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1651/5000
    500/500 [==============================] - 0s 84us/sample - loss: 50632007.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1652/5000
    500/500 [==============================] - 0s 78us/sample - loss: -37776232.0010 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1653/5000
    500/500 [==============================] - 0s 82us/sample - loss: 35753453.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1654/5000
    500/500 [==============================] - 0s 80us/sample - loss: -11063465.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1655/5000
    500/500 [==============================] - 0s 82us/sample - loss: 172674592.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1656/5000
    500/500 [==============================] - 0s 82us/sample - loss: 140527814.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1657/5000
    500/500 [==============================] - 0s 82us/sample - loss: 69635046.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1658/5000
    500/500 [==============================] - 0s 90us/sample - loss: -75484635.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1659/5000
    500/500 [==============================] - 0s 78us/sample - loss: 169977149.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1660/5000
    500/500 [==============================] - 0s 82us/sample - loss: -101159079.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1661/5000
    500/500 [==============================] - 0s 80us/sample - loss: -7092356.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1662/5000
    500/500 [==============================] - 0s 82us/sample - loss: 158175075.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1663/5000
    500/500 [==============================] - 0s 80us/sample - loss: -25286884.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1664/5000
    500/500 [==============================] - 0s 82us/sample - loss: 33605161.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1665/5000
    500/500 [==============================] - 0s 86us/sample - loss: -155104966.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1666/5000
    500/500 [==============================] - 0s 76us/sample - loss: -57763842.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1667/5000
    500/500 [==============================] - 0s 82us/sample - loss: 53582003.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1668/5000
    500/500 [==============================] - 0s 80us/sample - loss: 52208445.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1669/5000
    500/500 [==============================] - 0s 90us/sample - loss: 37850415.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1670/5000
    500/500 [==============================] - 0s 78us/sample - loss: -26564588.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1671/5000
    500/500 [==============================] - 0s 70us/sample - loss: -14752004.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1672/5000
    500/500 [==============================] - 0s 82us/sample - loss: -13396684.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1673/5000
    500/500 [==============================] - 0s 86us/sample - loss: -148664509.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1674/5000
    500/500 [==============================] - 0s 84us/sample - loss: -89301010.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1675/5000
    500/500 [==============================] - 0s 84us/sample - loss: -62153911.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1676/5000
    500/500 [==============================] - 0s 82us/sample - loss: -58817691.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1677/5000
    500/500 [==============================] - 0s 84us/sample - loss: -70399607.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1678/5000
    500/500 [==============================] - 0s 84us/sample - loss: -67935451.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1679/5000
    500/500 [==============================] - 0s 82us/sample - loss: -211868085.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1680/5000
    500/500 [==============================] - 0s 84us/sample - loss: 34171693.6940 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1681/5000
    500/500 [==============================] - 0s 92us/sample - loss: -76838675.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1682/5000
    500/500 [==============================] - 0s 78us/sample - loss: -30008741.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1683/5000
    500/500 [==============================] - 0s 84us/sample - loss: -231652186.5275 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1684/5000
    500/500 [==============================] - 0s 84us/sample - loss: -135159193.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1685/5000
    500/500 [==============================] - 0s 80us/sample - loss: 55435180.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1686/5000
    500/500 [==============================] - 0s 80us/sample - loss: 31102260.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1687/5000
    500/500 [==============================] - 0s 82us/sample - loss: -132470277.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1688/5000
    500/500 [==============================] - 0s 84us/sample - loss: -17686149.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1689/5000
    500/500 [==============================] - 0s 92us/sample - loss: 88399408.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1690/5000
    500/500 [==============================] - 0s 74us/sample - loss: -126906835.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1691/5000
    500/500 [==============================] - 0s 78us/sample - loss: -63122496.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1692/5000
    500/500 [==============================] - 0s 80us/sample - loss: -39338333.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1693/5000
    500/500 [==============================] - 0s 90us/sample - loss: -31600139.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1694/5000
    500/500 [==============================] - 0s 76us/sample - loss: 24226639.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1695/5000
    500/500 [==============================] - 0s 82us/sample - loss: -22632326.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1696/5000
    500/500 [==============================] - 0s 78us/sample - loss: -47051806.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1697/5000
    500/500 [==============================] - 0s 92us/sample - loss: 4057739.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1698/5000
    500/500 [==============================] - 0s 86us/sample - loss: -30645048.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1699/5000
    500/500 [==============================] - 0s 80us/sample - loss: 65100290.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1700/5000
    500/500 [==============================] - 0s 75us/sample - loss: 128729279.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1701/5000
    500/500 [==============================] - 0s 82us/sample - loss: 29414606.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1702/5000
    500/500 [==============================] - 0s 84us/sample - loss: 42311602.0470 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1703/5000
    500/500 [==============================] - 0s 78us/sample - loss: 76023824.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1704/5000
    500/500 [==============================] - 0s 84us/sample - loss: 155680246.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1705/5000
    500/500 [==============================] - 0s 92us/sample - loss: 117303225.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1706/5000
    500/500 [==============================] - 0s 78us/sample - loss: -17548674.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1707/5000
    500/500 [==============================] - 0s 84us/sample - loss: -99699373.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1708/5000
    500/500 [==============================] - 0s 82us/sample - loss: -3592315.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1709/5000
    500/500 [==============================] - 0s 77us/sample - loss: -79283639.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1710/5000
    500/500 [==============================] - 0s 80us/sample - loss: -149161496.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1711/5000
    500/500 [==============================] - 0s 82us/sample - loss: -126497402.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1712/5000
    500/500 [==============================] - 0s 80us/sample - loss: -12880151.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1713/5000
    500/500 [==============================] - 0s 88us/sample - loss: 32113026.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1714/5000
    500/500 [==============================] - 0s 84us/sample - loss: -83678310.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1715/5000
    500/500 [==============================] - 0s 82us/sample - loss: 205495428.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1716/5000
    500/500 [==============================] - 0s 94us/sample - loss: -63790895.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1717/5000
    500/500 [==============================] - 0s 88us/sample - loss: -213536980.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1718/5000
    500/500 [==============================] - 0s 94us/sample - loss: -82028921.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1719/5000
    500/500 [==============================] - 0s 76us/sample - loss: -47530584.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1720/5000
    500/500 [==============================] - 0s 92us/sample - loss: -34519815.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1721/5000
    500/500 [==============================] - 0s 76us/sample - loss: -31948705.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1722/5000
    500/500 [==============================] - 0s 72us/sample - loss: 80180959.3560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1723/5000
    500/500 [==============================] - 0s 86us/sample - loss: -142836894.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1724/5000
    500/500 [==============================] - 0s 73us/sample - loss: -31048529.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1725/5000
    500/500 [==============================] - 0s 80us/sample - loss: -148286343.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1726/5000
    500/500 [==============================] - 0s 84us/sample - loss: -79544342.0200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1727/5000
    500/500 [==============================] - 0s 84us/sample - loss: -134616550.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1728/5000
    500/500 [==============================] - 0s 86us/sample - loss: 127240205.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1729/5000
    500/500 [==============================] - ETA: 0s - loss: -332904032.0000 - accuracy: 0.0000e+0 - 0s 84us/sample - loss: 93139111.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1730/5000
    500/500 [==============================] - 0s 86us/sample - loss: 149587610.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1731/5000
    500/500 [==============================] - 0s 84us/sample - loss: 32292837.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1732/5000
    500/500 [==============================] - 0s 88us/sample - loss: 27903777.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1733/5000
    500/500 [==============================] - 0s 82us/sample - loss: 77688221.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1734/5000
    500/500 [==============================] - 0s 88us/sample - loss: -84833102.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1735/5000
    500/500 [==============================] - 0s 84us/sample - loss: 80319058.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1736/5000
    500/500 [==============================] - 0s 86us/sample - loss: 86432279.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1737/5000
    500/500 [==============================] - 0s 86us/sample - loss: -136947809.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1738/5000
    500/500 [==============================] - 0s 84us/sample - loss: 21954408.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1739/5000
    500/500 [==============================] - 0s 82us/sample - loss: 16341022.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1740/5000
    500/500 [==============================] - 0s 78us/sample - loss: 85403369.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1741/5000
    500/500 [==============================] - 0s 73us/sample - loss: -49619000.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1742/5000
    500/500 [==============================] - 0s 78us/sample - loss: -23047783.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1743/5000
    500/500 [==============================] - 0s 82us/sample - loss: 68183694.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1744/5000
    500/500 [==============================] - 0s 80us/sample - loss: -436584.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1745/5000
    500/500 [==============================] - 0s 88us/sample - loss: -139179657.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1746/5000
    500/500 [==============================] - 0s 74us/sample - loss: -22569179.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1747/5000
    500/500 [==============================] - 0s 77us/sample - loss: -34581024.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1748/5000
    500/500 [==============================] - 0s 78us/sample - loss: 44830055.1980 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1749/5000
    500/500 [==============================] - 0s 90us/sample - loss: 3531571.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1750/5000
    500/500 [==============================] - 0s 110us/sample - loss: -110507744.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1751/5000
    500/500 [==============================] - 0s 92us/sample - loss: -101465595.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1752/5000
    500/500 [==============================] - 0s 108us/sample - loss: -176433852.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1753/5000
    500/500 [==============================] - 0s 96us/sample - loss: 71503043.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1754/5000
    500/500 [==============================] - 0s 95us/sample - loss: -145695712.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1755/5000
    500/500 [==============================] - 0s 96us/sample - loss: 148784311.5180 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1756/5000
    500/500 [==============================] - 0s 96us/sample - loss: 13767488.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1757/5000
    500/500 [==============================] - 0s 84us/sample - loss: 76284305.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1758/5000
    500/500 [==============================] - 0s 108us/sample - loss: -97510017.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1759/5000
    500/500 [==============================] - 0s 94us/sample - loss: 146939505.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1760/5000
    500/500 [==============================] - 0s 114us/sample - loss: -77008241.5320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1761/5000
    500/500 [==============================] - 0s 98us/sample - loss: -119366428.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1762/5000
    500/500 [==============================] - 0s 104us/sample - loss: -90864675.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1763/5000
    500/500 [==============================] - 0s 91us/sample - loss: -6729913.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1764/5000
    500/500 [==============================] - 0s 90us/sample - loss: -37880455.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1765/5000
    500/500 [==============================] - 0s 92us/sample - loss: 30123858.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1766/5000
    500/500 [==============================] - 0s 102us/sample - loss: -71130300.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1767/5000
    500/500 [==============================] - 0s 100us/sample - loss: 100692265.3420 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1768/5000
    500/500 [==============================] - 0s 95us/sample - loss: -38790227.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1769/5000
    500/500 [==============================] - 0s 90us/sample - loss: 54400973.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1770/5000
    500/500 [==============================] - 0s 92us/sample - loss: -29612231.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1771/5000
    500/500 [==============================] - 0s 112us/sample - loss: 17974640.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1772/5000
    500/500 [==============================] - 0s 102us/sample - loss: -89485185.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1773/5000
    500/500 [==============================] - 0s 102us/sample - loss: 25066714.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1774/5000
    500/500 [==============================] - 0s 104us/sample - loss: 63029508.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1775/5000
    500/500 [==============================] - 0s 98us/sample - loss: 4517975.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1776/5000
    500/500 [==============================] - 0s 104us/sample - loss: -54236825.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1777/5000
    500/500 [==============================] - 0s 94us/sample - loss: 111358442.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1778/5000
    500/500 [==============================] - 0s 92us/sample - loss: -156489058.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1779/5000
    500/500 [==============================] - 0s 90us/sample - loss: -19459175.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1780/5000
    500/500 [==============================] - 0s 108us/sample - loss: -71624992.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1781/5000
    500/500 [==============================] - 0s 104us/sample - loss: 16217141.3160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1782/5000
    500/500 [==============================] - 0s 94us/sample - loss: 201332910.3700 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1783/5000
    500/500 [==============================] - 0s 100us/sample - loss: -106220676.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1784/5000
    500/500 [==============================] - 0s 110us/sample - loss: 88777246.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1785/5000
    500/500 [==============================] - 0s 96us/sample - loss: -39208560.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1786/5000
    500/500 [==============================] - 0s 96us/sample - loss: -22755708.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1787/5000
    500/500 [==============================] - 0s 96us/sample - loss: -73914744.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1788/5000
    500/500 [==============================] - 0s 92us/sample - loss: 47591146.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1789/5000
    500/500 [==============================] - 0s 102us/sample - loss: -34946317.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1790/5000
    500/500 [==============================] - 0s 88us/sample - loss: -46232410.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1791/5000
    500/500 [==============================] - 0s 84us/sample - loss: -25667599.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1792/5000
    500/500 [==============================] - 0s 96us/sample - loss: -285198.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1793/5000
    500/500 [==============================] - 0s 75us/sample - loss: -42864967.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1794/5000
    500/500 [==============================] - 0s 78us/sample - loss: -15791864.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1795/5000
    500/500 [==============================] - 0s 84us/sample - loss: -109710594.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1796/5000
    500/500 [==============================] - 0s 80us/sample - loss: -148522809.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1797/5000
    500/500 [==============================] - 0s 82us/sample - loss: -48769784.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1798/5000
    500/500 [==============================] - 0s 82us/sample - loss: 27500710.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1799/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32893561.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1800/5000
    500/500 [==============================] - 0s 92us/sample - loss: -7663918.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1801/5000
    500/500 [==============================] - 0s 78us/sample - loss: 115047381.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1802/5000
    500/500 [==============================] - 0s 82us/sample - loss: -4616738.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1803/5000
    500/500 [==============================] - 0s 82us/sample - loss: -99937494.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1804/5000
    500/500 [==============================] - 0s 90us/sample - loss: 146828085.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1805/5000
    500/500 [==============================] - 0s 78us/sample - loss: 16312701.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1806/5000
    500/500 [==============================] - 0s 82us/sample - loss: -130194482.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1807/5000
    500/500 [==============================] - 0s 84us/sample - loss: 152642316.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1808/5000
    500/500 [==============================] - 0s 90us/sample - loss: 22873033.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1809/5000
    500/500 [==============================] - 0s 76us/sample - loss: -9334686.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1810/5000
    500/500 [==============================] - 0s 84us/sample - loss: -60380721.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1811/5000
    500/500 [==============================] - 0s 84us/sample - loss: -48639361.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1812/5000
    500/500 [==============================] - 0s 84us/sample - loss: -90830574.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1813/5000
    500/500 [==============================] - 0s 80us/sample - loss: -184771549.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1814/5000
    500/500 [==============================] - 0s 82us/sample - loss: 16399577.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1815/5000
    500/500 [==============================] - 0s 86us/sample - loss: 41202647.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1816/5000
    500/500 [==============================] - 0s 82us/sample - loss: 67311670.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1817/5000
    500/500 [==============================] - 0s 86us/sample - loss: 25484214.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1818/5000
    500/500 [==============================] - 0s 92us/sample - loss: -36211693.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1819/5000
    500/500 [==============================] - 0s 78us/sample - loss: -23938451.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1820/5000
    500/500 [==============================] - 0s 74us/sample - loss: -6354989.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1821/5000
    500/500 [==============================] - 0s 86us/sample - loss: -9045664.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1822/5000
    500/500 [==============================] - 0s 77us/sample - loss: -61967280.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1823/5000
    500/500 [==============================] - 0s 82us/sample - loss: 163957195.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1824/5000
    500/500 [==============================] - 0s 86us/sample - loss: 160118666.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1825/5000
    500/500 [==============================] - 0s 86us/sample - loss: -6709097.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1826/5000
    500/500 [==============================] - 0s 90us/sample - loss: 148006440.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1827/5000
    500/500 [==============================] - 0s 78us/sample - loss: 138384492.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1828/5000
    500/500 [==============================] - 0s 84us/sample - loss: -75769855.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1829/5000
    500/500 [==============================] - 0s 84us/sample - loss: 63152573.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1830/5000
    500/500 [==============================] - 0s 86us/sample - loss: -101256301.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1831/5000
    500/500 [==============================] - 0s 96us/sample - loss: -42199752.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1832/5000
    500/500 [==============================] - 0s 80us/sample - loss: 8454545.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1833/5000
    500/500 [==============================] - 0s 73us/sample - loss: -56624355.1800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1834/5000
    500/500 [==============================] - 0s 82us/sample - loss: 98506825.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1835/5000
    500/500 [==============================] - 0s 82us/sample - loss: -63202029.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1836/5000
    500/500 [==============================] - 0s 82us/sample - loss: -7479911.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1837/5000
    500/500 [==============================] - 0s 82us/sample - loss: 65149532.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1838/5000
    500/500 [==============================] - 0s 80us/sample - loss: 62933378.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1839/5000
    500/500 [==============================] - 0s 82us/sample - loss: -82870981.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1840/5000
    500/500 [==============================] - 0s 78us/sample - loss: -10987056.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1841/5000
    500/500 [==============================] - 0s 84us/sample - loss: 186026604.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1842/5000
    500/500 [==============================] - 0s 80us/sample - loss: -124582337.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1843/5000
    500/500 [==============================] - 0s 76us/sample - loss: 152941305.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1844/5000
    500/500 [==============================] - 0s 84us/sample - loss: -38082835.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1845/5000
    500/500 [==============================] - 0s 77us/sample - loss: -99569840.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1846/5000
    500/500 [==============================] - 0s 80us/sample - loss: -49915120.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1847/5000
    500/500 [==============================] - 0s 86us/sample - loss: -62754509.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1848/5000
    500/500 [==============================] - 0s 80us/sample - loss: -99759246.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1849/5000
    500/500 [==============================] - 0s 84us/sample - loss: 51283256.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1850/5000
    500/500 [==============================] - 0s 84us/sample - loss: 168440444.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1851/5000
    500/500 [==============================] - 0s 82us/sample - loss: 174288528.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1852/5000
    500/500 [==============================] - 0s 84us/sample - loss: -57409258.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1853/5000
    500/500 [==============================] - 0s 86us/sample - loss: 112734091.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1854/5000
    500/500 [==============================] - 0s 83us/sample - loss: 140819232.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1855/5000
    500/500 [==============================] - 0s 82us/sample - loss: 54991490.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1856/5000
    500/500 [==============================] - 0s 78us/sample - loss: 33866108.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1857/5000
    500/500 [==============================] - 0s 84us/sample - loss: 155328967.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1858/5000
    500/500 [==============================] - 0s 80us/sample - loss: -52363172.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1859/5000
    500/500 [==============================] - 0s 82us/sample - loss: -82171397.3765 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1860/5000
    500/500 [==============================] - 0s 80us/sample - loss: 141854905.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1861/5000
    500/500 [==============================] - 0s 84us/sample - loss: 103894751.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1862/5000
    500/500 [==============================] - 0s 80us/sample - loss: 51913592.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1863/5000
    500/500 [==============================] - 0s 82us/sample - loss: 63959917.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1864/5000
    500/500 [==============================] - 0s 92us/sample - loss: 29205200.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1865/5000
    500/500 [==============================] - 0s 78us/sample - loss: 15558920.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1866/5000
    500/500 [==============================] - 0s 74us/sample - loss: 10880323.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1867/5000
    500/500 [==============================] - 0s 82us/sample - loss: -53384601.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1868/5000
    500/500 [==============================] - 0s 90us/sample - loss: -46138017.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1869/5000
    500/500 [==============================] - 0s 100us/sample - loss: 42398402.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1870/5000
    500/500 [==============================] - 0s 86us/sample - loss: 224777846.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1871/5000
    500/500 [==============================] - 0s 94us/sample - loss: 10198126.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1872/5000
    500/500 [==============================] - 0s 78us/sample - loss: -31730928.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1873/5000
    500/500 [==============================] - 0s 96us/sample - loss: 253474153.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1874/5000
    500/500 [==============================] - 0s 86us/sample - loss: -9171075.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1875/5000
    500/500 [==============================] - 0s 94us/sample - loss: 184755245.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1876/5000
    500/500 [==============================] - 0s 78us/sample - loss: -55371242.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1877/5000
    500/500 [==============================] - 0s 94us/sample - loss: -74042814.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1878/5000
    500/500 [==============================] - 0s 76us/sample - loss: 92341007.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1879/5000
    500/500 [==============================] - 0s 76us/sample - loss: -6926546.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1880/5000
    500/500 [==============================] - 0s 80us/sample - loss: 104706823.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1881/5000
    500/500 [==============================] - 0s 84us/sample - loss: 89655947.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1882/5000
    500/500 [==============================] - 0s 78us/sample - loss: -102449967.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1883/5000
    500/500 [==============================] - 0s 76us/sample - loss: -23003044.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1884/5000
    500/500 [==============================] - 0s 78us/sample - loss: 24282616.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1885/5000
    500/500 [==============================] - 0s 82us/sample - loss: 4991922.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1886/5000
    500/500 [==============================] - 0s 80us/sample - loss: -16762908.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1887/5000
    500/500 [==============================] - 0s 80us/sample - loss: 71534961.1860 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1888/5000
    500/500 [==============================] - 0s 78us/sample - loss: 76572892.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1889/5000
    500/500 [==============================] - 0s 84us/sample - loss: 145844686.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1890/5000
    500/500 [==============================] - 0s 78us/sample - loss: -56466656.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1891/5000
    500/500 [==============================] - 0s 88us/sample - loss: -186181966.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1892/5000
    500/500 [==============================] - 0s 82us/sample - loss: -66482402.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1893/5000
    500/500 [==============================] - 0s 80us/sample - loss: -32174819.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1894/5000
    500/500 [==============================] - 0s 84us/sample - loss: -84617692.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1895/5000
    500/500 [==============================] - 0s 84us/sample - loss: 130813541.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1896/5000
    500/500 [==============================] - 0s 84us/sample - loss: 100705158.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1897/5000
    500/500 [==============================] - 0s 80us/sample - loss: 127431250.9620 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1898/5000
    500/500 [==============================] - 0s 88us/sample - loss: 34692076.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1899/5000
    500/500 [==============================] - 0s 84us/sample - loss: 76539023.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1900/5000
    500/500 [==============================] - 0s 88us/sample - loss: 89849528.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1901/5000
    500/500 [==============================] - 0s 86us/sample - loss: 36156390.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1902/5000
    500/500 [==============================] - 0s 86us/sample - loss: 27424585.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1903/5000
    500/500 [==============================] - 0s 84us/sample - loss: -126633908.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1904/5000
    500/500 [==============================] - 0s 84us/sample - loss: -163840596.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1905/5000
    500/500 [==============================] - 0s 94us/sample - loss: -151966158.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1906/5000
    500/500 [==============================] - 0s 76us/sample - loss: 70705962.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1907/5000
    500/500 [==============================] - 0s 82us/sample - loss: 37424940.9720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1908/5000
    500/500 [==============================] - 0s 82us/sample - loss: -11929586.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1909/5000
    500/500 [==============================] - 0s 84us/sample - loss: 3469430.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1910/5000
    500/500 [==============================] - 0s 82us/sample - loss: -115706682.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1911/5000
    500/500 [==============================] - 0s 84us/sample - loss: -72218540.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1912/5000
    500/500 [==============================] - 0s 84us/sample - loss: -45833075.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1913/5000
    500/500 [==============================] - 0s 86us/sample - loss: 43140994.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1914/5000
    500/500 [==============================] - 0s 84us/sample - loss: -91268336.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1915/5000
    500/500 [==============================] - 0s 92us/sample - loss: -8413830.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1916/5000
    500/500 [==============================] - 0s 78us/sample - loss: -94102142.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1917/5000
    500/500 [==============================] - 0s 74us/sample - loss: -77278131.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1918/5000
    500/500 [==============================] - 0s 88us/sample - loss: -67392496.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1919/5000
    500/500 [==============================] - 0s 76us/sample - loss: -67607822.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1920/5000
    500/500 [==============================] - 0s 84us/sample - loss: -132341719.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1921/5000
    500/500 [==============================] - 0s 82us/sample - loss: -17870438.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1922/5000
    500/500 [==============================] - 0s 142us/sample - loss: 46717101.6317 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1923/5000
    500/500 [==============================] - 0s 113us/sample - loss: -81472591.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1924/5000
    500/500 [==============================] - 0s 74us/sample - loss: 18616912.0635 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1925/5000
    500/500 [==============================] - 0s 84us/sample - loss: -30652337.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1926/5000
    500/500 [==============================] - 0s 84us/sample - loss: -55934472.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1927/5000
    500/500 [==============================] - 0s 78us/sample - loss: -45907981.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1928/5000
    500/500 [==============================] - 0s 82us/sample - loss: 168971369.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1929/5000
    500/500 [==============================] - 0s 82us/sample - loss: 234037206.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1930/5000
    500/500 [==============================] - 0s 80us/sample - loss: -92176058.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1931/5000
    500/500 [==============================] - 0s 82us/sample - loss: 18090679.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1932/5000
    500/500 [==============================] - 0s 84us/sample - loss: 60223315.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1933/5000
    500/500 [==============================] - 0s 82us/sample - loss: 3173163.8555 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1934/5000
    500/500 [==============================] - 0s 76us/sample - loss: 8449839.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1935/5000
    500/500 [==============================] - 0s 78us/sample - loss: 111049831.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1936/5000
    500/500 [==============================] - 0s 84us/sample - loss: 81351229.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1937/5000
    500/500 [==============================] - 0s 80us/sample - loss: -222486298.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1938/5000
    500/500 [==============================] - 0s 82us/sample - loss: 176898074.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1939/5000
    500/500 [==============================] - 0s 80us/sample - loss: -25158012.3195 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1940/5000
    500/500 [==============================] - 0s 80us/sample - loss: 143614744.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1941/5000
    500/500 [==============================] - 0s 78us/sample - loss: 79062542.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1942/5000
    500/500 [==============================] - 0s 84us/sample - loss: -131425682.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1943/5000
    500/500 [==============================] - 0s 80us/sample - loss: -11724922.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1944/5000
    500/500 [==============================] - 0s 84us/sample - loss: -13301283.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1945/5000
    500/500 [==============================] - 0s 84us/sample - loss: 18134090.8520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1946/5000
    500/500 [==============================] - 0s 84us/sample - loss: 21574412.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1947/5000
    500/500 [==============================] - 0s 90us/sample - loss: 128068551.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1948/5000
    500/500 [==============================] - 0s 84us/sample - loss: 2418727.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1949/5000
    500/500 [==============================] - 0s 82us/sample - loss: 14871325.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1950/5000
    500/500 [==============================] - 0s 86us/sample - loss: 64641799.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1951/5000
    500/500 [==============================] - 0s 92us/sample - loss: 141052385.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1952/5000
    500/500 [==============================] - 0s 78us/sample - loss: 17925056.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1953/5000
    500/500 [==============================] - 0s 94us/sample - loss: 184112836.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1954/5000
    500/500 [==============================] - 0s 86us/sample - loss: 64832703.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1955/5000
    500/500 [==============================] - 0s 84us/sample - loss: 29454652.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1956/5000
    500/500 [==============================] - 0s 86us/sample - loss: -49200871.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1957/5000
    500/500 [==============================] - 0s 92us/sample - loss: -184911919.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1958/5000
    500/500 [==============================] - 0s 76us/sample - loss: -115663154.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1959/5000
    500/500 [==============================] - 0s 77us/sample - loss: 12559075.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1960/5000
    500/500 [==============================] - 0s 80us/sample - loss: 13683450.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1961/5000
    500/500 [==============================] - 0s 86us/sample - loss: -19624734.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1962/5000
    500/500 [==============================] - 0s 80us/sample - loss: 58131939.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1963/5000
    500/500 [==============================] - 0s 80us/sample - loss: 147101853.8840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1964/5000
    500/500 [==============================] - 0s 82us/sample - loss: 13480740.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1965/5000
    500/500 [==============================] - 0s 82us/sample - loss: 191093623.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1966/5000
    500/500 [==============================] - 0s 84us/sample - loss: 36243323.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1967/5000
    500/500 [==============================] - 0s 94us/sample - loss: 169117553.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1968/5000
    500/500 [==============================] - 0s 82us/sample - loss: -66609234.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1969/5000
    500/500 [==============================] - 0s 82us/sample - loss: -39682469.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1970/5000
    500/500 [==============================] - 0s 80us/sample - loss: -8998692.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1971/5000
    500/500 [==============================] - 0s 82us/sample - loss: -49004031.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1972/5000
    500/500 [==============================] - 0s 96us/sample - loss: 41504421.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1973/5000
    500/500 [==============================] - 0s 78us/sample - loss: -182255587.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1974/5000
    500/500 [==============================] - 0s 72us/sample - loss: -87133379.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1975/5000
    500/500 [==============================] - 0s 82us/sample - loss: -47369329.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1976/5000
    500/500 [==============================] - 0s 84us/sample - loss: -30060979.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1977/5000
    500/500 [==============================] - 0s 82us/sample - loss: -16017075.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1978/5000
    500/500 [==============================] - 0s 84us/sample - loss: -44596894.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1979/5000
    500/500 [==============================] - 0s 78us/sample - loss: 23073189.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1980/5000
    500/500 [==============================] - 0s 82us/sample - loss: 119231460.5439 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1981/5000
    500/500 [==============================] - 0s 80us/sample - loss: -21761021.3115 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1982/5000
    500/500 [==============================] - 0s 86us/sample - loss: 31828569.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1983/5000
    500/500 [==============================] - 0s 92us/sample - loss: -89068774.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1984/5000
    500/500 [==============================] - 0s 78us/sample - loss: -59828368.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1985/5000
    500/500 [==============================] - 0s 74us/sample - loss: 43673157.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1986/5000
    500/500 [==============================] - 0s 80us/sample - loss: -229189461.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1987/5000
    500/500 [==============================] - 0s 86us/sample - loss: -26295025.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1988/5000
    500/500 [==============================] - 0s 82us/sample - loss: 51507948.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1989/5000
    500/500 [==============================] - 0s 84us/sample - loss: -7092601.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1990/5000
    500/500 [==============================] - 0s 92us/sample - loss: -11821628.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1991/5000
    500/500 [==============================] - 0s 76us/sample - loss: 34504554.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1992/5000
    500/500 [==============================] - 0s 82us/sample - loss: -9794066.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1993/5000
    500/500 [==============================] - 0s 82us/sample - loss: -81024291.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1994/5000
    500/500 [==============================] - 0s 74us/sample - loss: 93490832.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1995/5000
    500/500 [==============================] - 0s 82us/sample - loss: -62453192.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1996/5000
    500/500 [==============================] - 0s 82us/sample - loss: -40102451.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1997/5000
    500/500 [==============================] - 0s 80us/sample - loss: 154601407.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1998/5000
    500/500 [==============================] - 0s 86us/sample - loss: -25975804.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 1999/5000
    500/500 [==============================] - 0s 82us/sample - loss: -38512969.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2000/5000
    500/500 [==============================] - 0s 82us/sample - loss: -12990202.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2001/5000
    500/500 [==============================] - 0s 94us/sample - loss: -50602714.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2002/5000
    500/500 [==============================] - 0s 74us/sample - loss: -60493259.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2003/5000
    500/500 [==============================] - 0s 84us/sample - loss: -259674847.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2004/5000
    500/500 [==============================] - 0s 82us/sample - loss: -174234351.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2005/5000
    500/500 [==============================] - 0s 82us/sample - loss: -210218567.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2006/5000
    500/500 [==============================] - 0s 88us/sample - loss: 70530866.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2007/5000
    500/500 [==============================] - 0s 82us/sample - loss: 21997800.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2008/5000
    500/500 [==============================] - 0s 84us/sample - loss: -28775109.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2009/5000
    500/500 [==============================] - 0s 80us/sample - loss: -50485013.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2010/5000
    500/500 [==============================] - 0s 84us/sample - loss: -1768089.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2011/5000
    500/500 [==============================] - 0s 88us/sample - loss: -58634899.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2012/5000
    500/500 [==============================] - 0s 82us/sample - loss: -78912904.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2013/5000
    500/500 [==============================] - 0s 82us/sample - loss: 131901351.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2014/5000
    500/500 [==============================] - 0s 80us/sample - loss: -95518769.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2015/5000
    500/500 [==============================] - 0s 86us/sample - loss: 80122692.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2016/5000
    500/500 [==============================] - 0s 84us/sample - loss: -123256543.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2017/5000
    500/500 [==============================] - 0s 84us/sample - loss: 92567193.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2018/5000
    500/500 [==============================] - 0s 86us/sample - loss: 113701139.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2019/5000
    500/500 [==============================] - 0s 94us/sample - loss: 22288780.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2020/5000
    500/500 [==============================] - 0s 78us/sample - loss: 159097919.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2021/5000
    500/500 [==============================] - 0s 78us/sample - loss: -27213328.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2022/5000
    500/500 [==============================] - 0s 80us/sample - loss: 71094693.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2023/5000
    500/500 [==============================] - 0s 90us/sample - loss: -108234911.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2024/5000
    500/500 [==============================] - 0s 76us/sample - loss: -40205350.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2025/5000
    500/500 [==============================] - 0s 75us/sample - loss: -136725434.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2026/5000
    500/500 [==============================] - 0s 78us/sample - loss: -90850964.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2027/5000
    500/500 [==============================] - 0s 84us/sample - loss: 156177742.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2028/5000
    500/500 [==============================] - 0s 80us/sample - loss: 28265056.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2029/5000
    500/500 [==============================] - 0s 77us/sample - loss: 4946312.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2030/5000
    500/500 [==============================] - 0s 82us/sample - loss: 51394277.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2031/5000
    500/500 [==============================] - 0s 82us/sample - loss: 106735428.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2032/5000
    500/500 [==============================] - 0s 78us/sample - loss: -88000753.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2033/5000
    500/500 [==============================] - 0s 84us/sample - loss: 296074807.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2034/5000
    500/500 [==============================] - 0s 84us/sample - loss: 15223259.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2035/5000
    500/500 [==============================] - 0s 84us/sample - loss: -147221230.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2036/5000
    500/500 [==============================] - 0s 84us/sample - loss: -9606591.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2037/5000
    500/500 [==============================] - 0s 82us/sample - loss: 17116961.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2038/5000
    500/500 [==============================] - 0s 78us/sample - loss: -137309287.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2039/5000
    500/500 [==============================] - 0s 82us/sample - loss: 28015715.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2040/5000
    500/500 [==============================] - 0s 75us/sample - loss: 26556267.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2041/5000
    500/500 [==============================] - 0s 80us/sample - loss: -77707721.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2042/5000
    500/500 [==============================] - 0s 84us/sample - loss: -145350850.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2043/5000
    500/500 [==============================] - 0s 80us/sample - loss: -117380342.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2044/5000
    500/500 [==============================] - 0s 84us/sample - loss: -126047469.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2045/5000
    500/500 [==============================] - 0s 82us/sample - loss: 39778523.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2046/5000
    500/500 [==============================] - 0s 82us/sample - loss: -161949176.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2047/5000
    500/500 [==============================] - 0s 94us/sample - loss: -94282551.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2048/5000
    500/500 [==============================] - 0s 84us/sample - loss: -30406816.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2049/5000
    500/500 [==============================] - 0s 84us/sample - loss: -34042308.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2050/5000
    500/500 [==============================] - 0s 84us/sample - loss: -168856699.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2051/5000
    500/500 [==============================] - 0s 82us/sample - loss: -123748545.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2052/5000
    500/500 [==============================] - 0s 88us/sample - loss: 47135627.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2053/5000
    500/500 [==============================] - 0s 86us/sample - loss: 199798485.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2054/5000
    500/500 [==============================] - 0s 94us/sample - loss: 7304462.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2055/5000
    500/500 [==============================] - 0s 76us/sample - loss: -34844601.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2056/5000
    500/500 [==============================] - 0s 76us/sample - loss: -83899886.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2057/5000
    500/500 [==============================] - 0s 82us/sample - loss: 64040032.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2058/5000
    500/500 [==============================] - 0s 84us/sample - loss: 86404871.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2059/5000
    500/500 [==============================] - 0s 84us/sample - loss: 200198783.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2060/5000
    500/500 [==============================] - 0s 84us/sample - loss: -35780301.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2061/5000
    500/500 [==============================] - 0s 86us/sample - loss: 41380471.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2062/5000
    500/500 [==============================] - 0s 90us/sample - loss: -145112826.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2063/5000
    500/500 [==============================] - 0s 82us/sample - loss: 176937647.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2064/5000
    500/500 [==============================] - 0s 92us/sample - loss: 61179184.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2065/5000
    500/500 [==============================] - 0s 76us/sample - loss: 20295305.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2066/5000
    500/500 [==============================] - 0s 77us/sample - loss: -12661859.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2067/5000
    500/500 [==============================] - 0s 82us/sample - loss: 181427344.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2068/5000
    500/500 [==============================] - 0s 82us/sample - loss: 122171835.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2069/5000
    500/500 [==============================] - 0s 78us/sample - loss: 142973119.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2070/5000
    500/500 [==============================] - 0s 82us/sample - loss: -66513028.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2071/5000
    500/500 [==============================] - 0s 78us/sample - loss: -28235752.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2072/5000
    500/500 [==============================] - 0s 80us/sample - loss: -33150127.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2073/5000
    500/500 [==============================] - 0s 76us/sample - loss: -65912166.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2074/5000
    500/500 [==============================] - 0s 84us/sample - loss: -92146906.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2075/5000
    500/500 [==============================] - 0s 76us/sample - loss: 18240775.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2076/5000
    500/500 [==============================] - 0s 86us/sample - loss: -16501955.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2077/5000
    500/500 [==============================] - 0s 82us/sample - loss: 110069308.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2078/5000
    500/500 [==============================] - 0s 86us/sample - loss: 8069758.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2079/5000
    500/500 [==============================] - 0s 84us/sample - loss: -7436179.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2080/5000
    500/500 [==============================] - 0s 82us/sample - loss: -147220599.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2081/5000
    500/500 [==============================] - 0s 92us/sample - loss: 29718495.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2082/5000
    500/500 [==============================] - 0s 76us/sample - loss: -109178755.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2083/5000
    500/500 [==============================] - 0s 75us/sample - loss: -112907566.0200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2084/5000
    500/500 [==============================] - 0s 80us/sample - loss: -13452463.3160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2085/5000
    500/500 [==============================] - 0s 86us/sample - loss: -106903600.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2086/5000
    500/500 [==============================] - 0s 82us/sample - loss: 177759.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2087/5000
    500/500 [==============================] - 0s 82us/sample - loss: -6035875.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2088/5000
    500/500 [==============================] - 0s 86us/sample - loss: 78199057.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2089/5000
    500/500 [==============================] - 0s 84us/sample - loss: 23217896.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2090/5000
    500/500 [==============================] - 0s 84us/sample - loss: 131023292.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2091/5000
    500/500 [==============================] - 0s 82us/sample - loss: 1928877.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2092/5000
    500/500 [==============================] - 0s 82us/sample - loss: -201258160.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2093/5000
    500/500 [==============================] - 0s 96us/sample - loss: -97339527.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2094/5000
    500/500 [==============================] - 0s 76us/sample - loss: -82676908.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2095/5000
    500/500 [==============================] - 0s 70us/sample - loss: -2148859.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2096/5000
    500/500 [==============================] - 0s 80us/sample - loss: -44339841.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2097/5000
    500/500 [==============================] - 0s 82us/sample - loss: 35151633.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2098/5000
    500/500 [==============================] - 0s 82us/sample - loss: 7226832.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2099/5000
    500/500 [==============================] - 0s 86us/sample - loss: -110055280.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2100/5000
    500/500 [==============================] - 0s 84us/sample - loss: -40933144.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2101/5000
    500/500 [==============================] - 0s 90us/sample - loss: -46286475.2950 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2102/5000
    500/500 [==============================] - 0s 80us/sample - loss: -24492651.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2103/5000
    500/500 [==============================] - 0s 82us/sample - loss: -40526435.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2104/5000
    500/500 [==============================] - 0s 80us/sample - loss: 96844160.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2105/5000
    500/500 [==============================] - 0s 84us/sample - loss: -154245588.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2106/5000
    500/500 [==============================] - 0s 80us/sample - loss: -81436029.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2107/5000
    500/500 [==============================] - 0s 84us/sample - loss: 29739027.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2108/5000
    500/500 [==============================] - 0s 82us/sample - loss: 90785322.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2109/5000
    500/500 [==============================] - 0s 88us/sample - loss: -39336276.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2110/5000
    500/500 [==============================] - 0s 84us/sample - loss: -110166036.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2111/5000
    500/500 [==============================] - 0s 82us/sample - loss: 46089976.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2112/5000
    500/500 [==============================] - 0s 110us/sample - loss: 180220891.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2113/5000
    500/500 [==============================] - 0s 99us/sample - loss: -157982711.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2114/5000
    500/500 [==============================] - 0s 94us/sample - loss: -70578510.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2115/5000
    500/500 [==============================] - 0s 90us/sample - loss: 4643908.3160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2116/5000
    500/500 [==============================] - 0s 92us/sample - loss: 30683809.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2117/5000
    500/500 [==============================] - 0s 92us/sample - loss: 154746473.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2118/5000
    500/500 [==============================] - 0s 94us/sample - loss: 39534028.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2119/5000
    500/500 [==============================] - 0s 102us/sample - loss: -163231560.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2120/5000
    500/500 [==============================] - 0s 98us/sample - loss: 50471026.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2121/5000
    500/500 [==============================] - 0s 93us/sample - loss: 14744688.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2122/5000
    500/500 [==============================] - 0s 96us/sample - loss: 109846313.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2123/5000
    500/500 [==============================] - 0s 102us/sample - loss: 67374547.2040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2124/5000
    500/500 [==============================] - 0s 94us/sample - loss: 44574799.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2125/5000
    500/500 [==============================] - 0s 95us/sample - loss: 80506795.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2126/5000
    500/500 [==============================] - 0s 92us/sample - loss: 94809267.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2127/5000
    500/500 [==============================] - 0s 90us/sample - loss: 37316782.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2128/5000
    500/500 [==============================] - 0s 88us/sample - loss: 68457428.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2129/5000
    500/500 [==============================] - 0s 108us/sample - loss: 35835095.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2130/5000
    500/500 [==============================] - 0s 98us/sample - loss: 171956146.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2131/5000
    500/500 [==============================] - 0s 91us/sample - loss: 100741183.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2132/5000
    500/500 [==============================] - 0s 92us/sample - loss: -136389974.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2133/5000
    500/500 [==============================] - 0s 92us/sample - loss: 13678747.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2134/5000
    500/500 [==============================] - 0s 102us/sample - loss: 43643810.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2135/5000
    500/500 [==============================] - 0s 93us/sample - loss: -144274304.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2136/5000
    500/500 [==============================] - 0s 112us/sample - loss: -78072651.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2137/5000
    500/500 [==============================] - 0s 98us/sample - loss: 49185339.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2138/5000
    500/500 [==============================] - 0s 102us/sample - loss: 6700467.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2139/5000
    500/500 [==============================] - 0s 106us/sample - loss: -111589818.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2140/5000
    500/500 [==============================] - 0s 100us/sample - loss: 43776848.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2141/5000
    500/500 [==============================] - 0s 93us/sample - loss: -131628103.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2142/5000
    500/500 [==============================] - 0s 92us/sample - loss: -2462402.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2143/5000
    500/500 [==============================] - 0s 94us/sample - loss: 3275183.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2144/5000
    500/500 [==============================] - 0s 102us/sample - loss: 29576738.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2145/5000
    500/500 [==============================] - 0s 98us/sample - loss: 85991641.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2146/5000
    500/500 [==============================] - 0s 98us/sample - loss: -3903979.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2147/5000
    500/500 [==============================] - 0s 100us/sample - loss: 137647261.6920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2148/5000
    500/500 [==============================] - 0s 92us/sample - loss: -74019651.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2149/5000
    500/500 [==============================] - 0s 96us/sample - loss: -106717547.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2150/5000
    500/500 [==============================] - 0s 92us/sample - loss: 107481660.7700 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2151/5000
    500/500 [==============================] - 0s 110us/sample - loss: -120988640.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2152/5000
    500/500 [==============================] - 0s 102us/sample - loss: -114841081.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2153/5000
    500/500 [==============================] - 0s 102us/sample - loss: -86015128.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2154/5000
    500/500 [==============================] - 0s 84us/sample - loss: 22499025.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2155/5000
    500/500 [==============================] - 0s 84us/sample - loss: -22825751.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2156/5000
    500/500 [==============================] - 0s 86us/sample - loss: 28016039.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2157/5000
    500/500 [==============================] - 0s 92us/sample - loss: -94734160.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2158/5000
    500/500 [==============================] - 0s 78us/sample - loss: 65451345.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2159/5000
    500/500 [==============================] - 0s 86us/sample - loss: -134559042.4920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2160/5000
    500/500 [==============================] - 0s 84us/sample - loss: -41695226.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2161/5000
    500/500 [==============================] - 0s 90us/sample - loss: 141214408.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2162/5000
    500/500 [==============================] - 0s 88us/sample - loss: 22963994.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2163/5000
    500/500 [==============================] - 0s 80us/sample - loss: 145663266.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2164/5000
    500/500 [==============================] - 0s 80us/sample - loss: 61134887.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2165/5000
    500/500 [==============================] - 0s 80us/sample - loss: -45229313.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2166/5000
    500/500 [==============================] - 0s 82us/sample - loss: -56870218.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2167/5000
    500/500 [==============================] - 0s 80us/sample - loss: -101109257.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2168/5000
    500/500 [==============================] - 0s 82us/sample - loss: -77351174.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2169/5000
    500/500 [==============================] - 0s 80us/sample - loss: 42564429.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2170/5000
    500/500 [==============================] - 0s 86us/sample - loss: 29826504.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2171/5000
    500/500 [==============================] - 0s 84us/sample - loss: 8495612.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2172/5000
    500/500 [==============================] - 0s 84us/sample - loss: 133388333.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2173/5000
    500/500 [==============================] - 0s 84us/sample - loss: 83804913.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2174/5000
    500/500 [==============================] - 0s 88us/sample - loss: 41673418.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2175/5000
    500/500 [==============================] - 0s 78us/sample - loss: 50044837.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2176/5000
    500/500 [==============================] - 0s 84us/sample - loss: 134855113.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2177/5000
    500/500 [==============================] - 0s 82us/sample - loss: 79835513.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2178/5000
    500/500 [==============================] - 0s 86us/sample - loss: -99666755.0730 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2179/5000
    500/500 [==============================] - 0s 90us/sample - loss: 33790406.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2180/5000
    500/500 [==============================] - 0s 74us/sample - loss: -87539058.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2181/5000
    500/500 [==============================] - 0s 82us/sample - loss: -88915363.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2182/5000
    500/500 [==============================] - 0s 80us/sample - loss: -60535369.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2183/5000
    500/500 [==============================] - 0s 90us/sample - loss: 147460086.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2184/5000
    500/500 [==============================] - 0s 78us/sample - loss: 51351667.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2185/5000
    500/500 [==============================] - 0s 74us/sample - loss: 64084015.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2186/5000
    500/500 [==============================] - 0s 82us/sample - loss: -15321340.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2187/5000
    500/500 [==============================] - 0s 84us/sample - loss: -199900151.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2188/5000
    500/500 [==============================] - 0s 84us/sample - loss: -17846972.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2189/5000
    500/500 [==============================] - 0s 84us/sample - loss: -60442724.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2190/5000
    500/500 [==============================] - 0s 82us/sample - loss: 12728812.1240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2191/5000
    500/500 [==============================] - 0s 90us/sample - loss: 124301848.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2192/5000
    500/500 [==============================] - 0s 78us/sample - loss: 180460887.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2193/5000
    500/500 [==============================] - 0s 84us/sample - loss: -80327100.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2194/5000
    500/500 [==============================] - 0s 82us/sample - loss: 85818189.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2195/5000
    500/500 [==============================] - 0s 96us/sample - loss: -95515665.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2196/5000
    500/500 [==============================] - 0s 86us/sample - loss: 64083337.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2197/5000
    500/500 [==============================] - 0s 92us/sample - loss: -141420862.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2198/5000
    500/500 [==============================] - 0s 76us/sample - loss: -5531500.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2199/5000
    500/500 [==============================] - 0s 84us/sample - loss: 45500635.0040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2200/5000
    500/500 [==============================] - 0s 82us/sample - loss: 190118592.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2201/5000
    500/500 [==============================] - 0s 84us/sample - loss: -166829574.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2202/5000
    500/500 [==============================] - 0s 96us/sample - loss: 91106652.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2203/5000
    500/500 [==============================] - 0s 78us/sample - loss: 122810252.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2204/5000
    500/500 [==============================] - 0s 84us/sample - loss: 55504098.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2205/5000
    500/500 [==============================] - 0s 82us/sample - loss: 97036345.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2206/5000
    500/500 [==============================] - 0s 88us/sample - loss: -61658804.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2207/5000
    500/500 [==============================] - 0s 92us/sample - loss: -29708157.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2208/5000
    500/500 [==============================] - 0s 96us/sample - loss: -7565286.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2209/5000
    500/500 [==============================] - 0s 74us/sample - loss: 7548148.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2210/5000
    500/500 [==============================] - 0s 82us/sample - loss: -52279177.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2211/5000
    500/500 [==============================] - 0s 80us/sample - loss: 190593642.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2212/5000
    500/500 [==============================] - 0s 84us/sample - loss: -16159713.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2213/5000
    500/500 [==============================] - 0s 82us/sample - loss: 110319670.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2214/5000
    500/500 [==============================] - 0s 75us/sample - loss: -85279809.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2215/5000
    500/500 [==============================] - 0s 80us/sample - loss: -50681584.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2216/5000
    500/500 [==============================] - 0s 86us/sample - loss: -3369232.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2217/5000
    500/500 [==============================] - 0s 80us/sample - loss: 115950686.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2218/5000
    500/500 [==============================] - 0s 76us/sample - loss: -29990643.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2219/5000
    500/500 [==============================] - 0s 80us/sample - loss: 26011460.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2220/5000
    500/500 [==============================] - 0s 84us/sample - loss: -8924829.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2221/5000
    500/500 [==============================] - 0s 82us/sample - loss: 129305516.5480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2222/5000
    500/500 [==============================] - 0s 84us/sample - loss: 25983073.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2223/5000
    500/500 [==============================] - 0s 78us/sample - loss: -91012398.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2224/5000
    500/500 [==============================] - 0s 84us/sample - loss: 53873828.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2225/5000
    500/500 [==============================] - 0s 80us/sample - loss: 81217276.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2226/5000
    500/500 [==============================] - 0s 82us/sample - loss: 141034582.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2227/5000
    500/500 [==============================] - 0s 80us/sample - loss: 10699437.8920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2228/5000
    500/500 [==============================] - 0s 80us/sample - loss: -27993672.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2229/5000
    500/500 [==============================] - 0s 96us/sample - loss: -58382206.2070 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2230/5000
    500/500 [==============================] - 0s 88us/sample - loss: 19271162.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2231/5000
    500/500 [==============================] - ETA: 0s - loss: -424331264.0000 - accuracy: 0.0000e+0 - 0s 92us/sample - loss: -53823534.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2232/5000
    500/500 [==============================] - 0s 76us/sample - loss: -33591669.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2233/5000
    500/500 [==============================] - 0s 84us/sample - loss: -48043361.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2234/5000
    500/500 [==============================] - 0s 90us/sample - loss: -38742966.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2235/5000
    500/500 [==============================] - 0s 78us/sample - loss: -46486396.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2236/5000
    500/500 [==============================] - 0s 86us/sample - loss: -22822952.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2237/5000
    500/500 [==============================] - 0s 94us/sample - loss: 30994999.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2238/5000
    500/500 [==============================] - 0s 80us/sample - loss: -214174828.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2239/5000
    500/500 [==============================] - 0s 86us/sample - loss: 24100871.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2240/5000
    500/500 [==============================] - 0s 86us/sample - loss: -94308289.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2241/5000
    500/500 [==============================] - 0s 84us/sample - loss: -215342806.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2242/5000
    500/500 [==============================] - 0s 90us/sample - loss: -149045299.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2243/5000
    500/500 [==============================] - 0s 76us/sample - loss: -71702622.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2244/5000
    500/500 [==============================] - 0s 73us/sample - loss: -24570531.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2245/5000
    500/500 [==============================] - 0s 82us/sample - loss: 211586056.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2246/5000
    500/500 [==============================] - 0s 82us/sample - loss: -22680737.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2247/5000
    500/500 [==============================] - 0s 82us/sample - loss: 124108478.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2248/5000
    500/500 [==============================] - 0s 84us/sample - loss: 58185826.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2249/5000
    500/500 [==============================] - 0s 82us/sample - loss: -178594906.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2250/5000
    500/500 [==============================] - 0s 88us/sample - loss: -25886861.8200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2251/5000
    500/500 [==============================] - 0s 78us/sample - loss: 179622542.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2252/5000
    500/500 [==============================] - 0s 84us/sample - loss: 78842841.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2253/5000
    500/500 [==============================] - 0s 86us/sample - loss: -34651381.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2254/5000
    500/500 [==============================] - 0s 92us/sample - loss: 8160936.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2255/5000
    500/500 [==============================] - 0s 78us/sample - loss: 100645570.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2256/5000
    500/500 [==============================] - 0s 74us/sample - loss: -54140033.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2257/5000
    500/500 [==============================] - 0s 82us/sample - loss: 16221402.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2258/5000
    500/500 [==============================] - 0s 82us/sample - loss: 13724750.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2259/5000
    500/500 [==============================] - 0s 78us/sample - loss: -120727696.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2260/5000
    500/500 [==============================] - 0s 82us/sample - loss: 20208568.5720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2261/5000
    500/500 [==============================] - 0s 80us/sample - loss: -29943559.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2262/5000
    500/500 [==============================] - 0s 84us/sample - loss: 225748118.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2263/5000
    500/500 [==============================] - 0s 84us/sample - loss: -39581462.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2264/5000
    500/500 [==============================] - 0s 84us/sample - loss: 31808391.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2265/5000
    500/500 [==============================] - 0s 86us/sample - loss: -5252504.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2266/5000
    500/500 [==============================] - 0s 86us/sample - loss: 48735898.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2267/5000
    500/500 [==============================] - 0s 76us/sample - loss: -79536707.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2268/5000
    500/500 [==============================] - 0s 80us/sample - loss: -77313222.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2269/5000
    500/500 [==============================] - 0s 80us/sample - loss: 64274209.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2270/5000
    500/500 [==============================] - 0s 82us/sample - loss: 62801008.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2271/5000
    500/500 [==============================] - 0s 80us/sample - loss: 91691304.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2272/5000
    500/500 [==============================] - 0s 82us/sample - loss: 170760136.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2273/5000
    500/500 [==============================] - 0s 78us/sample - loss: 14789394.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2274/5000
    500/500 [==============================] - 0s 84us/sample - loss: 69831526.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2275/5000
    500/500 [==============================] - 0s 82us/sample - loss: -84762824.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2276/5000
    500/500 [==============================] - 0s 82us/sample - loss: -64046843.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2277/5000
    500/500 [==============================] - 0s 86us/sample - loss: 106679222.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2278/5000
    500/500 [==============================] - 0s 86us/sample - loss: -78660783.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2279/5000
    500/500 [==============================] - 0s 81us/sample - loss: 42323500.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2280/5000
    500/500 [==============================] - 0s 80us/sample - loss: -47623932.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2281/5000
    500/500 [==============================] - 0s 86us/sample - loss: -2913951.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2282/5000
    500/500 [==============================] - 0s 94us/sample - loss: -63411755.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2283/5000
    500/500 [==============================] - 0s 78us/sample - loss: 117111823.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2284/5000
    500/500 [==============================] - 0s 74us/sample - loss: 34508536.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2285/5000
    500/500 [==============================] - 0s 82us/sample - loss: 81871029.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2286/5000
    500/500 [==============================] - 0s 92us/sample - loss: -17824923.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2287/5000
    500/500 [==============================] - 0s 86us/sample - loss: -130531057.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2288/5000
    500/500 [==============================] - 0s 96us/sample - loss: 101760412.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2289/5000
    500/500 [==============================] - 0s 78us/sample - loss: 40576855.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2290/5000
    500/500 [==============================] - 0s 98us/sample - loss: -109171140.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2291/5000
    500/500 [==============================] - 0s 86us/sample - loss: 163092098.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2292/5000
    500/500 [==============================] - 0s 84us/sample - loss: 161679237.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2293/5000
    500/500 [==============================] - 0s 88us/sample - loss: 7265133.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2294/5000
    500/500 [==============================] - 0s 90us/sample - loss: 18016902.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2295/5000
    500/500 [==============================] - 0s 96us/sample - loss: 148767722.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2296/5000
    500/500 [==============================] - 0s 76us/sample - loss: -23604495.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2297/5000
    500/500 [==============================] - 0s 76us/sample - loss: 41181426.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2298/5000
    500/500 [==============================] - 0s 82us/sample - loss: 45061514.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2299/5000
    500/500 [==============================] - 0s 86us/sample - loss: 133224082.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2300/5000
    500/500 [==============================] - 0s 86us/sample - loss: 117213025.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2301/5000
    500/500 [==============================] - 0s 82us/sample - loss: -81833173.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2302/5000
    500/500 [==============================] - 0s 84us/sample - loss: -88691392.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2303/5000
    500/500 [==============================] - 0s 80us/sample - loss: 73776785.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2304/5000
    500/500 [==============================] - 0s 80us/sample - loss: 135909406.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2305/5000
    500/500 [==============================] - 0s 80us/sample - loss: -28194480.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2306/5000
    500/500 [==============================] - 0s 84us/sample - loss: -73156487.6440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2307/5000
    500/500 [==============================] - 0s 82us/sample - loss: 49997252.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2308/5000
    500/500 [==============================] - 0s 80us/sample - loss: -121103903.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2309/5000
    500/500 [==============================] - 0s 86us/sample - loss: 37239683.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2310/5000
    500/500 [==============================] - 0s 82us/sample - loss: 94855134.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2311/5000
    500/500 [==============================] - 0s 75us/sample - loss: -117452954.9620 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2312/5000
    500/500 [==============================] - 0s 78us/sample - loss: 13640491.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2313/5000
    500/500 [==============================] - 0s 86us/sample - loss: 92728917.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2314/5000
    500/500 [==============================] - 0s 80us/sample - loss: 3840845.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2315/5000
    500/500 [==============================] - 0s 82us/sample - loss: -58762134.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2316/5000
    500/500 [==============================] - 0s 80us/sample - loss: 2715636.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2317/5000
    500/500 [==============================] - 0s 84us/sample - loss: 12591137.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2318/5000
    500/500 [==============================] - 0s 80us/sample - loss: -71138358.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2319/5000
    500/500 [==============================] - 0s 84us/sample - loss: 13644033.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2320/5000
    500/500 [==============================] - 0s 82us/sample - loss: 112315125.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2321/5000
    500/500 [==============================] - 0s 82us/sample - loss: -8672716.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2322/5000
    500/500 [==============================] - 0s 86us/sample - loss: -26531278.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2323/5000
    500/500 [==============================] - 0s 86us/sample - loss: -151040017.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2324/5000
    500/500 [==============================] - 0s 82us/sample - loss: 76392760.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2325/5000
    500/500 [==============================] - 0s 82us/sample - loss: -119639227.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2326/5000
    500/500 [==============================] - 0s 80us/sample - loss: -96274385.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2327/5000
    500/500 [==============================] - 0s 86us/sample - loss: 54572740.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2328/5000
    500/500 [==============================] - 0s 92us/sample - loss: 62129041.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2329/5000
    500/500 [==============================] - 0s 78us/sample - loss: 16081602.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2330/5000
    500/500 [==============================] - 0s 80us/sample - loss: -104719392.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2331/5000
    500/500 [==============================] - 0s 88us/sample - loss: 59770468.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2332/5000
    500/500 [==============================] - 0s 82us/sample - loss: 11809759.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2333/5000
    500/500 [==============================] - 0s 86us/sample - loss: 37369141.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2334/5000
    500/500 [==============================] - 0s 82us/sample - loss: -82867649.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2335/5000
    500/500 [==============================] - 0s 84us/sample - loss: -102491676.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2336/5000
    500/500 [==============================] - 0s 82us/sample - loss: 109942172.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2337/5000
    500/500 [==============================] - 0s 86us/sample - loss: 29631591.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2338/5000
    500/500 [==============================] - 0s 94us/sample - loss: 3500966.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2339/5000
    500/500 [==============================] - 0s 78us/sample - loss: 152557608.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2340/5000
    500/500 [==============================] - 0s 77us/sample - loss: 154216673.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2341/5000
    500/500 [==============================] - 0s 84us/sample - loss: -92867525.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2342/5000
    500/500 [==============================] - 0s 84us/sample - loss: 125606180.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2343/5000
    500/500 [==============================] - 0s 94us/sample - loss: 79158057.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2344/5000
    500/500 [==============================] - 0s 86us/sample - loss: 184934320.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2345/5000
    500/500 [==============================] - 0s 100us/sample - loss: -64835467.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2346/5000
    500/500 [==============================] - 0s 74us/sample - loss: 44602661.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2347/5000
    500/500 [==============================] - 0s 88us/sample - loss: -135529082.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2348/5000
    500/500 [==============================] - 0s 78us/sample - loss: -52586418.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2349/5000
    500/500 [==============================] - 0s 76us/sample - loss: 181872776.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2350/5000
    500/500 [==============================] - 0s 78us/sample - loss: 30080437.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2351/5000
    500/500 [==============================] - 0s 82us/sample - loss: -31573349.4710 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2352/5000
    500/500 [==============================] - 0s 80us/sample - loss: 93175932.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2353/5000
    500/500 [==============================] - 0s 84us/sample - loss: 1974821.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2354/5000
    500/500 [==============================] - 0s 78us/sample - loss: -38546844.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2355/5000
    500/500 [==============================] - 0s 82us/sample - loss: 22723990.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2356/5000
    500/500 [==============================] - 0s 80us/sample - loss: -26442621.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2357/5000
    500/500 [==============================] - 0s 82us/sample - loss: -9355119.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2358/5000
    500/500 [==============================] - 0s 80us/sample - loss: -57822144.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2359/5000
    500/500 [==============================] - 0s 84us/sample - loss: -83134344.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2360/5000
    500/500 [==============================] - 0s 84us/sample - loss: 126302493.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2361/5000
    500/500 [==============================] - 0s 82us/sample - loss: 52865546.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2362/5000
    500/500 [==============================] - 0s 82us/sample - loss: 86159591.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2363/5000
    500/500 [==============================] - 0s 90us/sample - loss: 15962512.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2364/5000
    500/500 [==============================] - 0s 72us/sample - loss: 1808829.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2365/5000
    500/500 [==============================] - 0s 51us/sample - loss: 7752471.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2366/5000
    500/500 [==============================] - 0s 68us/sample - loss: -16989505.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2367/5000
    500/500 [==============================] - 0s 67us/sample - loss: 21787151.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2368/5000
    500/500 [==============================] - 0s 105us/sample - loss: -66619019.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2369/5000
    500/500 [==============================] - 0s 86us/sample - loss: -6627498.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2370/5000
    500/500 [==============================] - 0s 86us/sample - loss: 37292834.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2371/5000
    500/500 [==============================] - 0s 82us/sample - loss: -71833304.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2372/5000
    500/500 [==============================] - 0s 82us/sample - loss: -202921082.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2373/5000
    500/500 [==============================] - 0s 94us/sample - loss: 18808065.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2374/5000
    500/500 [==============================] - 0s 90us/sample - loss: 130004840.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2375/5000
    500/500 [==============================] - 0s 94us/sample - loss: -49174104.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2376/5000
    500/500 [==============================] - 0s 76us/sample - loss: -88137304.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2377/5000
    500/500 [==============================] - 0s 94us/sample - loss: 146298091.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2378/5000
    500/500 [==============================] - 0s 78us/sample - loss: 22996649.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2379/5000
    500/500 [==============================] - 0s 77us/sample - loss: -72425393.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2380/5000
    500/500 [==============================] - 0s 94us/sample - loss: -1888083.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2381/5000
    500/500 [==============================] - 0s 78us/sample - loss: -145389658.8820 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2382/5000
    500/500 [==============================] - 0s 84us/sample - loss: 141936572.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2383/5000
    500/500 [==============================] - 0s 84us/sample - loss: -71007615.7300 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2384/5000
    500/500 [==============================] - 0s 84us/sample - loss: 6201064.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2385/5000
    500/500 [==============================] - 0s 84us/sample - loss: -2240706.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2386/5000
    500/500 [==============================] - 0s 94us/sample - loss: 87181816.5160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2387/5000
    500/500 [==============================] - 0s 78us/sample - loss: 98349586.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2388/5000
    500/500 [==============================] - 0s 82us/sample - loss: 249153565.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2389/5000
    500/500 [==============================] - 0s 78us/sample - loss: -46220136.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2390/5000
    500/500 [==============================] - 0s 90us/sample - loss: 71946908.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2391/5000
    500/500 [==============================] - ETA: 0s - loss: -894182016.0000 - accuracy: 0.0000e+0 - 0s 96us/sample - loss: 4020250.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2392/5000
    500/500 [==============================] - 0s 80us/sample - loss: -25205029.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2393/5000
    500/500 [==============================] - 0s 77us/sample - loss: -18830741.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2394/5000
    500/500 [==============================] - 0s 82us/sample - loss: 43400703.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2395/5000
    500/500 [==============================] - 0s 90us/sample - loss: 275089604.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2396/5000
    500/500 [==============================] - 0s 86us/sample - loss: 72120121.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2397/5000
    500/500 [==============================] - 0s 76us/sample - loss: -77803267.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2398/5000
    500/500 [==============================] - 0s 72us/sample - loss: 245328389.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2399/5000
    500/500 [==============================] - 0s 82us/sample - loss: 84827074.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2400/5000
    500/500 [==============================] - 0s 82us/sample - loss: 2638368.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2401/5000
    500/500 [==============================] - 0s 82us/sample - loss: -149520030.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2402/5000
    500/500 [==============================] - 0s 86us/sample - loss: -100509503.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2403/5000
    500/500 [==============================] - 0s 82us/sample - loss: -27577469.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2404/5000
    500/500 [==============================] - 0s 82us/sample - loss: 171576350.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2405/5000
    500/500 [==============================] - 0s 82us/sample - loss: -64828598.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2406/5000
    500/500 [==============================] - 0s 78us/sample - loss: -30022922.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2407/5000
    500/500 [==============================] - 0s 78us/sample - loss: -105478369.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2408/5000
    500/500 [==============================] - 0s 84us/sample - loss: -19848743.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2409/5000
    500/500 [==============================] - 0s 82us/sample - loss: -4010617.3430 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2410/5000
    500/500 [==============================] - 0s 82us/sample - loss: 77524121.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2411/5000
    500/500 [==============================] - 0s 86us/sample - loss: 150233295.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2412/5000
    500/500 [==============================] - 0s 82us/sample - loss: 13936152.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2413/5000
    500/500 [==============================] - 0s 86us/sample - loss: -55720565.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2414/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32681930.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2415/5000
    500/500 [==============================] - 0s 88us/sample - loss: 116275318.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2416/5000
    500/500 [==============================] - 0s 84us/sample - loss: 129029132.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2417/5000
    500/500 [==============================] - 0s 80us/sample - loss: -150106104.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2418/5000
    500/500 [==============================] - 0s 82us/sample - loss: 145686789.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2419/5000
    500/500 [==============================] - 0s 80us/sample - loss: -118950126.0120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2420/5000
    500/500 [==============================] - 0s 84us/sample - loss: -53065302.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2421/5000
    500/500 [==============================] - 0s 82us/sample - loss: -16535706.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2422/5000
    500/500 [==============================] - 0s 86us/sample - loss: 32861299.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2423/5000
    500/500 [==============================] - 0s 84us/sample - loss: 14648816.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2424/5000
    500/500 [==============================] - 0s 80us/sample - loss: -13650083.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2425/5000
    500/500 [==============================] - 0s 88us/sample - loss: -124912382.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2426/5000
    500/500 [==============================] - 0s 92us/sample - loss: 34677734.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2427/5000
    500/500 [==============================] - 0s 84us/sample - loss: -82849290.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2428/5000
    500/500 [==============================] - 0s 86us/sample - loss: 133340890.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2429/5000
    500/500 [==============================] - 0s 94us/sample - loss: 22626762.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2430/5000
    500/500 [==============================] - 0s 86us/sample - loss: -19349807.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2431/5000
    500/500 [==============================] - 0s 86us/sample - loss: -83899274.2440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2432/5000
    500/500 [==============================] - 0s 78us/sample - loss: 19860796.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2433/5000
    500/500 [==============================] - 0s 88us/sample - loss: 39262385.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2434/5000
    500/500 [==============================] - 0s 76us/sample - loss: -91300795.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2435/5000
    500/500 [==============================] - 0s 84us/sample - loss: -37787037.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2436/5000
    500/500 [==============================] - 0s 84us/sample - loss: -7034292.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2437/5000
    500/500 [==============================] - 0s 84us/sample - loss: 108924360.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2438/5000
    500/500 [==============================] - 0s 90us/sample - loss: -199947314.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2439/5000
    500/500 [==============================] - 0s 80us/sample - loss: 27794454.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2440/5000
    500/500 [==============================] - 0s 86us/sample - loss: -121351481.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2441/5000
    500/500 [==============================] - 0s 88us/sample - loss: 134462101.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2442/5000
    500/500 [==============================] - 0s 84us/sample - loss: 23293853.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2443/5000
    500/500 [==============================] - 0s 82us/sample - loss: 142760473.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2444/5000
    500/500 [==============================] - 0s 92us/sample - loss: 25748275.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2445/5000
    500/500 [==============================] - 0s 76us/sample - loss: 9215631.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2446/5000
    500/500 [==============================] - 0s 90us/sample - loss: 122930028.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2447/5000
    500/500 [==============================] - 0s 86us/sample - loss: -79943385.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2448/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32141705.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2449/5000
    500/500 [==============================] - 0s 82us/sample - loss: -167301304.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2450/5000
    500/500 [==============================] - 0s 84us/sample - loss: 147688078.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2451/5000
    500/500 [==============================] - 0s 77us/sample - loss: -64497347.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2452/5000
    500/500 [==============================] - 0s 80us/sample - loss: 50612080.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2453/5000
    500/500 [==============================] - 0s 84us/sample - loss: 107859828.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2454/5000
    500/500 [==============================] - 0s 90us/sample - loss: -60471783.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2455/5000
    500/500 [==============================] - 0s 78us/sample - loss: -54413766.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2456/5000
    500/500 [==============================] - 0s 82us/sample - loss: -6919446.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2457/5000
    500/500 [==============================] - 0s 82us/sample - loss: 49672559.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2458/5000
    500/500 [==============================] - 0s 88us/sample - loss: 134017832.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2459/5000
    500/500 [==============================] - 0s 94us/sample - loss: 29546178.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2460/5000
    500/500 [==============================] - 0s 76us/sample - loss: 137877448.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2461/5000
    500/500 [==============================] - 0s 77us/sample - loss: -31168676.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2462/5000
    500/500 [==============================] - 0s 90us/sample - loss: -43106907.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2463/5000
    500/500 [==============================] - 0s 76us/sample - loss: 164554130.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2464/5000
    500/500 [==============================] - 0s 86us/sample - loss: 90316944.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2465/5000
    500/500 [==============================] - 0s 150us/sample - loss: 34723767.0360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2466/5000
    500/500 [==============================] - 0s 112us/sample - loss: 49272360.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2467/5000
    500/500 [==============================] - 0s 76us/sample - loss: 50431972.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2468/5000
    500/500 [==============================] - 0s 90us/sample - loss: -91568743.9720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2469/5000
    500/500 [==============================] - 0s 86us/sample - loss: 133597943.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2470/5000
    500/500 [==============================] - 0s 82us/sample - loss: -145398098.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2471/5000
    500/500 [==============================] - 0s 82us/sample - loss: -190161739.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2472/5000
    500/500 [==============================] - 0s 84us/sample - loss: 90316435.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2473/5000
    500/500 [==============================] - 0s 104us/sample - loss: 41698590.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2474/5000
    500/500 [==============================] - 0s 104us/sample - loss: -14928924.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2475/5000
    500/500 [==============================] - 0s 102us/sample - loss: 108490517.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2476/5000
    500/500 [==============================] - 0s 104us/sample - loss: -133298459.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2477/5000
    500/500 [==============================] - 0s 102us/sample - loss: 78411217.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2478/5000
    500/500 [==============================] - 0s 106us/sample - loss: 150847486.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2479/5000
    500/500 [==============================] - 0s 104us/sample - loss: -54951012.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2480/5000
    500/500 [==============================] - 0s 90us/sample - loss: -26040848.1921 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2481/5000
    500/500 [==============================] - 0s 98us/sample - loss: 6487260.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2482/5000
    500/500 [==============================] - 0s 98us/sample - loss: -94677017.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2483/5000
    500/500 [==============================] - 0s 118us/sample - loss: 83874434.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2484/5000
    500/500 [==============================] - 0s 102us/sample - loss: -79537226.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2485/5000
    500/500 [==============================] - 0s 104us/sample - loss: -91139594.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2486/5000
    500/500 [==============================] - 0s 97us/sample - loss: 85216333.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2487/5000
    500/500 [==============================] - 0s 88us/sample - loss: 49747194.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2488/5000
    500/500 [==============================] - 0s 98us/sample - loss: 46217927.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2489/5000
    500/500 [==============================] - 0s 94us/sample - loss: 58438365.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2490/5000
    500/500 [==============================] - 0s 104us/sample - loss: -271678255.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2491/5000
    500/500 [==============================] - 0s 104us/sample - loss: 128783524.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2492/5000
    500/500 [==============================] - 0s 92us/sample - loss: -178332774.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2493/5000
    500/500 [==============================] - 0s 93us/sample - loss: -145456015.3640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2494/5000
    500/500 [==============================] - 0s 92us/sample - loss: 82369306.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2495/5000
    500/500 [==============================] - 0s 92us/sample - loss: -117550539.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2496/5000
    500/500 [==============================] - 0s 91us/sample - loss: 217426994.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2497/5000
    500/500 [==============================] - 0s 110us/sample - loss: -49804786.4280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2498/5000
    500/500 [==============================] - 0s 98us/sample - loss: -70105041.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2499/5000
    500/500 [==============================] - 0s 100us/sample - loss: 37533602.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2500/5000
    500/500 [==============================] - 0s 91us/sample - loss: -84010037.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2501/5000
    500/500 [==============================] - 0s 86us/sample - loss: 13296206.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2502/5000
    500/500 [==============================] - 0s 91us/sample - loss: -190218167.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2503/5000
    500/500 [==============================] - 0s 91us/sample - loss: -16056381.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2504/5000
    500/500 [==============================] - 0s 104us/sample - loss: -25516212.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2505/5000
    500/500 [==============================] - 0s 98us/sample - loss: 45763473.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2506/5000
    500/500 [==============================] - 0s 91us/sample - loss: -11936013.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2507/5000
    500/500 [==============================] - 0s 100us/sample - loss: -48543330.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2508/5000
    500/500 [==============================] - 0s 90us/sample - loss: -39778223.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2509/5000
    500/500 [==============================] - 0s 106us/sample - loss: 80934927.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2510/5000
    500/500 [==============================] - 0s 104us/sample - loss: -70737200.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2511/5000
    500/500 [==============================] - 0s 99us/sample - loss: -118182699.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2512/5000
    500/500 [==============================] - 0s 98us/sample - loss: -15346476.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2513/5000
    500/500 [==============================] - 0s 78us/sample - loss: -58093564.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2514/5000
    500/500 [==============================] - 0s 96us/sample - loss: -85847690.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2515/5000
    500/500 [==============================] - 0s 79us/sample - loss: -11385036.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2516/5000
    500/500 [==============================] - 0s 88us/sample - loss: 35773094.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2517/5000
    500/500 [==============================] - 0s 78us/sample - loss: 62860759.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2518/5000
    500/500 [==============================] - 0s 86us/sample - loss: 1710153.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2519/5000
    500/500 [==============================] - 0s 86us/sample - loss: -18104972.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2520/5000
    500/500 [==============================] - 0s 86us/sample - loss: 34875368.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2521/5000
    500/500 [==============================] - 0s 95us/sample - loss: 92924184.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2522/5000
    500/500 [==============================] - 0s 86us/sample - loss: 71324737.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2523/5000
    500/500 [==============================] - 0s 78us/sample - loss: 137086849.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2524/5000
    500/500 [==============================] - 0s 92us/sample - loss: 54056180.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2525/5000
    500/500 [==============================] - 0s 92us/sample - loss: 97724873.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2526/5000
    500/500 [==============================] - 0s 82us/sample - loss: 72585492.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2527/5000
    500/500 [==============================] - 0s 86us/sample - loss: 17053598.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2528/5000
    500/500 [==============================] - 0s 100us/sample - loss: -15888997.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2529/5000
    500/500 [==============================] - 0s 96us/sample - loss: -91636517.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2530/5000
    500/500 [==============================] - 0s 90us/sample - loss: -42652659.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2531/5000
    500/500 [==============================] - 0s 80us/sample - loss: -121282913.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2532/5000
    500/500 [==============================] - 0s 78us/sample - loss: 41740006.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2533/5000
    500/500 [==============================] - 0s 82us/sample - loss: 163736466.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2534/5000
    500/500 [==============================] - 0s 82us/sample - loss: -123758627.2300 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2535/5000
    500/500 [==============================] - 0s 90us/sample - loss: 16130244.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2536/5000
    500/500 [==============================] - 0s 78us/sample - loss: -46716288.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2537/5000
    500/500 [==============================] - 0s 73us/sample - loss: 65448992.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2538/5000
    500/500 [==============================] - 0s 82us/sample - loss: -150666791.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2539/5000
    500/500 [==============================] - 0s 82us/sample - loss: 72611988.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2540/5000
    500/500 [==============================] - 0s 84us/sample - loss: 31321801.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2541/5000
    500/500 [==============================] - 0s 78us/sample - loss: -231391721.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2542/5000
    500/500 [==============================] - 0s 100us/sample - loss: 31701520.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2543/5000
    500/500 [==============================] - 0s 94us/sample - loss: 38512878.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2544/5000
    500/500 [==============================] - 0s 86us/sample - loss: -45213548.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2545/5000
    500/500 [==============================] - 0s 84us/sample - loss: 228183292.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2546/5000
    500/500 [==============================] - 0s 88us/sample - loss: 20118687.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2547/5000
    500/500 [==============================] - 0s 90us/sample - loss: 85137985.5400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2548/5000
    500/500 [==============================] - 0s 88us/sample - loss: -63798830.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2549/5000
    500/500 [==============================] - 0s 80us/sample - loss: 11199914.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2550/5000
    500/500 [==============================] - 0s 88us/sample - loss: -68100161.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2551/5000
    500/500 [==============================] - 0s 88us/sample - loss: -138153415.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2552/5000
    500/500 [==============================] - 0s 84us/sample - loss: -86453687.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2553/5000
    500/500 [==============================] - 0s 94us/sample - loss: -90368814.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2554/5000
    500/500 [==============================] - 0s 96us/sample - loss: 96367288.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2555/5000
    500/500 [==============================] - 0s 94us/sample - loss: -69360547.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2556/5000
    500/500 [==============================] - 0s 90us/sample - loss: 150232550.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2557/5000
    500/500 [==============================] - 0s 78us/sample - loss: 79785056.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2558/5000
    500/500 [==============================] - 0s 92us/sample - loss: -80406933.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2559/5000
    500/500 [==============================] - 0s 90us/sample - loss: -94384837.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2560/5000
    500/500 [==============================] - 0s 88us/sample - loss: -45043219.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2561/5000
    500/500 [==============================] - 0s 101us/sample - loss: -40278764.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2562/5000
    500/500 [==============================] - 0s 84us/sample - loss: 55449248.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2563/5000
    500/500 [==============================] - 0s 84us/sample - loss: 10705118.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2564/5000
    500/500 [==============================] - 0s 94us/sample - loss: 32805764.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2565/5000
    500/500 [==============================] - 0s 94us/sample - loss: -27441432.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2566/5000
    500/500 [==============================] - 0s 92us/sample - loss: -4456455.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2567/5000
    500/500 [==============================] - 0s 80us/sample - loss: 34973015.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2568/5000
    500/500 [==============================] - 0s 92us/sample - loss: -11711817.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2569/5000
    500/500 [==============================] - 0s 98us/sample - loss: 144770799.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2570/5000
    500/500 [==============================] - 0s 92us/sample - loss: 52514005.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2571/5000
    500/500 [==============================] - 0s 92us/sample - loss: -33912900.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2572/5000
    500/500 [==============================] - 0s 80us/sample - loss: -53811603.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2573/5000
    500/500 [==============================] - 0s 80us/sample - loss: -58152824.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2574/5000
    500/500 [==============================] - 0s 82us/sample - loss: 150967359.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2575/5000
    500/500 [==============================] - 0s 80us/sample - loss: -64357543.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2576/5000
    500/500 [==============================] - 0s 90us/sample - loss: -71708910.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2577/5000
    500/500 [==============================] - 0s 88us/sample - loss: -880899.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2578/5000
    500/500 [==============================] - 0s 80us/sample - loss: 15365067.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2579/5000
    500/500 [==============================] - 0s 90us/sample - loss: 81250230.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2580/5000
    500/500 [==============================] - 0s 78us/sample - loss: 23009486.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2581/5000
    500/500 [==============================] - 0s 78us/sample - loss: 178047153.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2582/5000
    500/500 [==============================] - 0s 82us/sample - loss: -50451800.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2583/5000
    500/500 [==============================] - 0s 86us/sample - loss: -156211212.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2584/5000
    500/500 [==============================] - 0s 88us/sample - loss: 120732784.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2585/5000
    500/500 [==============================] - 0s 80us/sample - loss: -78595336.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2586/5000
    500/500 [==============================] - 0s 88us/sample - loss: 11424416.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2587/5000
    500/500 [==============================] - 0s 78us/sample - loss: 59555491.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2588/5000
    500/500 [==============================] - 0s 78us/sample - loss: -15278124.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2589/5000
    500/500 [==============================] - 0s 76us/sample - loss: 138125610.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2590/5000
    500/500 [==============================] - 0s 82us/sample - loss: 26042724.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2591/5000
    500/500 [==============================] - 0s 80us/sample - loss: 188425331.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2592/5000
    500/500 [==============================] - 0s 90us/sample - loss: 187760202.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2593/5000
    500/500 [==============================] - 0s 76us/sample - loss: 56565272.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2594/5000
    500/500 [==============================] - 0s 79us/sample - loss: -99102424.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2595/5000
    500/500 [==============================] - 0s 76us/sample - loss: -141872180.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2596/5000
    500/500 [==============================] - 0s 80us/sample - loss: 43809537.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2597/5000
    500/500 [==============================] - 0s 80us/sample - loss: -85410870.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2598/5000
    500/500 [==============================] - 0s 90us/sample - loss: 48353634.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2599/5000
    500/500 [==============================] - 0s 90us/sample - loss: 47441326.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2600/5000
    500/500 [==============================] - 0s 76us/sample - loss: -39544882.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2601/5000
    500/500 [==============================] - 0s 94us/sample - loss: 14831253.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2602/5000
    500/500 [==============================] - 0s 90us/sample - loss: 80622942.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2603/5000
    500/500 [==============================] - 0s 96us/sample - loss: 38136942.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2604/5000
    500/500 [==============================] - 0s 78us/sample - loss: 77757299.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2605/5000
    500/500 [==============================] - 0s 79us/sample - loss: 108379976.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2606/5000
    500/500 [==============================] - 0s 92us/sample - loss: -176783114.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2607/5000
    500/500 [==============================] - 0s 80us/sample - loss: 48769543.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2608/5000
    500/500 [==============================] - 0s 86us/sample - loss: -147844249.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2609/5000
    500/500 [==============================] - 0s 98us/sample - loss: 27578290.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2610/5000
    500/500 [==============================] - 0s 80us/sample - loss: 24158586.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2611/5000
    500/500 [==============================] - 0s 79us/sample - loss: 114407353.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2612/5000
    500/500 [==============================] - 0s 82us/sample - loss: -19299515.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2613/5000
    500/500 [==============================] - 0s 84us/sample - loss: 35323003.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2614/5000
    500/500 [==============================] - 0s 84us/sample - loss: -20370924.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2615/5000
    500/500 [==============================] - 0s 82us/sample - loss: 83073582.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2616/5000
    500/500 [==============================] - 0s 82us/sample - loss: -93162187.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2617/5000
    500/500 [==============================] - 0s 80us/sample - loss: 106567602.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2618/5000
    500/500 [==============================] - 0s 88us/sample - loss: 200444568.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2619/5000
    500/500 [==============================] - 0s 90us/sample - loss: -20907635.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2620/5000
    500/500 [==============================] - 0s 74us/sample - loss: -4928202.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2621/5000
    500/500 [==============================] - 0s 82us/sample - loss: -16466335.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2622/5000
    500/500 [==============================] - 0s 84us/sample - loss: -19110893.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2623/5000
    500/500 [==============================] - 0s 88us/sample - loss: -136662767.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2624/5000
    500/500 [==============================] - 0s 90us/sample - loss: -202925370.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2625/5000
    500/500 [==============================] - 0s 82us/sample - loss: 25635005.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2626/5000
    500/500 [==============================] - 0s 92us/sample - loss: 121939218.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2627/5000
    500/500 [==============================] - 0s 92us/sample - loss: -37353311.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2628/5000
    500/500 [==============================] - 0s 94us/sample - loss: 146164972.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2629/5000
    500/500 [==============================] - 0s 88us/sample - loss: -7799337.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2630/5000
    500/500 [==============================] - 0s 96us/sample - loss: -55532311.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2631/5000
    500/500 [==============================] - 0s 76us/sample - loss: -83151496.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2632/5000
    500/500 [==============================] - 0s 90us/sample - loss: -116270714.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2633/5000
    500/500 [==============================] - 0s 94us/sample - loss: 134919180.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2634/5000
    500/500 [==============================] - 0s 88us/sample - loss: -53760542.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2635/5000
    500/500 [==============================] - 0s 86us/sample - loss: 92691569.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2636/5000
    500/500 [==============================] - 0s 90us/sample - loss: -84636892.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2637/5000
    500/500 [==============================] - 0s 94us/sample - loss: -21747766.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2638/5000
    500/500 [==============================] - 0s 92us/sample - loss: 159208753.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2639/5000
    500/500 [==============================] - 0s 92us/sample - loss: 30034340.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2640/5000
    500/500 [==============================] - 0s 90us/sample - loss: -43750238.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2641/5000
    500/500 [==============================] - 0s 90us/sample - loss: 34007640.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2642/5000
    500/500 [==============================] - 0s 92us/sample - loss: -183642937.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2643/5000
    500/500 [==============================] - 0s 88us/sample - loss: 110623870.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2644/5000
    500/500 [==============================] - 0s 90us/sample - loss: 6614185.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2645/5000
    500/500 [==============================] - 0s 98us/sample - loss: 63727846.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2646/5000
    500/500 [==============================] - 0s 92us/sample - loss: 171798655.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2647/5000
    500/500 [==============================] - 0s 96us/sample - loss: -3118023.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2648/5000
    500/500 [==============================] - 0s 94us/sample - loss: 101166418.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2649/5000
    500/500 [==============================] - 0s 90us/sample - loss: 44065546.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2650/5000
    500/500 [==============================] - 0s 98us/sample - loss: -16794050.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2651/5000
    500/500 [==============================] - 0s 84us/sample - loss: 72556385.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2652/5000
    500/500 [==============================] - 0s 94us/sample - loss: 210136363.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2653/5000
    500/500 [==============================] - 0s 90us/sample - loss: 198296251.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2654/5000
    500/500 [==============================] - 0s 126us/sample - loss: -244719857.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2655/5000
    500/500 [==============================] - 0s 94us/sample - loss: 106727389.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2656/5000
    500/500 [==============================] - 0s 45us/sample - loss: -29980571.2810 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2657/5000
    500/500 [==============================] - 0s 89us/sample - loss: 46465585.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2658/5000
    500/500 [==============================] - 0s 68us/sample - loss: 62636606.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2659/5000
    500/500 [==============================] - 0s 37us/sample - loss: 7256380.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2660/5000
    500/500 [==============================] - 0s 62us/sample - loss: -21936985.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2661/5000
    500/500 [==============================] - 0s 67us/sample - loss: -43216652.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2662/5000
    500/500 [==============================] - 0s 67us/sample - loss: 103153551.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2663/5000
    500/500 [==============================] - 0s 72us/sample - loss: -95456204.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2664/5000
    500/500 [==============================] - 0s 64us/sample - loss: -114087689.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2665/5000
    500/500 [==============================] - 0s 94us/sample - loss: 159504372.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2666/5000
    500/500 [==============================] - 0s 71us/sample - loss: -94348643.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2667/5000
    500/500 [==============================] - 0s 58us/sample - loss: -15364766.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2668/5000
    500/500 [==============================] - 0s 102us/sample - loss: -52967652.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2669/5000
    500/500 [==============================] - 0s 116us/sample - loss: 108537370.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2670/5000
    500/500 [==============================] - 0s 120us/sample - loss: -63526749.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2671/5000
    500/500 [==============================] - 0s 102us/sample - loss: -127759079.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2672/5000
    500/500 [==============================] - 0s 98us/sample - loss: 12488893.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2673/5000
    500/500 [==============================] - 0s 78us/sample - loss: -139598878.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2674/5000
    500/500 [==============================] - 0s 76us/sample - loss: -148158453.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2675/5000
    500/500 [==============================] - 0s 76us/sample - loss: 41500266.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2676/5000
    500/500 [==============================] - 0s 77us/sample - loss: 192397911.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2677/5000
    500/500 [==============================] - 0s 84us/sample - loss: -236777396.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2678/5000
    500/500 [==============================] - 0s 88us/sample - loss: -42626197.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2679/5000
    500/500 [==============================] - 0s 86us/sample - loss: 143173537.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2680/5000
    500/500 [==============================] - 0s 84us/sample - loss: 76547748.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2681/5000
    500/500 [==============================] - 0s 81us/sample - loss: -23624876.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2682/5000
    500/500 [==============================] - 0s 106us/sample - loss: -46494531.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2683/5000
    500/500 [==============================] - 0s 90us/sample - loss: 40211466.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2684/5000
    500/500 [==============================] - 0s 90us/sample - loss: -50008953.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2685/5000
    500/500 [==============================] - 0s 76us/sample - loss: -104287173.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2686/5000
    500/500 [==============================] - 0s 92us/sample - loss: -75061030.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2687/5000
    500/500 [==============================] - 0s 78us/sample - loss: -54775533.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2688/5000
    500/500 [==============================] - 0s 77us/sample - loss: -19913411.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2689/5000
    500/500 [==============================] - 0s 90us/sample - loss: -59930754.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2690/5000
    500/500 [==============================] - 0s 82us/sample - loss: 48974499.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2691/5000
    500/500 [==============================] - 0s 88us/sample - loss: 79985804.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2692/5000
    500/500 [==============================] - 0s 80us/sample - loss: 83694421.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2693/5000
    500/500 [==============================] - 0s 80us/sample - loss: -28992597.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2694/5000
    500/500 [==============================] - 0s 90us/sample - loss: 80316541.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2695/5000
    500/500 [==============================] - 0s 80us/sample - loss: -48253239.4280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2696/5000
    500/500 [==============================] - 0s 84us/sample - loss: -27094496.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2697/5000
    500/500 [==============================] - 0s 82us/sample - loss: -30219742.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2698/5000
    500/500 [==============================] - 0s 78us/sample - loss: 6282924.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2699/5000
    500/500 [==============================] - 0s 80us/sample - loss: -73670994.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2700/5000
    500/500 [==============================] - 0s 80us/sample - loss: 2486786.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2701/5000
    500/500 [==============================] - 0s 80us/sample - loss: 24226555.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2702/5000
    500/500 [==============================] - 0s 80us/sample - loss: 29864748.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2703/5000
    500/500 [==============================] - 0s 82us/sample - loss: 57472063.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2704/5000
    500/500 [==============================] - 0s 80us/sample - loss: -88189343.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2705/5000
    500/500 [==============================] - 0s 80us/sample - loss: 78573455.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2706/5000
    500/500 [==============================] - 0s 80us/sample - loss: -92769075.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2707/5000
    500/500 [==============================] - 0s 86us/sample - loss: 5714908.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2708/5000
    500/500 [==============================] - 0s 82us/sample - loss: -7121276.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2709/5000
    500/500 [==============================] - 0s 86us/sample - loss: 101609311.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2710/5000
    500/500 [==============================] - 0s 98us/sample - loss: 10593633.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2711/5000
    500/500 [==============================] - 0s 88us/sample - loss: 220970052.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2712/5000
    500/500 [==============================] - 0s 82us/sample - loss: -4933751.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2713/5000
    500/500 [==============================] - 0s 90us/sample - loss: 200711037.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2714/5000
    500/500 [==============================] - 0s 96us/sample - loss: 80163232.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2715/5000
    500/500 [==============================] - 0s 92us/sample - loss: 84272124.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2716/5000
    500/500 [==============================] - 0s 80us/sample - loss: -46125898.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2717/5000
    500/500 [==============================] - 0s 82us/sample - loss: 71431696.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2718/5000
    500/500 [==============================] - 0s 86us/sample - loss: -24032603.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2719/5000
    500/500 [==============================] - 0s 88us/sample - loss: 14456293.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2720/5000
    500/500 [==============================] - 0s 84us/sample - loss: 150365229.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2721/5000
    500/500 [==============================] - 0s 86us/sample - loss: -25744372.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2722/5000
    500/500 [==============================] - 0s 84us/sample - loss: -55951388.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2723/5000
    500/500 [==============================] - 0s 92us/sample - loss: -48377669.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2724/5000
    500/500 [==============================] - 0s 90us/sample - loss: -114537297.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2725/5000
    500/500 [==============================] - 0s 90us/sample - loss: -1837922.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2726/5000
    500/500 [==============================] - 0s 80us/sample - loss: -6646866.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2727/5000
    500/500 [==============================] - 0s 82us/sample - loss: -52398322.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2728/5000
    500/500 [==============================] - 0s 82us/sample - loss: -48048458.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2729/5000
    500/500 [==============================] - 0s 81us/sample - loss: 135838634.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2730/5000
    500/500 [==============================] - 0s 74us/sample - loss: -118971714.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2731/5000
    500/500 [==============================] - 0s 80us/sample - loss: -59116337.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2732/5000
    500/500 [==============================] - 0s 80us/sample - loss: -88180014.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2733/5000
    500/500 [==============================] - 0s 78us/sample - loss: -161739461.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2734/5000
    500/500 [==============================] - 0s 84us/sample - loss: 39332130.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2735/5000
    500/500 [==============================] - 0s 78us/sample - loss: -27229604.0020 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2736/5000
    500/500 [==============================] - 0s 84us/sample - loss: -20865527.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2737/5000
    500/500 [==============================] - 0s 84us/sample - loss: 57185157.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2738/5000
    500/500 [==============================] - 0s 88us/sample - loss: -192583944.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2739/5000
    500/500 [==============================] - 0s 84us/sample - loss: -59744532.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2740/5000
    500/500 [==============================] - 0s 80us/sample - loss: -78836175.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2741/5000
    500/500 [==============================] - 0s 78us/sample - loss: -26233461.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2742/5000
    500/500 [==============================] - 0s 84us/sample - loss: 97382487.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2743/5000
    500/500 [==============================] - 0s 76us/sample - loss: -53945478.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2744/5000
    500/500 [==============================] - 0s 78us/sample - loss: -136388110.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2745/5000
    500/500 [==============================] - 0s 84us/sample - loss: 148131119.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2746/5000
    500/500 [==============================] - 0s 80us/sample - loss: 31624518.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2747/5000
    500/500 [==============================] - 0s 88us/sample - loss: 183987152.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2748/5000
    500/500 [==============================] - 0s 82us/sample - loss: -39068674.3000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2749/5000
    500/500 [==============================] - 0s 82us/sample - loss: -38350223.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2750/5000
    500/500 [==============================] - 0s 84us/sample - loss: -60791496.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2751/5000
    500/500 [==============================] - 0s 82us/sample - loss: 45159202.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2752/5000
    500/500 [==============================] - 0s 82us/sample - loss: -95241616.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2753/5000
    500/500 [==============================] - 0s 80us/sample - loss: 259816235.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2754/5000
    500/500 [==============================] - 0s 96us/sample - loss: 6597496.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2755/5000
    500/500 [==============================] - 0s 74us/sample - loss: 20295.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2756/5000
    500/500 [==============================] - 0s 74us/sample - loss: -214185730.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2757/5000
    500/500 [==============================] - 0s 82us/sample - loss: 17015961.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2758/5000
    500/500 [==============================] - 0s 94us/sample - loss: -194908304.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2759/5000
    500/500 [==============================] - 0s 90us/sample - loss: -110710874.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2760/5000
    500/500 [==============================] - 0s 84us/sample - loss: -27146293.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2761/5000
    500/500 [==============================] - 0s 84us/sample - loss: -47120319.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2762/5000
    500/500 [==============================] - 0s 88us/sample - loss: 75456133.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2763/5000
    500/500 [==============================] - 0s 96us/sample - loss: -112729375.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2764/5000
    500/500 [==============================] - 0s 92us/sample - loss: -193513520.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2765/5000
    500/500 [==============================] - 0s 79us/sample - loss: 74427148.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2766/5000
    500/500 [==============================] - 0s 92us/sample - loss: 87309738.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2767/5000
    500/500 [==============================] - 0s 92us/sample - loss: 26361981.1842 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2768/5000
    500/500 [==============================] - 0s 94us/sample - loss: 78717730.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2769/5000
    500/500 [==============================] - 0s 82us/sample - loss: -66523484.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2770/5000
    500/500 [==============================] - 0s 78us/sample - loss: -29997233.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2771/5000
    500/500 [==============================] - 0s 96us/sample - loss: -74216572.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2772/5000
    500/500 [==============================] - 0s 94us/sample - loss: -122652057.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2773/5000
    500/500 [==============================] - 0s 92us/sample - loss: -277699659.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2774/5000
    500/500 [==============================] - 0s 90us/sample - loss: 14683441.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2775/5000
    500/500 [==============================] - 0s 98us/sample - loss: -5478131.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2776/5000
    500/500 [==============================] - 0s 96us/sample - loss: -49154867.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2777/5000
    500/500 [==============================] - 0s 94us/sample - loss: 57125362.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2778/5000
    500/500 [==============================] - 0s 102us/sample - loss: -95931545.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2779/5000
    500/500 [==============================] - 0s 94us/sample - loss: 85301330.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2780/5000
    500/500 [==============================] - 0s 102us/sample - loss: -11666348.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2781/5000
    500/500 [==============================] - 0s 100us/sample - loss: -30057838.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2782/5000
    500/500 [==============================] - 0s 98us/sample - loss: 120243865.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2783/5000
    500/500 [==============================] - 0s 102us/sample - loss: 33696085.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2784/5000
    500/500 [==============================] - 0s 86us/sample - loss: 105407493.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2785/5000
    500/500 [==============================] - 0s 98us/sample - loss: 146701896.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2786/5000
    500/500 [==============================] - 0s 100us/sample - loss: 156237699.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2787/5000
    500/500 [==============================] - 0s 98us/sample - loss: -51563011.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2788/5000
    500/500 [==============================] - 0s 88us/sample - loss: -34623734.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2789/5000
    500/500 [==============================] - 0s 92us/sample - loss: 87456949.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2790/5000
    500/500 [==============================] - 0s 96us/sample - loss: -152940918.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2791/5000
    500/500 [==============================] - 0s 96us/sample - loss: 61034957.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2792/5000
    500/500 [==============================] - 0s 94us/sample - loss: -19623282.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2793/5000
    500/500 [==============================] - 0s 88us/sample - loss: -12520613.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2794/5000
    500/500 [==============================] - 0s 98us/sample - loss: 187512403.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2795/5000
    500/500 [==============================] - 0s 102us/sample - loss: 33723175.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2796/5000
    500/500 [==============================] - 0s 98us/sample - loss: -154106554.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2797/5000
    500/500 [==============================] - 0s 106us/sample - loss: -40525865.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2798/5000
    500/500 [==============================] - 0s 102us/sample - loss: -11248997.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2799/5000
    500/500 [==============================] - 0s 94us/sample - loss: 106810022.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2800/5000
    500/500 [==============================] - 0s 88us/sample - loss: 28428908.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2801/5000
    500/500 [==============================] - 0s 96us/sample - loss: 21738889.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2802/5000
    500/500 [==============================] - 0s 84us/sample - loss: 32958782.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2803/5000
    500/500 [==============================] - 0s 82us/sample - loss: 186615935.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2804/5000
    500/500 [==============================] - 0s 94us/sample - loss: 40511734.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2805/5000
    500/500 [==============================] - 0s 92us/sample - loss: 10296269.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2806/5000
    500/500 [==============================] - 0s 80us/sample - loss: 37849736.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2807/5000
    500/500 [==============================] - 0s 98us/sample - loss: -86955218.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2808/5000
    500/500 [==============================] - 0s 94us/sample - loss: -181896792.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2809/5000
    500/500 [==============================] - 0s 96us/sample - loss: 85134184.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2810/5000
    500/500 [==============================] - 0s 94us/sample - loss: 73079518.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2811/5000
    500/500 [==============================] - 0s 94us/sample - loss: 52646586.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2812/5000
    500/500 [==============================] - 0s 96us/sample - loss: -84035655.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2813/5000
    500/500 [==============================] - 0s 92us/sample - loss: -90178334.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2814/5000
    500/500 [==============================] - 0s 78us/sample - loss: 63968927.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2815/5000
    500/500 [==============================] - 0s 76us/sample - loss: -34415571.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2816/5000
    500/500 [==============================] - 0s 88us/sample - loss: 23979452.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2817/5000
    500/500 [==============================] - 0s 86us/sample - loss: -39199903.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2818/5000
    500/500 [==============================] - 0s 86us/sample - loss: 34258239.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2819/5000
    500/500 [==============================] - 0s 78us/sample - loss: 17592390.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2820/5000
    500/500 [==============================] - 0s 86us/sample - loss: 28324964.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2821/5000
    500/500 [==============================] - 0s 92us/sample - loss: -88209683.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2822/5000
    500/500 [==============================] - 0s 80us/sample - loss: 130350719.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2823/5000
    500/500 [==============================] - 0s 78us/sample - loss: 72535957.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2824/5000
    500/500 [==============================] - 0s 82us/sample - loss: 51623988.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2825/5000
    500/500 [==============================] - 0s 80us/sample - loss: -170052664.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2826/5000
    500/500 [==============================] - 0s 82us/sample - loss: -11525524.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2827/5000
    500/500 [==============================] - 0s 85us/sample - loss: 3130373.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2828/5000
    500/500 [==============================] - 0s 126us/sample - loss: 32203984.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2829/5000
    500/500 [==============================] - 0s 120us/sample - loss: 141069713.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2830/5000
    500/500 [==============================] - 0s 92us/sample - loss: 33731100.4340 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2831/5000
    500/500 [==============================] - 0s 100us/sample - loss: 56157707.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2832/5000
    500/500 [==============================] - 0s 102us/sample - loss: -94298337.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2833/5000
    500/500 [==============================] - 0s 100us/sample - loss: -231238465.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2834/5000
    500/500 [==============================] - 0s 100us/sample - loss: 82606442.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2835/5000
    500/500 [==============================] - 0s 94us/sample - loss: -149611459.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2836/5000
    500/500 [==============================] - 0s 86us/sample - loss: 87466564.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2837/5000
    500/500 [==============================] - 0s 104us/sample - loss: 6475237.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2838/5000
    500/500 [==============================] - 0s 100us/sample - loss: -54490211.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2839/5000
    500/500 [==============================] - 0s 100us/sample - loss: 67465623.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2840/5000
    500/500 [==============================] - 0s 95us/sample - loss: 140998426.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2841/5000
    500/500 [==============================] - 0s 86us/sample - loss: 35424492.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2842/5000
    500/500 [==============================] - 0s 104us/sample - loss: -70276180.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2843/5000
    500/500 [==============================] - 0s 94us/sample - loss: 49676226.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2844/5000
    500/500 [==============================] - 0s 98us/sample - loss: 246024648.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2845/5000
    500/500 [==============================] - 0s 96us/sample - loss: -99408881.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2846/5000
    500/500 [==============================] - 0s 108us/sample - loss: -26832139.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2847/5000
    500/500 [==============================] - 0s 122us/sample - loss: -1415256.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2848/5000
    500/500 [==============================] - 0s 110us/sample - loss: 202117840.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2849/5000
    500/500 [==============================] - 0s 115us/sample - loss: 88939583.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2850/5000
    500/500 [==============================] - 0s 116us/sample - loss: -153914645.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2851/5000
    500/500 [==============================] - 0s 108us/sample - loss: 12972534.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2852/5000
    500/500 [==============================] - 0s 104us/sample - loss: -34944293.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2853/5000
    500/500 [==============================] - 0s 102us/sample - loss: 120578518.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2854/5000
    500/500 [==============================] - 0s 108us/sample - loss: 11631027.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2855/5000
    500/500 [==============================] - 0s 96us/sample - loss: 49289787.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2856/5000
    500/500 [==============================] - 0s 94us/sample - loss: 23085936.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2857/5000
    500/500 [==============================] - 0s 112us/sample - loss: 119730313.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2858/5000
    500/500 [==============================] - 0s 130us/sample - loss: 28620608.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2859/5000
    500/500 [==============================] - 0s 114us/sample - loss: -44077929.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2860/5000
    500/500 [==============================] - 0s 121us/sample - loss: -156641314.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2861/5000
    500/500 [==============================] - 0s 92us/sample - loss: -102275148.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2862/5000
    500/500 [==============================] - 0s 87us/sample - loss: 13862237.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2863/5000
    500/500 [==============================] - 0s 100us/sample - loss: 150133089.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2864/5000
    500/500 [==============================] - 0s 95us/sample - loss: -66271134.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2865/5000
    500/500 [==============================] - 0s 92us/sample - loss: 17607408.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2866/5000
    500/500 [==============================] - 0s 114us/sample - loss: 215803734.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2867/5000
    500/500 [==============================] - 0s 106us/sample - loss: 210210380.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2868/5000
    500/500 [==============================] - 0s 124us/sample - loss: -61814764.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2869/5000
    500/500 [==============================] - 0s 108us/sample - loss: 88951241.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2870/5000
    500/500 [==============================] - 0s 96us/sample - loss: 203654939.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2871/5000
    500/500 [==============================] - 0s 76us/sample - loss: -173832376.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2872/5000
    500/500 [==============================] - 0s 82us/sample - loss: 60245870.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2873/5000
    500/500 [==============================] - 0s 110us/sample - loss: -123471227.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2874/5000
    500/500 [==============================] - 0s 82us/sample - loss: 30828383.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2875/5000
    500/500 [==============================] - 0s 84us/sample - loss: -121401203.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2876/5000
    500/500 [==============================] - 0s 92us/sample - loss: -62861659.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2877/5000
    500/500 [==============================] - 0s 82us/sample - loss: -109285336.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2878/5000
    500/500 [==============================] - 0s 77us/sample - loss: -65325762.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2879/5000
    500/500 [==============================] - 0s 82us/sample - loss: 147814014.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2880/5000
    500/500 [==============================] - 0s 84us/sample - loss: -125588147.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2881/5000
    500/500 [==============================] - 0s 80us/sample - loss: 99698956.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2882/5000
    500/500 [==============================] - 0s 80us/sample - loss: -29085591.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2883/5000
    500/500 [==============================] - 0s 80us/sample - loss: -41239949.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2884/5000
    500/500 [==============================] - 0s 88us/sample - loss: -75315783.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2885/5000
    500/500 [==============================] - 0s 84us/sample - loss: -14498604.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2886/5000
    500/500 [==============================] - 0s 82us/sample - loss: 101742484.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2887/5000
    500/500 [==============================] - 0s 86us/sample - loss: -35398665.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2888/5000
    500/500 [==============================] - 0s 94us/sample - loss: 68023700.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2889/5000
    500/500 [==============================] - 0s 80us/sample - loss: -67208248.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2890/5000
    500/500 [==============================] - 0s 102us/sample - loss: 95152409.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2891/5000
    500/500 [==============================] - 0s 86us/sample - loss: 140828230.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2892/5000
    500/500 [==============================] - 0s 100us/sample - loss: 35078181.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2893/5000
    500/500 [==============================] - 0s 92us/sample - loss: 60089015.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2894/5000
    500/500 [==============================] - 0s 94us/sample - loss: -82360372.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2895/5000
    500/500 [==============================] - 0s 100us/sample - loss: 67049523.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2896/5000
    500/500 [==============================] - 0s 96us/sample - loss: -78275266.6600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2897/5000
    500/500 [==============================] - 0s 94us/sample - loss: 15920213.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2898/5000
    500/500 [==============================] - 0s 92us/sample - loss: -19630011.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2899/5000
    500/500 [==============================] - 0s 84us/sample - loss: -22935376.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2900/5000
    500/500 [==============================] - 0s 88us/sample - loss: -31211594.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2901/5000
    500/500 [==============================] - 0s 96us/sample - loss: 157612340.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2902/5000
    500/500 [==============================] - 0s 100us/sample - loss: 41249990.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2903/5000
    500/500 [==============================] - 0s 98us/sample - loss: 159587882.2360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2904/5000
    500/500 [==============================] - 0s 88us/sample - loss: -104260880.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2905/5000
    500/500 [==============================] - 0s 90us/sample - loss: -7056171.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2906/5000
    500/500 [==============================] - 0s 94us/sample - loss: -177813028.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2907/5000
    500/500 [==============================] - 0s 90us/sample - loss: 8552881.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2908/5000
    500/500 [==============================] - 0s 180us/sample - loss: 180765394.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2909/5000
    500/500 [==============================] - 0s 126us/sample - loss: -14923672.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2910/5000
    500/500 [==============================] - 0s 101us/sample - loss: -66686606.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2911/5000
    500/500 [==============================] - 0s 87us/sample - loss: -15418762.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2912/5000
    500/500 [==============================] - 0s 96us/sample - loss: -24207060.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2913/5000
    500/500 [==============================] - 0s 88us/sample - loss: 13073599.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2914/5000
    500/500 [==============================] - 0s 90us/sample - loss: 117253656.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2915/5000
    500/500 [==============================] - 0s 88us/sample - loss: 22677243.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2916/5000
    500/500 [==============================] - 0s 84us/sample - loss: -89676269.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2917/5000
    500/500 [==============================] - 0s 88us/sample - loss: -93717436.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2918/5000
    500/500 [==============================] - 0s 90us/sample - loss: 50638933.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2919/5000
    500/500 [==============================] - 0s 82us/sample - loss: -38290924.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2920/5000
    500/500 [==============================] - 0s 88us/sample - loss: 132829624.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2921/5000
    500/500 [==============================] - 0s 92us/sample - loss: -85307999.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2922/5000
    500/500 [==============================] - 0s 92us/sample - loss: -50233205.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2923/5000
    500/500 [==============================] - 0s 96us/sample - loss: 96595862.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2924/5000
    500/500 [==============================] - 0s 84us/sample - loss: 130484898.2360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2925/5000
    500/500 [==============================] - 0s 82us/sample - loss: 65303160.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2926/5000
    500/500 [==============================] - 0s 94us/sample - loss: -107914217.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2927/5000
    500/500 [==============================] - 0s 88us/sample - loss: -165653809.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2928/5000
    500/500 [==============================] - 0s 78us/sample - loss: 40825800.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2929/5000
    500/500 [==============================] - 0s 82us/sample - loss: 71885100.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2930/5000
    500/500 [==============================] - 0s 86us/sample - loss: -186606730.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2931/5000
    500/500 [==============================] - 0s 88us/sample - loss: -89554554.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2932/5000
    500/500 [==============================] - 0s 98us/sample - loss: 134723925.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2933/5000
    500/500 [==============================] - 0s 90us/sample - loss: 215549311.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2934/5000
    500/500 [==============================] - 0s 82us/sample - loss: -137703077.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2935/5000
    500/500 [==============================] - 0s 76us/sample - loss: 43792072.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2936/5000
    500/500 [==============================] - 0s 78us/sample - loss: 77540620.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2937/5000
    500/500 [==============================] - 0s 74us/sample - loss: 4990611.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2938/5000
    500/500 [==============================] - 0s 84us/sample - loss: 17068663.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2939/5000
    500/500 [==============================] - 0s 86us/sample - loss: -39275245.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2940/5000
    500/500 [==============================] - 0s 88us/sample - loss: -60860311.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2941/5000
    500/500 [==============================] - 0s 84us/sample - loss: 35622027.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2942/5000
    500/500 [==============================] - 0s 86us/sample - loss: -191028081.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2943/5000
    500/500 [==============================] - 0s 100us/sample - loss: -7537342.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2944/5000
    500/500 [==============================] - 0s 94us/sample - loss: -9732302.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2945/5000
    500/500 [==============================] - 0s 82us/sample - loss: -28949253.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2946/5000
    500/500 [==============================] - 0s 92us/sample - loss: 213436184.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2947/5000
    500/500 [==============================] - 0s 78us/sample - loss: 26251857.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2948/5000
    500/500 [==============================] - 0s 96us/sample - loss: 84366535.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2949/5000
    500/500 [==============================] - 0s 78us/sample - loss: 103033731.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2950/5000
    500/500 [==============================] - 0s 78us/sample - loss: -41078246.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2951/5000
    500/500 [==============================] - 0s 86us/sample - loss: 38328400.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2952/5000
    500/500 [==============================] - 0s 80us/sample - loss: 45011059.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2953/5000
    500/500 [==============================] - 0s 92us/sample - loss: -63733197.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2954/5000
    500/500 [==============================] - 0s 96us/sample - loss: 151110543.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2955/5000
    500/500 [==============================] - 0s 76us/sample - loss: -20814822.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2956/5000
    500/500 [==============================] - 0s 77us/sample - loss: 114484081.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2957/5000
    500/500 [==============================] - 0s 80us/sample - loss: 15097579.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2958/5000
    500/500 [==============================] - 0s 92us/sample - loss: 39709496.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2959/5000
    500/500 [==============================] - 0s 92us/sample - loss: -17163218.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2960/5000
    500/500 [==============================] - 0s 92us/sample - loss: -33526270.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2961/5000
    500/500 [==============================] - 0s 78us/sample - loss: 90918409.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2962/5000
    500/500 [==============================] - 0s 85us/sample - loss: 51778577.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2963/5000
    500/500 [==============================] - 0s 78us/sample - loss: 74908362.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2964/5000
    500/500 [==============================] - 0s 77us/sample - loss: -126228672.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2965/5000
    500/500 [==============================] - 0s 82us/sample - loss: -32052251.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2966/5000
    500/500 [==============================] - 0s 82us/sample - loss: 165647094.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2967/5000
    500/500 [==============================] - 0s 80us/sample - loss: 90913341.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2968/5000
    500/500 [==============================] - 0s 84us/sample - loss: 26452292.8500 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2969/5000
    500/500 [==============================] - 0s 86us/sample - loss: 9531856.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2970/5000
    500/500 [==============================] - 0s 82us/sample - loss: 107040758.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2971/5000
    500/500 [==============================] - 0s 82us/sample - loss: -5159362.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2972/5000
    500/500 [==============================] - 0s 92us/sample - loss: 29062663.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2973/5000
    500/500 [==============================] - 0s 96us/sample - loss: -137937269.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2974/5000
    500/500 [==============================] - 0s 82us/sample - loss: -76537009.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2975/5000
    500/500 [==============================] - 0s 84us/sample - loss: 81207340.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2976/5000
    500/500 [==============================] - 0s 94us/sample - loss: 131236670.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2977/5000
    500/500 [==============================] - 0s 94us/sample - loss: 55881789.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2978/5000
    500/500 [==============================] - 0s 80us/sample - loss: 8442591.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2979/5000
    500/500 [==============================] - 0s 82us/sample - loss: 128482216.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2980/5000
    500/500 [==============================] - 0s 94us/sample - loss: 41590009.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2981/5000
    500/500 [==============================] - 0s 80us/sample - loss: -132124540.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2982/5000
    500/500 [==============================] - 0s 88us/sample - loss: -41941049.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2983/5000
    500/500 [==============================] - 0s 94us/sample - loss: 41122732.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2984/5000
    500/500 [==============================] - 0s 88us/sample - loss: -19870931.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2985/5000
    500/500 [==============================] - 0s 78us/sample - loss: -7970301.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2986/5000
    500/500 [==============================] - 0s 86us/sample - loss: 9111575.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2987/5000
    500/500 [==============================] - 0s 84us/sample - loss: -79027369.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2988/5000
    500/500 [==============================] - 0s 86us/sample - loss: 131103340.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2989/5000
    500/500 [==============================] - 0s 88us/sample - loss: 88862369.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2990/5000
    500/500 [==============================] - 0s 98us/sample - loss: -8399250.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2991/5000
    500/500 [==============================] - 0s 78us/sample - loss: -72117418.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2992/5000
    500/500 [==============================] - 0s 78us/sample - loss: 99995211.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2993/5000
    500/500 [==============================] - 0s 86us/sample - loss: 63314888.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2994/5000
    500/500 [==============================] - 0s 90us/sample - loss: 41688241.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2995/5000
    500/500 [==============================] - 0s 94us/sample - loss: -64043757.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2996/5000
    500/500 [==============================] - 0s 104us/sample - loss: 210548251.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2997/5000
    500/500 [==============================] - 0s 98us/sample - loss: -187129122.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2998/5000
    500/500 [==============================] - 0s 88us/sample - loss: -7965742.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 2999/5000
    500/500 [==============================] - 0s 96us/sample - loss: -58986587.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3000/5000
    500/500 [==============================] - 0s 86us/sample - loss: 51942738.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3001/5000
    500/500 [==============================] - 0s 80us/sample - loss: 60061652.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3002/5000
    500/500 [==============================] - 0s 88us/sample - loss: 72453710.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3003/5000
    500/500 [==============================] - 0s 92us/sample - loss: -152714064.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3004/5000
    500/500 [==============================] - 0s 80us/sample - loss: 16815682.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3005/5000
    500/500 [==============================] - 0s 80us/sample - loss: 52623838.9800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3006/5000
    500/500 [==============================] - 0s 86us/sample - loss: 60967413.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3007/5000
    500/500 [==============================] - 0s 84us/sample - loss: -81579057.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3008/5000
    500/500 [==============================] - 0s 92us/sample - loss: 35264696.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3009/5000
    500/500 [==============================] - 0s 90us/sample - loss: -77766305.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3010/5000
    500/500 [==============================] - 0s 80us/sample - loss: -131202056.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3011/5000
    500/500 [==============================] - 0s 84us/sample - loss: 170383651.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3012/5000
    500/500 [==============================] - 0s 88us/sample - loss: 29287491.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3013/5000
    500/500 [==============================] - 0s 82us/sample - loss: -4639952.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3014/5000
    500/500 [==============================] - 0s 84us/sample - loss: 286169.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3015/5000
    500/500 [==============================] - 0s 78us/sample - loss: -7111783.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3016/5000
    500/500 [==============================] - 0s 90us/sample - loss: 77534.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3017/5000
    500/500 [==============================] - 0s 82us/sample - loss: -55440143.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3018/5000
    500/500 [==============================] - 0s 80us/sample - loss: -133542409.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3019/5000
    500/500 [==============================] - 0s 88us/sample - loss: -45131674.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3020/5000
    500/500 [==============================] - 0s 86us/sample - loss: 86570313.4680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3021/5000
    500/500 [==============================] - 0s 84us/sample - loss: 27234422.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3022/5000
    500/500 [==============================] - 0s 88us/sample - loss: 48004647.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3023/5000
    500/500 [==============================] - 0s 84us/sample - loss: 146910985.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3024/5000
    500/500 [==============================] - 0s 88us/sample - loss: 13421251.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3025/5000
    500/500 [==============================] - 0s 78us/sample - loss: 38850753.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3026/5000
    500/500 [==============================] - 0s 94us/sample - loss: 17095839.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3027/5000
    500/500 [==============================] - 0s 78us/sample - loss: 81811153.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3028/5000
    500/500 [==============================] - 0s 77us/sample - loss: 8457723.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3029/5000
    500/500 [==============================] - 0s 84us/sample - loss: 35033422.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3030/5000
    500/500 [==============================] - 0s 86us/sample - loss: -129272277.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3031/5000
    500/500 [==============================] - 0s 102us/sample - loss: -25358829.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3032/5000
    500/500 [==============================] - 0s 90us/sample - loss: -79451384.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3033/5000
    500/500 [==============================] - 0s 88us/sample - loss: 5028291.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3034/5000
    500/500 [==============================] - 0s 84us/sample - loss: 47192645.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3035/5000
    500/500 [==============================] - 0s 86us/sample - loss: -22214225.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3036/5000
    500/500 [==============================] - 0s 92us/sample - loss: 33962686.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3037/5000
    500/500 [==============================] - 0s 88us/sample - loss: -77702356.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3038/5000
    500/500 [==============================] - 0s 82us/sample - loss: -98412799.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3039/5000
    500/500 [==============================] - 0s 80us/sample - loss: -161867625.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3040/5000
    500/500 [==============================] - 0s 80us/sample - loss: -21040703.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3041/5000
    500/500 [==============================] - 0s 74us/sample - loss: -48548131.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3042/5000
    500/500 [==============================] - 0s 82us/sample - loss: 9439459.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3043/5000
    500/500 [==============================] - 0s 84us/sample - loss: -113964690.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3044/5000
    500/500 [==============================] - 0s 94us/sample - loss: -155422613.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3045/5000
    500/500 [==============================] - 0s 80us/sample - loss: 50611709.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3046/5000
    500/500 [==============================] - 0s 88us/sample - loss: -54666640.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3047/5000
    500/500 [==============================] - 0s 88us/sample - loss: 9183140.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3048/5000
    500/500 [==============================] - 0s 92us/sample - loss: -94635066.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3049/5000
    500/500 [==============================] - 0s 82us/sample - loss: 77916479.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3050/5000
    500/500 [==============================] - 0s 92us/sample - loss: -54037154.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3051/5000
    500/500 [==============================] - 0s 74us/sample - loss: -43887746.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3052/5000
    500/500 [==============================] - 0s 76us/sample - loss: -238288386.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3053/5000
    500/500 [==============================] - 0s 80us/sample - loss: 29281177.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3054/5000
    500/500 [==============================] - 0s 86us/sample - loss: 18744716.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3055/5000
    500/500 [==============================] - 0s 80us/sample - loss: -19583387.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3056/5000
    500/500 [==============================] - 0s 84us/sample - loss: 47782266.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3057/5000
    500/500 [==============================] - 0s 88us/sample - loss: 5017354.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3058/5000
    500/500 [==============================] - 0s 84us/sample - loss: -105390552.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3059/5000
    500/500 [==============================] - 0s 80us/sample - loss: 128784670.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3060/5000
    500/500 [==============================] - 0s 80us/sample - loss: -91212617.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3061/5000
    500/500 [==============================] - 0s 98us/sample - loss: -145903941.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3062/5000
    500/500 [==============================] - 0s 90us/sample - loss: 170703988.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3063/5000
    500/500 [==============================] - 0s 94us/sample - loss: 256971767.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3064/5000
    500/500 [==============================] - 0s 90us/sample - loss: 228798.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3065/5000
    500/500 [==============================] - 0s 78us/sample - loss: -60454369.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3066/5000
    500/500 [==============================] - 0s 86us/sample - loss: -90092684.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3067/5000
    500/500 [==============================] - 0s 82us/sample - loss: -18160140.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3068/5000
    500/500 [==============================] - 0s 82us/sample - loss: -17507640.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3069/5000
    500/500 [==============================] - 0s 84us/sample - loss: 48204092.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3070/5000
    500/500 [==============================] - 0s 80us/sample - loss: -43279228.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3071/5000
    500/500 [==============================] - 0s 84us/sample - loss: -65698672.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3072/5000
    500/500 [==============================] - 0s 80us/sample - loss: 59974806.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3073/5000
    500/500 [==============================] - 0s 80us/sample - loss: -78782997.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3074/5000
    500/500 [==============================] - 0s 82us/sample - loss: -8700956.2860 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3075/5000
    500/500 [==============================] - 0s 80us/sample - loss: 38508250.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3076/5000
    500/500 [==============================] - 0s 84us/sample - loss: 153944324.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3077/5000
    500/500 [==============================] - 0s 80us/sample - loss: -239144803.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3078/5000
    500/500 [==============================] - 0s 82us/sample - loss: -180878236.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3079/5000
    500/500 [==============================] - 0s 82us/sample - loss: 65910679.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3080/5000
    500/500 [==============================] - 0s 86us/sample - loss: 69170048.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3081/5000
    500/500 [==============================] - 0s 84us/sample - loss: 97937278.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3082/5000
    500/500 [==============================] - 0s 80us/sample - loss: -134287964.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3083/5000
    500/500 [==============================] - 0s 94us/sample - loss: -87855929.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3084/5000
    500/500 [==============================] - 0s 74us/sample - loss: 140620802.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3085/5000
    500/500 [==============================] - 0s 73us/sample - loss: 77112013.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3086/5000
    500/500 [==============================] - 0s 84us/sample - loss: -49527361.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3087/5000
    500/500 [==============================] - 0s 88us/sample - loss: -102051230.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3088/5000
    500/500 [==============================] - 0s 82us/sample - loss: 137794662.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3089/5000
    500/500 [==============================] - 0s 84us/sample - loss: 140476721.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3090/5000
    500/500 [==============================] - 0s 86us/sample - loss: -24476387.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3091/5000
    500/500 [==============================] - 0s 94us/sample - loss: 142712078.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3092/5000
    500/500 [==============================] - 0s 76us/sample - loss: -135005714.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3093/5000
    500/500 [==============================] - 0s 94us/sample - loss: -87057242.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3094/5000
    500/500 [==============================] - 0s 90us/sample - loss: -95041207.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3095/5000
    500/500 [==============================] - 0s 92us/sample - loss: 116196001.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3096/5000
    500/500 [==============================] - 0s 76us/sample - loss: -87961720.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3097/5000
    500/500 [==============================] - 0s 88us/sample - loss: 14507302.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3098/5000
    500/500 [==============================] - 0s 80us/sample - loss: -84991330.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3099/5000
    500/500 [==============================] - 0s 75us/sample - loss: 75493165.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3100/5000
    500/500 [==============================] - 0s 82us/sample - loss: -95895151.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3101/5000
    500/500 [==============================] - 0s 82us/sample - loss: -219962029.9820 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3102/5000
    500/500 [==============================] - 0s 80us/sample - loss: -122259542.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3103/5000
    500/500 [==============================] - 0s 74us/sample - loss: 128510242.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3104/5000
    500/500 [==============================] - 0s 80us/sample - loss: 90278259.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3105/5000
    500/500 [==============================] - 0s 82us/sample - loss: 134854195.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3106/5000
    500/500 [==============================] - 0s 82us/sample - loss: -172542866.3060 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3107/5000
    500/500 [==============================] - 0s 76us/sample - loss: -6178757.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3108/5000
    500/500 [==============================] - 0s 78us/sample - loss: -14896496.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3109/5000
    500/500 [==============================] - 0s 84us/sample - loss: 233217743.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3110/5000
    500/500 [==============================] - 0s 80us/sample - loss: 119191262.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3111/5000
    500/500 [==============================] - 0s 84us/sample - loss: 120672694.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3112/5000
    500/500 [==============================] - 0s 92us/sample - loss: 41741121.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3113/5000
    500/500 [==============================] - 0s 82us/sample - loss: 16055915.7125 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3114/5000
    500/500 [==============================] - 0s 84us/sample - loss: -175857353.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3115/5000
    500/500 [==============================] - 0s 80us/sample - loss: -5759989.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3116/5000
    500/500 [==============================] - 0s 84us/sample - loss: 83810342.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3117/5000
    500/500 [==============================] - 0s 82us/sample - loss: 22452056.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3118/5000
    500/500 [==============================] - 0s 80us/sample - loss: 44511796.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3119/5000
    500/500 [==============================] - 0s 82us/sample - loss: -40966633.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3120/5000
    500/500 [==============================] - 0s 84us/sample - loss: -10459439.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3121/5000
    500/500 [==============================] - 0s 94us/sample - loss: 6336100.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3122/5000
    500/500 [==============================] - 0s 82us/sample - loss: 58861590.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3123/5000
    500/500 [==============================] - 0s 82us/sample - loss: -154567988.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3124/5000
    500/500 [==============================] - 0s 86us/sample - loss: 218643009.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3125/5000
    500/500 [==============================] - 0s 96us/sample - loss: 40592403.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3126/5000
    500/500 [==============================] - 0s 78us/sample - loss: 156245044.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3127/5000
    500/500 [==============================] - 0s 77us/sample - loss: -101987278.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3128/5000
    500/500 [==============================] - 0s 82us/sample - loss: 21475467.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3129/5000
    500/500 [==============================] - 0s 84us/sample - loss: -13973133.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3130/5000
    500/500 [==============================] - 0s 82us/sample - loss: 150495327.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3131/5000
    500/500 [==============================] - 0s 94us/sample - loss: 192054747.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3132/5000
    500/500 [==============================] - 0s 88us/sample - loss: 85789118.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3133/5000
    500/500 [==============================] - 0s 76us/sample - loss: -158766551.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3134/5000
    500/500 [==============================] - 0s 80us/sample - loss: 44490558.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3135/5000
    500/500 [==============================] - 0s 88us/sample - loss: 152251399.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3136/5000
    500/500 [==============================] - 0s 90us/sample - loss: 88854151.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3137/5000
    500/500 [==============================] - 0s 84us/sample - loss: 72175979.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3138/5000
    500/500 [==============================] - 0s 79us/sample - loss: 52341586.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3139/5000
    500/500 [==============================] - 0s 86us/sample - loss: 45036833.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3140/5000
    500/500 [==============================] - 0s 80us/sample - loss: 49171674.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3141/5000
    500/500 [==============================] - 0s 84us/sample - loss: 14517300.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3142/5000
    500/500 [==============================] - 0s 90us/sample - loss: -69001553.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3143/5000
    500/500 [==============================] - 0s 78us/sample - loss: 40738846.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3144/5000
    500/500 [==============================] - 0s 84us/sample - loss: -15915249.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3145/5000
    500/500 [==============================] - 0s 84us/sample - loss: 76669682.6280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3146/5000
    500/500 [==============================] - 0s 92us/sample - loss: -23688722.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3147/5000
    500/500 [==============================] - 0s 84us/sample - loss: 95700230.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3148/5000
    500/500 [==============================] - 0s 82us/sample - loss: 28988536.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3149/5000
    500/500 [==============================] - 0s 92us/sample - loss: 57557324.9960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3150/5000
    500/500 [==============================] - 0s 84us/sample - loss: 81802929.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3151/5000
    500/500 [==============================] - 0s 82us/sample - loss: 742090.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3152/5000
    500/500 [==============================] - 0s 77us/sample - loss: -121463985.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3153/5000
    500/500 [==============================] - 0s 82us/sample - loss: -43441150.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3154/5000
    500/500 [==============================] - 0s 79us/sample - loss: 50028025.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3155/5000
    500/500 [==============================] - 0s 80us/sample - loss: -80672666.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3156/5000
    500/500 [==============================] - 0s 82us/sample - loss: 25857167.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3157/5000
    500/500 [==============================] - 0s 82us/sample - loss: 129628166.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3158/5000
    500/500 [==============================] - 0s 86us/sample - loss: 95939664.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3159/5000
    500/500 [==============================] - 0s 84us/sample - loss: -2647601.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3160/5000
    500/500 [==============================] - 0s 84us/sample - loss: -49349924.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3161/5000
    500/500 [==============================] - 0s 92us/sample - loss: 49539539.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3162/5000
    500/500 [==============================] - 0s 86us/sample - loss: -99573652.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3163/5000
    500/500 [==============================] - 0s 82us/sample - loss: 40593141.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3164/5000
    500/500 [==============================] - 0s 78us/sample - loss: -29416124.1320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3165/5000
    500/500 [==============================] - 0s 92us/sample - loss: -7956504.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3166/5000
    500/500 [==============================] - 0s 91us/sample - loss: -171059586.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3167/5000
    500/500 [==============================] - 0s 88us/sample - loss: 19307936.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3168/5000
    500/500 [==============================] - 0s 96us/sample - loss: 107218293.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3169/5000
    500/500 [==============================] - 0s 88us/sample - loss: 45412747.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3170/5000
    500/500 [==============================] - 0s 90us/sample - loss: 47398296.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3171/5000
    500/500 [==============================] - 0s 88us/sample - loss: 31834673.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3172/5000
    500/500 [==============================] - 0s 104us/sample - loss: 27569204.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3173/5000
    500/500 [==============================] - 0s 104us/sample - loss: 206367565.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3174/5000
    500/500 [==============================] - 0s 94us/sample - loss: -29099090.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3175/5000
    500/500 [==============================] - 0s 87us/sample - loss: -73694439.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3176/5000
    500/500 [==============================] - 0s 90us/sample - loss: -19610651.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3177/5000
    500/500 [==============================] - 0s 104us/sample - loss: -31589475.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3178/5000
    500/500 [==============================] - 0s 110us/sample - loss: -50175755.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3179/5000
    500/500 [==============================] - 0s 104us/sample - loss: 45359258.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3180/5000
    500/500 [==============================] - 0s 94us/sample - loss: 99712487.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3181/5000
    500/500 [==============================] - 0s 102us/sample - loss: 140978182.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3182/5000
    500/500 [==============================] - 0s 100us/sample - loss: 23703782.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3183/5000
    500/500 [==============================] - 0s 95us/sample - loss: -8731092.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3184/5000
    500/500 [==============================] - 0s 94us/sample - loss: -20689764.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3185/5000
    500/500 [==============================] - 0s 88us/sample - loss: -22144170.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3186/5000
    500/500 [==============================] - 0s 104us/sample - loss: 77815905.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3187/5000
    500/500 [==============================] - 0s 90us/sample - loss: 84309029.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3188/5000
    500/500 [==============================] - 0s 108us/sample - loss: 55963929.5199 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3189/5000
    500/500 [==============================] - 0s 102us/sample - loss: -55181249.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3190/5000
    500/500 [==============================] - 0s 96us/sample - loss: 50765071.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3191/5000
    500/500 [==============================] - 0s 100us/sample - loss: 68491556.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3192/5000
    500/500 [==============================] - 0s 100us/sample - loss: 25043176.3240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3193/5000
    500/500 [==============================] - 0s 100us/sample - loss: 54437902.4801 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3194/5000
    500/500 [==============================] - 0s 98us/sample - loss: -38853047.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3195/5000
    500/500 [==============================] - 0s 91us/sample - loss: -67507499.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3196/5000
    500/500 [==============================] - 0s 94us/sample - loss: 24059892.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3197/5000
    500/500 [==============================] - 0s 106us/sample - loss: -17676280.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3198/5000
    500/500 [==============================] - 0s 102us/sample - loss: 98457771.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3199/5000
    500/500 [==============================] - 0s 93us/sample - loss: 89891519.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3200/5000
    500/500 [==============================] - 0s 91us/sample - loss: 61570852.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3201/5000
    500/500 [==============================] - 0s 88us/sample - loss: 18367161.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3202/5000
    500/500 [==============================] - 0s 104us/sample - loss: 121224560.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3203/5000
    500/500 [==============================] - 0s 104us/sample - loss: -27715458.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3204/5000
    500/500 [==============================] - 0s 95us/sample - loss: 72689297.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3205/5000
    500/500 [==============================] - 0s 80us/sample - loss: -164316191.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3206/5000
    500/500 [==============================] - 0s 98us/sample - loss: -92909595.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3207/5000
    500/500 [==============================] - 0s 98us/sample - loss: -148194241.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3208/5000
    500/500 [==============================] - 0s 102us/sample - loss: 31150200.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3209/5000
    500/500 [==============================] - 0s 98us/sample - loss: 54622883.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3210/5000
    500/500 [==============================] - 0s 92us/sample - loss: 34876248.0799 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3211/5000
    500/500 [==============================] - 0s 102us/sample - loss: -43683817.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3212/5000
    500/500 [==============================] - 0s 88us/sample - loss: -1211370.7400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3213/5000
    500/500 [==============================] - 0s 94us/sample - loss: 48948997.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3214/5000
    500/500 [==============================] - 0s 106us/sample - loss: 46177957.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3215/5000
    500/500 [==============================] - 0s 84us/sample - loss: -75053541.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3216/5000
    500/500 [==============================] - 0s 92us/sample - loss: -6995206.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3217/5000
    500/500 [==============================] - 0s 86us/sample - loss: 140908674.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3218/5000
    500/500 [==============================] - 0s 80us/sample - loss: 53062358.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3219/5000
    500/500 [==============================] - 0s 90us/sample - loss: 93527285.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3220/5000
    500/500 [==============================] - 0s 92us/sample - loss: 68649238.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3221/5000
    500/500 [==============================] - 0s 78us/sample - loss: -43437807.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3222/5000
    500/500 [==============================] - 0s 94us/sample - loss: 190138613.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3223/5000
    500/500 [==============================] - 0s 86us/sample - loss: -46792361.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3224/5000
    500/500 [==============================] - 0s 94us/sample - loss: 83858489.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3225/5000
    500/500 [==============================] - 0s 78us/sample - loss: 4014469.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3226/5000
    500/500 [==============================] - 0s 90us/sample - loss: -83903704.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3227/5000
    500/500 [==============================] - 0s 78us/sample - loss: -56815560.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3228/5000
    500/500 [==============================] - 0s 74us/sample - loss: -92302402.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3229/5000
    500/500 [==============================] - 0s 82us/sample - loss: 108344364.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3230/5000
    500/500 [==============================] - 0s 84us/sample - loss: -20676102.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3231/5000
    500/500 [==============================] - 0s 82us/sample - loss: 95879750.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3232/5000
    500/500 [==============================] - 0s 84us/sample - loss: 56086474.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3233/5000
    500/500 [==============================] - 0s 86us/sample - loss: 168507618.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3234/5000
    500/500 [==============================] - 0s 84us/sample - loss: -21745153.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3235/5000
    500/500 [==============================] - 0s 82us/sample - loss: 49637481.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3236/5000
    500/500 [==============================] - 0s 84us/sample - loss: -22766443.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3237/5000
    500/500 [==============================] - 0s 84us/sample - loss: 46392860.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3238/5000
    500/500 [==============================] - 0s 84us/sample - loss: 23794856.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3239/5000
    500/500 [==============================] - 0s 88us/sample - loss: 68302444.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3240/5000
    500/500 [==============================] - 0s 86us/sample - loss: 37125237.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3241/5000
    500/500 [==============================] - 0s 88us/sample - loss: 89542828.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3242/5000
    500/500 [==============================] - 0s 94us/sample - loss: 14595116.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3243/5000
    500/500 [==============================] - 0s 76us/sample - loss: -29124864.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3244/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32229114.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3245/5000
    500/500 [==============================] - 0s 80us/sample - loss: 27995043.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3246/5000
    500/500 [==============================] - 0s 82us/sample - loss: -194491801.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3247/5000
    500/500 [==============================] - 0s 80us/sample - loss: 61331005.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3248/5000
    500/500 [==============================] - 0s 82us/sample - loss: -77009077.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3249/5000
    500/500 [==============================] - 0s 82us/sample - loss: 65354160.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3250/5000
    500/500 [==============================] - 0s 82us/sample - loss: -17882622.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3251/5000
    500/500 [==============================] - 0s 82us/sample - loss: 25009668.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3252/5000
    500/500 [==============================] - 0s 82us/sample - loss: 139250928.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3253/5000
    500/500 [==============================] - 0s 82us/sample - loss: -87418570.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3254/5000
    500/500 [==============================] - 0s 80us/sample - loss: -69439245.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3255/5000
    500/500 [==============================] - 0s 84us/sample - loss: 58080829.4740 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3256/5000
    500/500 [==============================] - 0s 84us/sample - loss: -118891058.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3257/5000
    500/500 [==============================] - 0s 94us/sample - loss: -90775378.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3258/5000
    500/500 [==============================] - 0s 84us/sample - loss: -76279054.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3259/5000
    500/500 [==============================] - 0s 94us/sample - loss: -65704099.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3260/5000
    500/500 [==============================] - 0s 80us/sample - loss: -123140976.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3261/5000
    500/500 [==============================] - 0s 84us/sample - loss: 113591951.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3262/5000
    500/500 [==============================] - 0s 76us/sample - loss: 25113343.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3263/5000
    500/500 [==============================] - 0s 88us/sample - loss: -57137003.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3264/5000
    500/500 [==============================] - 0s 96us/sample - loss: 69113082.4660 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3265/5000
    500/500 [==============================] - 0s 74us/sample - loss: -129350936.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3266/5000
    500/500 [==============================] - 0s 84us/sample - loss: -130903817.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3267/5000
    500/500 [==============================] - 0s 86us/sample - loss: 12671373.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3268/5000
    500/500 [==============================] - 0s 78us/sample - loss: 128448805.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3269/5000
    500/500 [==============================] - 0s 84us/sample - loss: -75812683.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3270/5000
    500/500 [==============================] - 0s 80us/sample - loss: 135819313.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3271/5000
    500/500 [==============================] - 0s 86us/sample - loss: -41286937.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3272/5000
    500/500 [==============================] - 0s 82us/sample - loss: 109604365.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3273/5000
    500/500 [==============================] - 0s 86us/sample - loss: -28883407.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3274/5000
    500/500 [==============================] - 0s 82us/sample - loss: 157269507.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3275/5000
    500/500 [==============================] - 0s 84us/sample - loss: 71463054.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3276/5000
    500/500 [==============================] - 0s 92us/sample - loss: -63367346.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3277/5000
    500/500 [==============================] - 0s 78us/sample - loss: 429571.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3278/5000
    500/500 [==============================] - 0s 73us/sample - loss: 29067841.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3279/5000
    500/500 [==============================] - 0s 84us/sample - loss: 236475180.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3280/5000
    500/500 [==============================] - 0s 88us/sample - loss: -20450422.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3281/5000
    500/500 [==============================] - 0s 84us/sample - loss: 149433046.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3282/5000
    500/500 [==============================] - 0s 94us/sample - loss: 121775070.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3283/5000
    500/500 [==============================] - 0s 76us/sample - loss: 42812480.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3284/5000
    500/500 [==============================] - 0s 82us/sample - loss: 35837835.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3285/5000
    500/500 [==============================] - 0s 88us/sample - loss: -107412234.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3286/5000
    500/500 [==============================] - 0s 82us/sample - loss: -76866220.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3287/5000
    500/500 [==============================] - 0s 92us/sample - loss: 32283615.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3288/5000
    500/500 [==============================] - 0s 84us/sample - loss: 136772539.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3289/5000
    500/500 [==============================] - 0s 73us/sample - loss: 116962927.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3290/5000
    500/500 [==============================] - 0s 80us/sample - loss: -79852064.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3291/5000
    500/500 [==============================] - 0s 88us/sample - loss: 127473227.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3292/5000
    500/500 [==============================] - 0s 90us/sample - loss: -60777988.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3293/5000
    500/500 [==============================] - 0s 80us/sample - loss: -25691500.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3294/5000
    500/500 [==============================] - 0s 84us/sample - loss: 76158457.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3295/5000
    500/500 [==============================] - 0s 84us/sample - loss: -43664742.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3296/5000
    500/500 [==============================] - 0s 80us/sample - loss: 104034341.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3297/5000
    500/500 [==============================] - 0s 82us/sample - loss: 96066242.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3298/5000
    500/500 [==============================] - 0s 80us/sample - loss: 13890195.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3299/5000
    500/500 [==============================] - 0s 84us/sample - loss: -115327150.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3300/5000
    500/500 [==============================] - 0s 92us/sample - loss: -61346047.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3301/5000
    500/500 [==============================] - 0s 84us/sample - loss: 45541187.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3302/5000
    500/500 [==============================] - 0s 96us/sample - loss: -8620420.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3303/5000
    500/500 [==============================] - 0s 84us/sample - loss: -89093416.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3304/5000
    500/500 [==============================] - 0s 86us/sample - loss: -155988860.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3305/5000
    500/500 [==============================] - 0s 74us/sample - loss: -44068327.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3306/5000
    500/500 [==============================] - 0s 80us/sample - loss: 27432893.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3307/5000
    500/500 [==============================] - 0s 88us/sample - loss: 20106825.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3308/5000
    500/500 [==============================] - 0s 92us/sample - loss: -30798352.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3309/5000
    500/500 [==============================] - 0s 80us/sample - loss: 8515527.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3310/5000
    500/500 [==============================] - 0s 75us/sample - loss: -156984939.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3311/5000
    500/500 [==============================] - 0s 96us/sample - loss: 20213429.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3312/5000
    500/500 [==============================] - 0s 78us/sample - loss: -46330811.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3313/5000
    500/500 [==============================] - 0s 84us/sample - loss: 1127581.1839 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3314/5000
    500/500 [==============================] - 0s 82us/sample - loss: 87450596.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3315/5000
    500/500 [==============================] - 0s 92us/sample - loss: 84692733.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3316/5000
    500/500 [==============================] - 0s 76us/sample - loss: 94548576.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3317/5000
    500/500 [==============================] - 0s 76us/sample - loss: 43618651.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3318/5000
    500/500 [==============================] - 0s 80us/sample - loss: 39805886.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3319/5000
    500/500 [==============================] - 0s 84us/sample - loss: 73903236.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3320/5000
    500/500 [==============================] - 0s 82us/sample - loss: 104699604.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3321/5000
    500/500 [==============================] - 0s 82us/sample - loss: -146862339.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3322/5000
    500/500 [==============================] - 0s 86us/sample - loss: 3841096.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3323/5000
    500/500 [==============================] - 0s 84us/sample - loss: 49006588.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3324/5000
    500/500 [==============================] - 0s 82us/sample - loss: 108544913.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3325/5000
    500/500 [==============================] - 0s 84us/sample - loss: -65496124.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3326/5000
    500/500 [==============================] - 0s 82us/sample - loss: -18561531.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3327/5000
    500/500 [==============================] - 0s 96us/sample - loss: 89691244.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3328/5000
    500/500 [==============================] - 0s 86us/sample - loss: -39842313.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3329/5000
    500/500 [==============================] - 0s 76us/sample - loss: -60797769.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3330/5000
    500/500 [==============================] - 0s 86us/sample - loss: 39973547.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3331/5000
    500/500 [==============================] - 0s 84us/sample - loss: -22710235.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3332/5000
    500/500 [==============================] - 0s 82us/sample - loss: -2772582.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3333/5000
    500/500 [==============================] - 0s 88us/sample - loss: 27436975.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3334/5000
    500/500 [==============================] - 0s 82us/sample - loss: -41199988.8300 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3335/5000
    500/500 [==============================] - 0s 82us/sample - loss: -26346990.1460 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3336/5000
    500/500 [==============================] - 0s 82us/sample - loss: -127349704.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3337/5000
    500/500 [==============================] - 0s 82us/sample - loss: -46064605.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3338/5000
    500/500 [==============================] - 0s 84us/sample - loss: 59102253.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3339/5000
    500/500 [==============================] - 0s 82us/sample - loss: 14860579.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3340/5000
    500/500 [==============================] - 0s 84us/sample - loss: 45959102.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3341/5000
    500/500 [==============================] - 0s 80us/sample - loss: -70775600.2500 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3342/5000
    500/500 [==============================] - 0s 82us/sample - loss: -36282044.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3343/5000
    500/500 [==============================] - 0s 86us/sample - loss: -11816644.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3344/5000
    500/500 [==============================] - 0s 82us/sample - loss: -8174166.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3345/5000
    500/500 [==============================] - 0s 84us/sample - loss: -78854233.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3346/5000
    500/500 [==============================] - 0s 86us/sample - loss: 116652147.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3347/5000
    500/500 [==============================] - 0s 84us/sample - loss: 52399392.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3348/5000
    500/500 [==============================] - 0s 86us/sample - loss: 140701523.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3349/5000
    500/500 [==============================] - 0s 78us/sample - loss: 86817467.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3350/5000
    500/500 [==============================] - 0s 82us/sample - loss: 104964451.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3351/5000
    500/500 [==============================] - 0s 86us/sample - loss: 17434859.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3352/5000
    500/500 [==============================] - 0s 76us/sample - loss: -8792047.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3353/5000
    500/500 [==============================] - 0s 82us/sample - loss: -38218594.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3354/5000
    500/500 [==============================] - 0s 82us/sample - loss: -95176500.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3355/5000
    500/500 [==============================] - 0s 92us/sample - loss: -128793276.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3356/5000
    500/500 [==============================] - 0s 76us/sample - loss: -190697458.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3357/5000
    500/500 [==============================] - 0s 84us/sample - loss: 62764702.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3358/5000
    500/500 [==============================] - 0s 82us/sample - loss: -46629102.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3359/5000
    500/500 [==============================] - 0s 82us/sample - loss: -26996494.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3360/5000
    500/500 [==============================] - 0s 88us/sample - loss: 90949650.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3361/5000
    500/500 [==============================] - 0s 84us/sample - loss: -70183013.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3362/5000
    500/500 [==============================] - 0s 76us/sample - loss: 12441340.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3363/5000
    500/500 [==============================] - 0s 82us/sample - loss: -76034004.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3364/5000
    500/500 [==============================] - 0s 86us/sample - loss: 105675292.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3365/5000
    500/500 [==============================] - 0s 84us/sample - loss: 31121823.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3366/5000
    500/500 [==============================] - 0s 84us/sample - loss: -154513836.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3367/5000
    500/500 [==============================] - 0s 84us/sample - loss: 47106127.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3368/5000
    500/500 [==============================] - 0s 92us/sample - loss: -140868648.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3369/5000
    500/500 [==============================] - 0s 76us/sample - loss: -7511578.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3370/5000
    500/500 [==============================] - 0s 84us/sample - loss: 45709803.0700 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3371/5000
    500/500 [==============================] - 0s 84us/sample - loss: 44441607.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3372/5000
    500/500 [==============================] - 0s 84us/sample - loss: 29948057.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3373/5000
    500/500 [==============================] - 0s 86us/sample - loss: 24777065.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3374/5000
    500/500 [==============================] - 0s 88us/sample - loss: 63714015.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3375/5000
    500/500 [==============================] - 0s 82us/sample - loss: 17429808.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3376/5000
    500/500 [==============================] - 0s 84us/sample - loss: 29974802.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3377/5000
    500/500 [==============================] - 0s 88us/sample - loss: 2021174.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3378/5000
    500/500 [==============================] - 0s 92us/sample - loss: 113099398.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3379/5000
    500/500 [==============================] - 0s 86us/sample - loss: 88296886.7240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3380/5000
    500/500 [==============================] - 0s 94us/sample - loss: -16055734.6600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3381/5000
    500/500 [==============================] - 0s 82us/sample - loss: -188440025.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3382/5000
    500/500 [==============================] - 0s 92us/sample - loss: -41417977.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3383/5000
    500/500 [==============================] - 0s 86us/sample - loss: 56137093.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3384/5000
    500/500 [==============================] - 0s 80us/sample - loss: -57597246.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3385/5000
    500/500 [==============================] - 0s 82us/sample - loss: -13612256.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3386/5000
    500/500 [==============================] - 0s 90us/sample - loss: -114741024.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3387/5000
    500/500 [==============================] - 0s 80us/sample - loss: 69134923.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3388/5000
    500/500 [==============================] - 0s 90us/sample - loss: -31439074.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3389/5000
    500/500 [==============================] - 0s 78us/sample - loss: 11611955.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3390/5000
    500/500 [==============================] - 0s 72us/sample - loss: 20258538.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3391/5000
    500/500 [==============================] - 0s 78us/sample - loss: -73183042.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3392/5000
    500/500 [==============================] - 0s 82us/sample - loss: 6951704.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3393/5000
    500/500 [==============================] - 0s 80us/sample - loss: 13920423.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3394/5000
    500/500 [==============================] - 0s 86us/sample - loss: 55597636.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3395/5000
    500/500 [==============================] - 0s 98us/sample - loss: 45121172.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3396/5000
    500/500 [==============================] - 0s 76us/sample - loss: -150634501.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3397/5000
    500/500 [==============================] - 0s 74us/sample - loss: -16716817.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3398/5000
    500/500 [==============================] - 0s 80us/sample - loss: -28665278.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3399/5000
    500/500 [==============================] - 0s 82us/sample - loss: -62296043.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3400/5000
    500/500 [==============================] - 0s 78us/sample - loss: -62647258.6230 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3401/5000
    500/500 [==============================] - 0s 84us/sample - loss: 12234738.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3402/5000
    500/500 [==============================] - 0s 80us/sample - loss: -98220213.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3403/5000
    500/500 [==============================] - 0s 84us/sample - loss: 79624819.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3404/5000
    500/500 [==============================] - 0s 82us/sample - loss: -10335804.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3405/5000
    500/500 [==============================] - 0s 80us/sample - loss: 78105809.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3406/5000
    500/500 [==============================] - 0s 86us/sample - loss: -67927183.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3407/5000
    500/500 [==============================] - 0s 74us/sample - loss: -1402331.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3408/5000
    500/500 [==============================] - 0s 86us/sample - loss: 178909863.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3409/5000
    500/500 [==============================] - 0s 82us/sample - loss: -58797174.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3410/5000
    500/500 [==============================] - 0s 86us/sample - loss: -29778736.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3411/5000
    500/500 [==============================] - 0s 86us/sample - loss: 4013225.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3412/5000
    500/500 [==============================] - 0s 82us/sample - loss: -94634118.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3413/5000
    500/500 [==============================] - 0s 86us/sample - loss: -173132458.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3414/5000
    500/500 [==============================] - 0s 86us/sample - loss: -75772563.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3415/5000
    500/500 [==============================] - 0s 84us/sample - loss: -42317075.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3416/5000
    500/500 [==============================] - 0s 94us/sample - loss: 19919045.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3417/5000
    500/500 [==============================] - 0s 86us/sample - loss: -59209229.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3418/5000
    500/500 [==============================] - 0s 94us/sample - loss: 21041607.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3419/5000
    500/500 [==============================] - 0s 76us/sample - loss: 24869343.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3420/5000
    500/500 [==============================] - 0s 84us/sample - loss: -39236827.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3421/5000
    500/500 [==============================] - 0s 82us/sample - loss: -47952116.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3422/5000
    500/500 [==============================] - 0s 77us/sample - loss: -84364482.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3423/5000
    500/500 [==============================] - 0s 82us/sample - loss: 113545069.0580 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3424/5000
    500/500 [==============================] - 0s 82us/sample - loss: -95612145.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3425/5000
    500/500 [==============================] - 0s 88us/sample - loss: -122635055.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3426/5000
    500/500 [==============================] - 0s 78us/sample - loss: -207905469.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3427/5000
    500/500 [==============================] - 0s 92us/sample - loss: 12688350.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3428/5000
    500/500 [==============================] - 0s 88us/sample - loss: -31295530.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3429/5000
    500/500 [==============================] - 0s 82us/sample - loss: 117452835.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3430/5000
    500/500 [==============================] - 0s 84us/sample - loss: 41502220.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3431/5000
    500/500 [==============================] - 0s 80us/sample - loss: 74268463.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3432/5000
    500/500 [==============================] - 0s 86us/sample - loss: -28784050.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3433/5000
    500/500 [==============================] - 0s 86us/sample - loss: 150979394.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3434/5000
    500/500 [==============================] - 0s 80us/sample - loss: 91375512.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3435/5000
    500/500 [==============================] - 0s 84us/sample - loss: 3314571.9000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3436/5000
    500/500 [==============================] - 0s 82us/sample - loss: 111384308.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3437/5000
    500/500 [==============================] - 0s 84us/sample - loss: 8988435.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3438/5000
    500/500 [==============================] - 0s 82us/sample - loss: 80963269.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3439/5000
    500/500 [==============================] - 0s 84us/sample - loss: -38683111.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3440/5000
    500/500 [==============================] - 0s 94us/sample - loss: -39907140.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3441/5000
    500/500 [==============================] - 0s 76us/sample - loss: 108498283.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3442/5000
    500/500 [==============================] - 0s 72us/sample - loss: -117215321.4440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3443/5000
    500/500 [==============================] - 0s 82us/sample - loss: 105274021.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3444/5000
    500/500 [==============================] - 0s 82us/sample - loss: -46948861.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3445/5000
    500/500 [==============================] - 0s 82us/sample - loss: 1582165.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3446/5000
    500/500 [==============================] - 0s 84us/sample - loss: -120667761.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3447/5000
    500/500 [==============================] - 0s 90us/sample - loss: -28997174.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3448/5000
    500/500 [==============================] - 0s 76us/sample - loss: 43999370.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3449/5000
    500/500 [==============================] - 0s 84us/sample - loss: 205461627.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3450/5000
    500/500 [==============================] - 0s 82us/sample - loss: 80956958.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3451/5000
    500/500 [==============================] - 0s 84us/sample - loss: 90347076.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3452/5000
    500/500 [==============================] - 0s 86us/sample - loss: -77922294.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3453/5000
    500/500 [==============================] - 0s 78us/sample - loss: -15364209.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3454/5000
    500/500 [==============================] - 0s 82us/sample - loss: -82993917.1800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3455/5000
    500/500 [==============================] - 0s 82us/sample - loss: 21268136.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3456/5000
    500/500 [==============================] - 0s 86us/sample - loss: -3689089.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3457/5000
    500/500 [==============================] - 0s 84us/sample - loss: 65699237.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3458/5000
    500/500 [==============================] - ETA: 0s - loss: 400903008.0000 - accuracy: 0.0000e+ - 0s 82us/sample - loss: 9976889.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3459/5000
    500/500 [==============================] - 0s 86us/sample - loss: 170035414.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3460/5000
    500/500 [==============================] - 0s 84us/sample - loss: -39807570.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3461/5000
    500/500 [==============================] - 0s 70us/sample - loss: -51524918.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3462/5000
    500/500 [==============================] - 0s 80us/sample - loss: -32022352.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3463/5000
    500/500 [==============================] - 0s 94us/sample - loss: -57921638.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3464/5000
    500/500 [==============================] - 0s 80us/sample - loss: 127814841.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3465/5000
    500/500 [==============================] - 0s 76us/sample - loss: 145703553.8600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3466/5000
    500/500 [==============================] - 0s 84us/sample - loss: -65630012.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3467/5000
    500/500 [==============================] - 0s 86us/sample - loss: -16427702.9080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3468/5000
    500/500 [==============================] - 0s 84us/sample - loss: -4755189.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3469/5000
    500/500 [==============================] - 0s 84us/sample - loss: 83411553.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3470/5000
    500/500 [==============================] - 0s 82us/sample - loss: 110468461.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3471/5000
    500/500 [==============================] - 0s 82us/sample - loss: 89478808.1560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3472/5000
    500/500 [==============================] - 0s 80us/sample - loss: 8672801.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3473/5000
    500/500 [==============================] - 0s 84us/sample - loss: 55248348.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3474/5000
    500/500 [==============================] - 0s 84us/sample - loss: 137766102.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3475/5000
    500/500 [==============================] - 0s 84us/sample - loss: -103260494.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3476/5000
    500/500 [==============================] - 0s 94us/sample - loss: 103198024.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3477/5000
    500/500 [==============================] - 0s 74us/sample - loss: 48622315.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3478/5000
    500/500 [==============================] - 0s 82us/sample - loss: -35440433.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3479/5000
    500/500 [==============================] - 0s 80us/sample - loss: -2815575.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3480/5000
    500/500 [==============================] - 0s 94us/sample - loss: 10069618.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3481/5000
    500/500 [==============================] - 0s 80us/sample - loss: -36599393.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3482/5000
    500/500 [==============================] - 0s 84us/sample - loss: 38043873.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3483/5000
    500/500 [==============================] - 0s 82us/sample - loss: -25704533.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3484/5000
    500/500 [==============================] - 0s 86us/sample - loss: 213203209.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3485/5000
    500/500 [==============================] - 0s 76us/sample - loss: 32701157.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3486/5000
    500/500 [==============================] - 0s 92us/sample - loss: -78446688.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3487/5000
    500/500 [==============================] - 0s 76us/sample - loss: 41348249.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3488/5000
    500/500 [==============================] - 0s 82us/sample - loss: 97506741.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3489/5000
    500/500 [==============================] - 0s 78us/sample - loss: -37134999.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3490/5000
    500/500 [==============================] - 0s 82us/sample - loss: 105813703.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3491/5000
    500/500 [==============================] - 0s 80us/sample - loss: -14377165.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3492/5000
    500/500 [==============================] - 0s 82us/sample - loss: 41821695.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3493/5000
    500/500 [==============================] - 0s 82us/sample - loss: 20602067.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3494/5000
    500/500 [==============================] - 0s 84us/sample - loss: -35006948.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3495/5000
    500/500 [==============================] - 0s 80us/sample - loss: -252511770.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3496/5000
    500/500 [==============================] - 0s 82us/sample - loss: 129688434.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3497/5000
    500/500 [==============================] - 0s 96us/sample - loss: -14446347.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3498/5000
    500/500 [==============================] - 0s 88us/sample - loss: 112136701.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3499/5000
    500/500 [==============================] - 0s 86us/sample - loss: -74807126.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3500/5000
    500/500 [==============================] - 0s 78us/sample - loss: -70324329.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3501/5000
    500/500 [==============================] - 0s 118us/sample - loss: -19170440.6600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3502/5000
    500/500 [==============================] - 0s 116us/sample - loss: 34435250.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3503/5000
    500/500 [==============================] - 0s 76us/sample - loss: 62272014.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3504/5000
    500/500 [==============================] - 0s 82us/sample - loss: 45464571.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3505/5000
    500/500 [==============================] - 0s 86us/sample - loss: 66453109.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3506/5000
    500/500 [==============================] - 0s 94us/sample - loss: 262031168.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3507/5000
    500/500 [==============================] - 0s 88us/sample - loss: 23921908.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3508/5000
    500/500 [==============================] - 0s 94us/sample - loss: 8692951.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3509/5000
    500/500 [==============================] - 0s 78us/sample - loss: -102302059.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3510/5000
    500/500 [==============================] - 0s 96us/sample - loss: 59591765.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3511/5000
    500/500 [==============================] - 0s 88us/sample - loss: 17556913.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3512/5000
    500/500 [==============================] - 0s 88us/sample - loss: -126164109.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3513/5000
    500/500 [==============================] - 0s 76us/sample - loss: 30603733.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3514/5000
    500/500 [==============================] - 0s 90us/sample - loss: 151090112.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3515/5000
    500/500 [==============================] - 0s 78us/sample - loss: -18785076.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3516/5000
    500/500 [==============================] - 0s 82us/sample - loss: -130890.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3517/5000
    500/500 [==============================] - 0s 80us/sample - loss: -69847160.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3518/5000
    500/500 [==============================] - 0s 88us/sample - loss: -81452225.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3519/5000
    500/500 [==============================] - 0s 94us/sample - loss: -120376013.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3520/5000
    500/500 [==============================] - 0s 74us/sample - loss: 116679023.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3521/5000
    500/500 [==============================] - 0s 74us/sample - loss: -98071564.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3522/5000
    500/500 [==============================] - 0s 88us/sample - loss: 44124612.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3523/5000
    500/500 [==============================] - 0s 76us/sample - loss: 8107630.1290 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3524/5000
    500/500 [==============================] - 0s 80us/sample - loss: 114189960.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3525/5000
    500/500 [==============================] - 0s 84us/sample - loss: 185076753.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3526/5000
    500/500 [==============================] - 0s 82us/sample - loss: -114818200.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3527/5000
    500/500 [==============================] - 0s 82us/sample - loss: -45830893.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3528/5000
    500/500 [==============================] - 0s 86us/sample - loss: 46275649.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3529/5000
    500/500 [==============================] - 0s 82us/sample - loss: 128702365.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3530/5000
    500/500 [==============================] - 0s 82us/sample - loss: 5622112.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3531/5000
    500/500 [==============================] - 0s 120us/sample - loss: 44724781.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3532/5000
    500/500 [==============================] - 0s 104us/sample - loss: 46518256.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3533/5000
    500/500 [==============================] - 0s 102us/sample - loss: -29494603.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3534/5000
    500/500 [==============================] - 0s 91us/sample - loss: 200567413.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3535/5000
    500/500 [==============================] - 0s 96us/sample - loss: 144797796.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3536/5000
    500/500 [==============================] - 0s 102us/sample - loss: -219277234.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3537/5000
    500/500 [==============================] - 0s 97us/sample - loss: -17907167.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3538/5000
    500/500 [==============================] - 0s 98us/sample - loss: 56054880.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3539/5000
    500/500 [==============================] - 0s 102us/sample - loss: -105943452.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3540/5000
    500/500 [==============================] - 0s 81us/sample - loss: -153054747.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3541/5000
    500/500 [==============================] - 0s 110us/sample - loss: 63767710.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3542/5000
    500/500 [==============================] - 0s 102us/sample - loss: -85241080.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3543/5000
    500/500 [==============================] - 0s 98us/sample - loss: -29272733.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3544/5000
    500/500 [==============================] - 0s 118us/sample - loss: 152455074.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3545/5000
    500/500 [==============================] - 0s 106us/sample - loss: -113419634.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3546/5000
    500/500 [==============================] - 0s 100us/sample - loss: -30901284.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3547/5000
    500/500 [==============================] - 0s 100us/sample - loss: 11093976.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3548/5000
    500/500 [==============================] - 0s 96us/sample - loss: 134043547.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3549/5000
    500/500 [==============================] - 0s 114us/sample - loss: 50447346.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3550/5000
    500/500 [==============================] - 0s 102us/sample - loss: 102408179.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3551/5000
    500/500 [==============================] - 0s 102us/sample - loss: -141357147.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3552/5000
    500/500 [==============================] - 0s 102us/sample - loss: -22115112.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3553/5000
    500/500 [==============================] - 0s 100us/sample - loss: 1344699.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3554/5000
    500/500 [==============================] - 0s 94us/sample - loss: -61421364.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3555/5000
    500/500 [==============================] - 0s 102us/sample - loss: 5194337.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3556/5000
    500/500 [==============================] - 0s 89us/sample - loss: 41839783.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3557/5000
    500/500 [==============================] - 0s 100us/sample - loss: -89811666.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3558/5000
    500/500 [==============================] - 0s 93us/sample - loss: -92433299.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3559/5000
    500/500 [==============================] - 0s 110us/sample - loss: -80426961.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3560/5000
    500/500 [==============================] - 0s 98us/sample - loss: -71077507.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3561/5000
    500/500 [==============================] - 0s 100us/sample - loss: -94328240.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3562/5000
    500/500 [==============================] - 0s 100us/sample - loss: 22150748.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3563/5000
    500/500 [==============================] - 0s 98us/sample - loss: 91650288.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3564/5000
    500/500 [==============================] - 0s 96us/sample - loss: 16569756.6200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3565/5000
    500/500 [==============================] - 0s 104us/sample - loss: -71371544.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3566/5000
    500/500 [==============================] - 0s 100us/sample - loss: 69552684.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3567/5000
    500/500 [==============================] - 0s 94us/sample - loss: -45215403.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3568/5000
    500/500 [==============================] - 0s 96us/sample - loss: 142611117.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3569/5000
    500/500 [==============================] - 0s 108us/sample - loss: 125195602.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3570/5000
    500/500 [==============================] - 0s 108us/sample - loss: -12638303.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3571/5000
    500/500 [==============================] - 0s 98us/sample - loss: 268252923.0120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3572/5000
    500/500 [==============================] - 0s 95us/sample - loss: -97319218.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3573/5000
    500/500 [==============================] - 0s 92us/sample - loss: -145940056.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3574/5000
    500/500 [==============================] - 0s 82us/sample - loss: 20621155.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3575/5000
    500/500 [==============================] - 0s 78us/sample - loss: 177618810.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3576/5000
    500/500 [==============================] - 0s 92us/sample - loss: 34239147.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3577/5000
    500/500 [==============================] - 0s 76us/sample - loss: 5079156.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3578/5000
    500/500 [==============================] - 0s 79us/sample - loss: -38947421.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3579/5000
    500/500 [==============================] - 0s 82us/sample - loss: -21411560.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3580/5000
    500/500 [==============================] - 0s 76us/sample - loss: 28938343.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3581/5000
    500/500 [==============================] - 0s 82us/sample - loss: -58390180.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3582/5000
    500/500 [==============================] - 0s 84us/sample - loss: -31891300.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3583/5000
    500/500 [==============================] - 0s 84us/sample - loss: -1658353.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3584/5000
    500/500 [==============================] - 0s 78us/sample - loss: 55545479.4561 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3585/5000
    500/500 [==============================] - 0s 86us/sample - loss: 21755403.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3586/5000
    500/500 [==============================] - 0s 80us/sample - loss: 89626921.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3587/5000
    500/500 [==============================] - 0s 86us/sample - loss: 62747249.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3588/5000
    500/500 [==============================] - 0s 82us/sample - loss: 24357731.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3589/5000
    500/500 [==============================] - 0s 86us/sample - loss: 65874339.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3590/5000
    500/500 [==============================] - 0s 86us/sample - loss: -57449978.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3591/5000
    500/500 [==============================] - 0s 82us/sample - loss: 12738829.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3592/5000
    500/500 [==============================] - 0s 84us/sample - loss: 41743523.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3593/5000
    500/500 [==============================] - 0s 94us/sample - loss: -2264962.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3594/5000
    500/500 [==============================] - 0s 76us/sample - loss: -2421822.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3595/5000
    500/500 [==============================] - 0s 81us/sample - loss: -227118830.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3596/5000
    500/500 [==============================] - 0s 84us/sample - loss: -121131115.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3597/5000
    500/500 [==============================] - 0s 80us/sample - loss: -76539698.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3598/5000
    500/500 [==============================] - 0s 86us/sample - loss: 99746414.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3599/5000
    500/500 [==============================] - 0s 82us/sample - loss: -196172872.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3600/5000
    500/500 [==============================] - 0s 84us/sample - loss: -14980424.5800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3601/5000
    500/500 [==============================] - 0s 84us/sample - loss: 60051355.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3602/5000
    500/500 [==============================] - 0s 80us/sample - loss: 221503326.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3603/5000
    500/500 [==============================] - 0s 86us/sample - loss: -80670667.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3604/5000
    500/500 [==============================] - 0s 92us/sample - loss: 23764395.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3605/5000
    500/500 [==============================] - 0s 78us/sample - loss: 85781401.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3606/5000
    500/500 [==============================] - 0s 82us/sample - loss: 74258715.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3607/5000
    500/500 [==============================] - 0s 88us/sample - loss: -69157510.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3608/5000
    500/500 [==============================] - 0s 80us/sample - loss: -160341868.9910 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3609/5000
    500/500 [==============================] - 0s 92us/sample - loss: -4353871.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3610/5000
    500/500 [==============================] - 0s 76us/sample - loss: -19514700.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3611/5000
    500/500 [==============================] - 0s 84us/sample - loss: 65458663.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3612/5000
    500/500 [==============================] - 0s 86us/sample - loss: 86888306.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3613/5000
    500/500 [==============================] - 0s 76us/sample - loss: -14400684.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3614/5000
    500/500 [==============================] - 0s 82us/sample - loss: -185842616.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3615/5000
    500/500 [==============================] - 0s 75us/sample - loss: 35740351.4870 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3616/5000
    500/500 [==============================] - 0s 90us/sample - loss: -7278347.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3617/5000
    500/500 [==============================] - 0s 84us/sample - loss: 89437427.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3618/5000
    500/500 [==============================] - 0s 76us/sample - loss: 174891734.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3619/5000
    500/500 [==============================] - 0s 80us/sample - loss: -138333364.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3620/5000
    500/500 [==============================] - 0s 86us/sample - loss: 149563971.8440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3621/5000
    500/500 [==============================] - 0s 82us/sample - loss: 29323353.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3622/5000
    500/500 [==============================] - 0s 84us/sample - loss: 147154861.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3623/5000
    500/500 [==============================] - 0s 82us/sample - loss: 177580731.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3624/5000
    500/500 [==============================] - ETA: 0s - loss: -351744320.0000 - accuracy: 0.0000e+0 - 0s 84us/sample - loss: 85717538.7500 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3625/5000
    500/500 [==============================] - 0s 90us/sample - loss: 19373933.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3626/5000
    500/500 [==============================] - 0s 86us/sample - loss: 66268808.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3627/5000
    500/500 [==============================] - 0s 88us/sample - loss: -90918737.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3628/5000
    500/500 [==============================] - 0s 80us/sample - loss: 130580005.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3629/5000
    500/500 [==============================] - 0s 86us/sample - loss: -173898398.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3630/5000
    500/500 [==============================] - 0s 82us/sample - loss: -118672698.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3631/5000
    500/500 [==============================] - 0s 86us/sample - loss: -25138592.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3632/5000
    500/500 [==============================] - 0s 90us/sample - loss: -69958454.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3633/5000
    500/500 [==============================] - 0s 78us/sample - loss: -6870773.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3634/5000
    500/500 [==============================] - 0s 84us/sample - loss: 66670300.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3635/5000
    500/500 [==============================] - 0s 82us/sample - loss: 110631550.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3636/5000
    500/500 [==============================] - 0s 88us/sample - loss: 48829039.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3637/5000
    500/500 [==============================] - 0s 92us/sample - loss: -22659670.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3638/5000
    500/500 [==============================] - 0s 76us/sample - loss: -51960260.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3639/5000
    500/500 [==============================] - 0s 82us/sample - loss: 91951094.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3640/5000
    500/500 [==============================] - 0s 82us/sample - loss: 7713732.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3641/5000
    500/500 [==============================] - 0s 84us/sample - loss: 22705650.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3642/5000
    500/500 [==============================] - 0s 94us/sample - loss: -28648413.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3643/5000
    500/500 [==============================] - 0s 80us/sample - loss: -166768761.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3644/5000
    500/500 [==============================] - 0s 74us/sample - loss: -121418909.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3645/5000
    500/500 [==============================] - 0s 86us/sample - loss: -40481464.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3646/5000
    500/500 [==============================] - 0s 86us/sample - loss: -51734283.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3647/5000
    500/500 [==============================] - 0s 96us/sample - loss: -90424656.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3648/5000
    500/500 [==============================] - 0s 88us/sample - loss: 166534882.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3649/5000
    500/500 [==============================] - 0s 96us/sample - loss: 54521936.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3650/5000
    500/500 [==============================] - 0s 76us/sample - loss: -41830867.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3651/5000
    500/500 [==============================] - 0s 88us/sample - loss: 59228408.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3652/5000
    500/500 [==============================] - 0s 76us/sample - loss: 39551382.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3653/5000
    500/500 [==============================] - 0s 75us/sample - loss: -94071489.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3654/5000
    500/500 [==============================] - 0s 84us/sample - loss: 45171004.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3655/5000
    500/500 [==============================] - 0s 90us/sample - loss: -136960716.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3656/5000
    500/500 [==============================] - 0s 86us/sample - loss: -18140244.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3657/5000
    500/500 [==============================] - 0s 88us/sample - loss: 38623582.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3658/5000
    500/500 [==============================] - 0s 88us/sample - loss: 40601873.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3659/5000
    500/500 [==============================] - 0s 80us/sample - loss: 11381999.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3660/5000
    500/500 [==============================] - 0s 94us/sample - loss: 213199962.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3661/5000
    500/500 [==============================] - 0s 76us/sample - loss: 59911913.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3662/5000
    500/500 [==============================] - 0s 86us/sample - loss: 62242940.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3663/5000
    500/500 [==============================] - 0s 82us/sample - loss: -55040712.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3664/5000
    500/500 [==============================] - 0s 82us/sample - loss: 40723448.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3665/5000
    500/500 [==============================] - 0s 92us/sample - loss: 21248330.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3666/5000
    500/500 [==============================] - 0s 78us/sample - loss: -263391348.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3667/5000
    500/500 [==============================] - 0s 75us/sample - loss: 190356380.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3668/5000
    500/500 [==============================] - 0s 82us/sample - loss: 93336216.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3669/5000
    500/500 [==============================] - 0s 96us/sample - loss: 53543101.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3670/5000
    500/500 [==============================] - 0s 92us/sample - loss: 121902342.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3671/5000
    500/500 [==============================] - 0s 96us/sample - loss: 91253122.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3672/5000
    500/500 [==============================] - 0s 76us/sample - loss: 98519409.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3673/5000
    500/500 [==============================] - 0s 80us/sample - loss: 16705789.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3674/5000
    500/500 [==============================] - 0s 88us/sample - loss: -74424589.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3675/5000
    500/500 [==============================] - 0s 78us/sample - loss: 111413775.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3676/5000
    500/500 [==============================] - 0s 86us/sample - loss: -107289190.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3677/5000
    500/500 [==============================] - 0s 88us/sample - loss: 68884443.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3678/5000
    500/500 [==============================] - 0s 76us/sample - loss: -62405905.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3679/5000
    500/500 [==============================] - 0s 84us/sample - loss: 89127246.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3680/5000
    500/500 [==============================] - 0s 82us/sample - loss: -43190377.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3681/5000
    500/500 [==============================] - 0s 92us/sample - loss: 141773954.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3682/5000
    500/500 [==============================] - 0s 86us/sample - loss: -161206522.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3683/5000
    500/500 [==============================] - 0s 80us/sample - loss: 18688743.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3684/5000
    500/500 [==============================] - 0s 86us/sample - loss: 100177838.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3685/5000
    500/500 [==============================] - 0s 92us/sample - loss: 69379854.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3686/5000
    500/500 [==============================] - 0s 78us/sample - loss: -39516783.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3687/5000
    500/500 [==============================] - 0s 90us/sample - loss: 55952202.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3688/5000
    500/500 [==============================] - 0s 76us/sample - loss: 153059261.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3689/5000
    500/500 [==============================] - 0s 84us/sample - loss: 8520393.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3690/5000
    500/500 [==============================] - 0s 80us/sample - loss: -65651276.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3691/5000
    500/500 [==============================] - 0s 86us/sample - loss: -17760750.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3692/5000
    500/500 [==============================] - 0s 100us/sample - loss: 113261627.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3693/5000
    500/500 [==============================] - 0s 84us/sample - loss: -145630941.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3694/5000
    500/500 [==============================] - 0s 98us/sample - loss: -20148022.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3695/5000
    500/500 [==============================] - 0s 78us/sample - loss: 98133384.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3696/5000
    500/500 [==============================] - 0s 92us/sample - loss: -33380478.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3697/5000
    500/500 [==============================] - 0s 84us/sample - loss: -108586231.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3698/5000
    500/500 [==============================] - 0s 94us/sample - loss: 102921214.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3699/5000
    500/500 [==============================] - 0s 78us/sample - loss: 9134298.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3700/5000
    500/500 [==============================] - 0s 84us/sample - loss: -9979832.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3701/5000
    500/500 [==============================] - 0s 84us/sample - loss: -36154140.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3702/5000
    500/500 [==============================] - 0s 94us/sample - loss: -128690584.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3703/5000
    500/500 [==============================] - 0s 82us/sample - loss: 25454063.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3704/5000
    500/500 [==============================] - 0s 82us/sample - loss: -110944861.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3705/5000
    500/500 [==============================] - 0s 92us/sample - loss: 80054188.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3706/5000
    500/500 [==============================] - 0s 94us/sample - loss: 26659675.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3707/5000
    500/500 [==============================] - 0s 82us/sample - loss: -12379362.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3708/5000
    500/500 [==============================] - 0s 78us/sample - loss: -70107761.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3709/5000
    500/500 [==============================] - 0s 86us/sample - loss: 89641997.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3710/5000
    500/500 [==============================] - 0s 96us/sample - loss: 10189223.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3711/5000
    500/500 [==============================] - 0s 78us/sample - loss: -91761614.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3712/5000
    500/500 [==============================] - 0s 72us/sample - loss: 124705954.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3713/5000
    500/500 [==============================] - 0s 82us/sample - loss: 81710404.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3714/5000
    500/500 [==============================] - 0s 82us/sample - loss: -201057796.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3715/5000
    500/500 [==============================] - 0s 84us/sample - loss: 41300362.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3716/5000
    500/500 [==============================] - 0s 86us/sample - loss: 24012476.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3717/5000
    500/500 [==============================] - 0s 78us/sample - loss: 23708102.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3718/5000
    500/500 [==============================] - 0s 82us/sample - loss: -72789614.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3719/5000
    500/500 [==============================] - 0s 84us/sample - loss: -107034306.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3720/5000
    500/500 [==============================] - 0s 82us/sample - loss: 79634127.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3721/5000
    500/500 [==============================] - 0s 84us/sample - loss: -64102617.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3722/5000
    500/500 [==============================] - 0s 90us/sample - loss: 58174557.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3723/5000
    500/500 [==============================] - 0s 78us/sample - loss: 34163811.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3724/5000
    500/500 [==============================] - 0s 82us/sample - loss: 141881548.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3725/5000
    500/500 [==============================] - 0s 80us/sample - loss: -106108963.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3726/5000
    500/500 [==============================] - 0s 92us/sample - loss: -100265612.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3727/5000
    500/500 [==============================] - 0s 84us/sample - loss: 166731239.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3728/5000
    500/500 [==============================] - 0s 80us/sample - loss: 15888384.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3729/5000
    500/500 [==============================] - 0s 88us/sample - loss: 11648383.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3730/5000
    500/500 [==============================] - 0s 94us/sample - loss: -86504517.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3731/5000
    500/500 [==============================] - 0s 78us/sample - loss: -81200187.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3732/5000
    500/500 [==============================] - 0s 84us/sample - loss: -144522724.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3733/5000
    500/500 [==============================] - 0s 82us/sample - loss: -11805792.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3734/5000
    500/500 [==============================] - 0s 86us/sample - loss: -93065902.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3735/5000
    500/500 [==============================] - 0s 80us/sample - loss: 256638317.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3736/5000
    500/500 [==============================] - 0s 82us/sample - loss: -30132884.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3737/5000
    500/500 [==============================] - 0s 88us/sample - loss: 137413825.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3738/5000
    500/500 [==============================] - 0s 84us/sample - loss: 26914363.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3739/5000
    500/500 [==============================] - 0s 82us/sample - loss: -50486383.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3740/5000
    500/500 [==============================] - 0s 84us/sample - loss: 170013452.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3741/5000
    500/500 [==============================] - 0s 90us/sample - loss: 87818980.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3742/5000
    500/500 [==============================] - 0s 76us/sample - loss: 97758860.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3743/5000
    500/500 [==============================] - 0s 82us/sample - loss: -63765020.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3744/5000
    500/500 [==============================] - 0s 84us/sample - loss: -7931647.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3745/5000
    500/500 [==============================] - 0s 88us/sample - loss: 77274987.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3746/5000
    500/500 [==============================] - 0s 82us/sample - loss: 146785827.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3747/5000
    500/500 [==============================] - 0s 84us/sample - loss: 33648440.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3748/5000
    500/500 [==============================] - 0s 88us/sample - loss: 88253669.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3749/5000
    500/500 [==============================] - 0s 86us/sample - loss: 87189076.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3750/5000
    500/500 [==============================] - 0s 86us/sample - loss: 139101931.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3751/5000
    500/500 [==============================] - 0s 86us/sample - loss: 35166578.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3752/5000
    500/500 [==============================] - 0s 92us/sample - loss: 49004724.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3753/5000
    500/500 [==============================] - 0s 76us/sample - loss: 5885678.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3754/5000
    500/500 [==============================] - 0s 92us/sample - loss: 151974266.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3755/5000
    500/500 [==============================] - 0s 90us/sample - loss: 208178828.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3756/5000
    500/500 [==============================] - 0s 86us/sample - loss: 29064229.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3757/5000
    500/500 [==============================] - 0s 80us/sample - loss: -5723068.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3758/5000
    500/500 [==============================] - 0s 86us/sample - loss: 35927266.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3759/5000
    500/500 [==============================] - 0s 94us/sample - loss: -31718099.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3760/5000
    500/500 [==============================] - 0s 80us/sample - loss: 132151082.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3761/5000
    500/500 [==============================] - 0s 90us/sample - loss: -113144691.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3762/5000
    500/500 [==============================] - 0s 78us/sample - loss: -142325096.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3763/5000
    500/500 [==============================] - 0s 74us/sample - loss: -124547102.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3764/5000
    500/500 [==============================] - 0s 80us/sample - loss: 165438859.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3765/5000
    500/500 [==============================] - 0s 86us/sample - loss: 128071765.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3766/5000
    500/500 [==============================] - 0s 80us/sample - loss: -60405311.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3767/5000
    500/500 [==============================] - 0s 78us/sample - loss: -31436269.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3768/5000
    500/500 [==============================] - 0s 84us/sample - loss: 55022061.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3769/5000
    500/500 [==============================] - 0s 90us/sample - loss: -56884737.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3770/5000
    500/500 [==============================] - 0s 84us/sample - loss: -259513185.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3771/5000
    500/500 [==============================] - 0s 82us/sample - loss: 113074530.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3772/5000
    500/500 [==============================] - 0s 86us/sample - loss: 85349500.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3773/5000
    500/500 [==============================] - 0s 92us/sample - loss: -34225073.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3774/5000
    500/500 [==============================] - 0s 78us/sample - loss: -23381867.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3775/5000
    500/500 [==============================] - 0s 75us/sample - loss: -147931321.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3776/5000
    500/500 [==============================] - 0s 84us/sample - loss: -123697661.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3777/5000
    500/500 [==============================] - 0s 82us/sample - loss: 84989218.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3778/5000
    500/500 [==============================] - 0s 82us/sample - loss: 139032522.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3779/5000
    500/500 [==============================] - 0s 84us/sample - loss: -43586803.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3780/5000
    500/500 [==============================] - 0s 86us/sample - loss: -99247163.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3781/5000
    500/500 [==============================] - 0s 94us/sample - loss: -97563968.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3782/5000
    500/500 [==============================] - 0s 78us/sample - loss: 39371817.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3783/5000
    500/500 [==============================] - 0s 96us/sample - loss: 43831978.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3784/5000
    500/500 [==============================] - 0s 88us/sample - loss: 87069250.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3785/5000
    500/500 [==============================] - 0s 90us/sample - loss: 134299609.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3786/5000
    500/500 [==============================] - 0s 78us/sample - loss: 17065156.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3787/5000
    500/500 [==============================] - 0s 94us/sample - loss: 105949878.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3788/5000
    500/500 [==============================] - 0s 78us/sample - loss: 208782473.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3789/5000
    500/500 [==============================] - 0s 75us/sample - loss: 83472285.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3790/5000
    500/500 [==============================] - 0s 88us/sample - loss: -111693878.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3791/5000
    500/500 [==============================] - 0s 92us/sample - loss: 190322033.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3792/5000
    500/500 [==============================] - 0s 98us/sample - loss: 21536671.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3793/5000
    500/500 [==============================] - 0s 78us/sample - loss: 31518534.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3794/5000
    500/500 [==============================] - 0s 94us/sample - loss: -122270409.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3795/5000
    500/500 [==============================] - 0s 98us/sample - loss: 49320659.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3796/5000
    500/500 [==============================] - 0s 86us/sample - loss: 203568553.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3797/5000
    500/500 [==============================] - 0s 82us/sample - loss: -40309736.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3798/5000
    500/500 [==============================] - 0s 86us/sample - loss: -118426276.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3799/5000
    500/500 [==============================] - 0s 96us/sample - loss: -167798988.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3800/5000
    500/500 [==============================] - 0s 94us/sample - loss: -42916171.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3801/5000
    500/500 [==============================] - 0s 76us/sample - loss: 17096941.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3802/5000
    500/500 [==============================] - 0s 86us/sample - loss: 26009760.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3803/5000
    500/500 [==============================] - 0s 88us/sample - loss: -79227081.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3804/5000
    500/500 [==============================] - 0s 80us/sample - loss: 67261015.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3805/5000
    500/500 [==============================] - 0s 77us/sample - loss: -11628637.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3806/5000
    500/500 [==============================] - 0s 78us/sample - loss: 118416881.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3807/5000
    500/500 [==============================] - 0s 82us/sample - loss: 13833753.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3808/5000
    500/500 [==============================] - 0s 82us/sample - loss: -133425776.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3809/5000
    500/500 [==============================] - 0s 84us/sample - loss: -191778535.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3810/5000
    500/500 [==============================] - 0s 84us/sample - loss: 55043051.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3811/5000
    500/500 [==============================] - 0s 86us/sample - loss: -56202490.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3812/5000
    500/500 [==============================] - 0s 92us/sample - loss: 109215431.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3813/5000
    500/500 [==============================] - 0s 80us/sample - loss: -15251118.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3814/5000
    500/500 [==============================] - 0s 77us/sample - loss: -202145887.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3815/5000
    500/500 [==============================] - 0s 82us/sample - loss: -49994272.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3816/5000
    500/500 [==============================] - 0s 84us/sample - loss: 110698739.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3817/5000
    500/500 [==============================] - 0s 84us/sample - loss: -429716.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3818/5000
    500/500 [==============================] - 0s 80us/sample - loss: 35295114.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3819/5000
    500/500 [==============================] - 0s 88us/sample - loss: 67229505.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3820/5000
    500/500 [==============================] - 0s 78us/sample - loss: -26437945.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3821/5000
    500/500 [==============================] - 0s 86us/sample - loss: -80170199.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3822/5000
    500/500 [==============================] - 0s 94us/sample - loss: -75928270.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3823/5000
    500/500 [==============================] - 0s 78us/sample - loss: -123494820.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3824/5000
    500/500 [==============================] - 0s 73us/sample - loss: 121698296.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3825/5000
    500/500 [==============================] - 0s 84us/sample - loss: 36878896.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3826/5000
    500/500 [==============================] - 0s 86us/sample - loss: 154256158.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3827/5000
    500/500 [==============================] - 0s 84us/sample - loss: 121273472.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3828/5000
    500/500 [==============================] - 0s 82us/sample - loss: 203017516.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3829/5000
    500/500 [==============================] - 0s 94us/sample - loss: 72146673.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3830/5000
    500/500 [==============================] - 0s 90us/sample - loss: -35146076.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3831/5000
    500/500 [==============================] - 0s 92us/sample - loss: 121826047.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3832/5000
    500/500 [==============================] - 0s 80us/sample - loss: 8857768.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3833/5000
    500/500 [==============================] - 0s 94us/sample - loss: 42793003.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3834/5000
    500/500 [==============================] - 0s 88us/sample - loss: -42773588.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3835/5000
    500/500 [==============================] - 0s 98us/sample - loss: -59574325.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3836/5000
    500/500 [==============================] - 0s 78us/sample - loss: -33411769.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3837/5000
    500/500 [==============================] - 0s 86us/sample - loss: -52635763.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3838/5000
    500/500 [==============================] - 0s 94us/sample - loss: 47543997.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3839/5000
    500/500 [==============================] - 0s 80us/sample - loss: 24319298.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3840/5000
    500/500 [==============================] - 0s 88us/sample - loss: 47300565.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3841/5000
    500/500 [==============================] - 0s 92us/sample - loss: -81684304.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3842/5000
    500/500 [==============================] - 0s 80us/sample - loss: 12316240.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3843/5000
    500/500 [==============================] - 0s 92us/sample - loss: -114344566.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3844/5000
    500/500 [==============================] - 0s 100us/sample - loss: -264205929.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3845/5000
    500/500 [==============================] - 0s 84us/sample - loss: 82436320.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3846/5000
    500/500 [==============================] - 0s 94us/sample - loss: 137353750.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3847/5000
    500/500 [==============================] - 0s 76us/sample - loss: -114231067.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3848/5000
    500/500 [==============================] - 0s 92us/sample - loss: -14617561.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3849/5000
    500/500 [==============================] - 0s 90us/sample - loss: 148411282.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3850/5000
    500/500 [==============================] - 0s 96us/sample - loss: 35131983.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3851/5000
    500/500 [==============================] - 0s 76us/sample - loss: -177569119.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3852/5000
    500/500 [==============================] - 0s 74us/sample - loss: -21995088.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3853/5000
    500/500 [==============================] - 0s 80us/sample - loss: 76755434.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3854/5000
    500/500 [==============================] - 0s 86us/sample - loss: -5792813.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3855/5000
    500/500 [==============================] - 0s 82us/sample - loss: 78555940.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3856/5000
    500/500 [==============================] - 0s 86us/sample - loss: 63644650.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3857/5000
    500/500 [==============================] - 0s 82us/sample - loss: -31563106.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3858/5000
    500/500 [==============================] - 0s 90us/sample - loss: 143421230.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3859/5000
    500/500 [==============================] - 0s 82us/sample - loss: -21407585.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3860/5000
    500/500 [==============================] - 0s 84us/sample - loss: -46911121.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3861/5000
    500/500 [==============================] - 0s 78us/sample - loss: -20599438.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3862/5000
    500/500 [==============================] - 0s 90us/sample - loss: 91836169.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3863/5000
    500/500 [==============================] - 0s 76us/sample - loss: -67324328.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3864/5000
    500/500 [==============================] - 0s 74us/sample - loss: 80546824.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3865/5000
    500/500 [==============================] - 0s 80us/sample - loss: 58811198.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3866/5000
    500/500 [==============================] - 0s 84us/sample - loss: -78601467.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3867/5000
    500/500 [==============================] - 0s 84us/sample - loss: -14910951.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3868/5000
    500/500 [==============================] - 0s 80us/sample - loss: -20909225.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3869/5000
    500/500 [==============================] - 0s 84us/sample - loss: -34553027.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3870/5000
    500/500 [==============================] - 0s 84us/sample - loss: 199905243.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3871/5000
    500/500 [==============================] - 0s 86us/sample - loss: -110771473.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3872/5000
    500/500 [==============================] - 0s 86us/sample - loss: 56632218.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3873/5000
    500/500 [==============================] - 0s 84us/sample - loss: -137741255.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3874/5000
    500/500 [==============================] - 0s 86us/sample - loss: -82958194.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3875/5000
    500/500 [==============================] - 0s 98us/sample - loss: 150092475.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3876/5000
    500/500 [==============================] - 0s 92us/sample - loss: 47809367.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3877/5000
    500/500 [==============================] - 0s 98us/sample - loss: -157638018.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3878/5000
    500/500 [==============================] - 0s 82us/sample - loss: 55286378.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3879/5000
    500/500 [==============================] - 0s 84us/sample - loss: -6462455.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3880/5000
    500/500 [==============================] - 0s 100us/sample - loss: 184050280.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3881/5000
    500/500 [==============================] - 0s 90us/sample - loss: 84658174.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3882/5000
    500/500 [==============================] - 0s 94us/sample - loss: -72637731.3000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3883/5000
    500/500 [==============================] - 0s 80us/sample - loss: -137346100.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3884/5000
    500/500 [==============================] - 0s 82us/sample - loss: -47009141.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3885/5000
    500/500 [==============================] - 0s 82us/sample - loss: -105745121.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3886/5000
    500/500 [==============================] - 0s 92us/sample - loss: 71143922.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3887/5000
    500/500 [==============================] - 0s 84us/sample - loss: -51103806.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3888/5000
    500/500 [==============================] - 0s 120us/sample - loss: 85079414.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3889/5000
    500/500 [==============================] - 0s 116us/sample - loss: 147414018.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3890/5000
    500/500 [==============================] - 0s 99us/sample - loss: -142259499.4120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3891/5000
    500/500 [==============================] - 0s 98us/sample - loss: -135649105.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3892/5000
    500/500 [==============================] - 0s 96us/sample - loss: -93316573.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3893/5000
    500/500 [==============================] - 0s 96us/sample - loss: 69327674.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3894/5000
    500/500 [==============================] - 0s 124us/sample - loss: 18926070.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3895/5000
    500/500 [==============================] - 0s 102us/sample - loss: 27320543.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3896/5000
    500/500 [==============================] - 0s 104us/sample - loss: -1013226.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3897/5000
    500/500 [==============================] - 0s 128us/sample - loss: -131491079.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3898/5000
    500/500 [==============================] - 0s 128us/sample - loss: 18143237.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3899/5000
    500/500 [==============================] - 0s 136us/sample - loss: 17687339.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3900/5000
    500/500 [==============================] - 0s 126us/sample - loss: -129924617.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3901/5000
    500/500 [==============================] - 0s 112us/sample - loss: 76147541.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3902/5000
    500/500 [==============================] - 0s 124us/sample - loss: -24303039.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3903/5000
    500/500 [==============================] - 0s 112us/sample - loss: 111428265.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3904/5000
    500/500 [==============================] - 0s 110us/sample - loss: 74584409.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3905/5000
    500/500 [==============================] - 0s 108us/sample - loss: -5200943.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3906/5000
    500/500 [==============================] - 0s 112us/sample - loss: 101104951.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3907/5000
    500/500 [==============================] - 0s 109us/sample - loss: 38793027.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3908/5000
    500/500 [==============================] - 0s 110us/sample - loss: 146983908.5640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3909/5000
    500/500 [==============================] - 0s 110us/sample - loss: 23666028.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3910/5000
    500/500 [==============================] - 0s 114us/sample - loss: 71254878.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3911/5000
    500/500 [==============================] - 0s 101us/sample - loss: 95328875.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3912/5000
    500/500 [==============================] - 0s 108us/sample - loss: 4336376.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3913/5000
    500/500 [==============================] - 0s 114us/sample - loss: -299473349.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3914/5000
    500/500 [==============================] - 0s 98us/sample - loss: 8448545.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3915/5000
    500/500 [==============================] - 0s 96us/sample - loss: -25355323.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3916/5000
    500/500 [==============================] - 0s 100us/sample - loss: -136831877.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3917/5000
    500/500 [==============================] - 0s 90us/sample - loss: 98750706.7860 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3918/5000
    500/500 [==============================] - 0s 94us/sample - loss: -8906618.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3919/5000
    500/500 [==============================] - 0s 103us/sample - loss: 109614066.8797 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3920/5000
    500/500 [==============================] - 0s 92us/sample - loss: 45473897.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3921/5000
    500/500 [==============================] - 0s 90us/sample - loss: 82405403.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3922/5000
    500/500 [==============================] - 0s 120us/sample - loss: 13888001.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3923/5000
    500/500 [==============================] - 0s 112us/sample - loss: 148065456.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3924/5000
    500/500 [==============================] - 0s 112us/sample - loss: 26237872.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3925/5000
    500/500 [==============================] - 0s 104us/sample - loss: -210486253.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3926/5000
    500/500 [==============================] - 0s 100us/sample - loss: 141167346.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3927/5000
    500/500 [==============================] - 0s 96us/sample - loss: -294655776.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3928/5000
    500/500 [==============================] - 0s 116us/sample - loss: 13108810.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3929/5000
    500/500 [==============================] - 0s 118us/sample - loss: 76946487.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3930/5000
    500/500 [==============================] - 0s 140us/sample - loss: 98170044.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3931/5000
    500/500 [==============================] - 0s 92us/sample - loss: 2983737.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3932/5000
    500/500 [==============================] - 0s 88us/sample - loss: 96518454.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3933/5000
    500/500 [==============================] - 0s 100us/sample - loss: -143378597.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3934/5000
    500/500 [==============================] - 0s 92us/sample - loss: 46727783.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3935/5000
    500/500 [==============================] - 0s 92us/sample - loss: 20301672.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3936/5000
    500/500 [==============================] - 0s 82us/sample - loss: -87007589.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3937/5000
    500/500 [==============================] - 0s 82us/sample - loss: 138519060.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3938/5000
    500/500 [==============================] - 0s 96us/sample - loss: 3389152.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3939/5000
    500/500 [==============================] - 0s 94us/sample - loss: 204391529.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3940/5000
    500/500 [==============================] - 0s 86us/sample - loss: -61430492.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3941/5000
    500/500 [==============================] - 0s 80us/sample - loss: 105276686.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3942/5000
    500/500 [==============================] - 0s 90us/sample - loss: 77717392.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3943/5000
    500/500 [==============================] - 0s 94us/sample - loss: 39638543.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3944/5000
    500/500 [==============================] - 0s 98us/sample - loss: -26699008.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3945/5000
    500/500 [==============================] - 0s 76us/sample - loss: -41754084.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3946/5000
    500/500 [==============================] - 0s 73us/sample - loss: -88567403.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3947/5000
    500/500 [==============================] - 0s 80us/sample - loss: 123100429.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3948/5000
    500/500 [==============================] - 0s 92us/sample - loss: 39596348.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3949/5000
    500/500 [==============================] - 0s 80us/sample - loss: 83723497.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3950/5000
    500/500 [==============================] - 0s 76us/sample - loss: -68086667.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3951/5000
    500/500 [==============================] - 0s 90us/sample - loss: -11330533.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3952/5000
    500/500 [==============================] - 0s 78us/sample - loss: -168352321.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3953/5000
    500/500 [==============================] - 0s 82us/sample - loss: 57105372.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3954/5000
    500/500 [==============================] - 0s 82us/sample - loss: -72659946.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3955/5000
    500/500 [==============================] - 0s 82us/sample - loss: 44103373.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3956/5000
    500/500 [==============================] - 0s 82us/sample - loss: -147494798.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3957/5000
    500/500 [==============================] - 0s 80us/sample - loss: -64489330.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3958/5000
    500/500 [==============================] - 0s 86us/sample - loss: 132670345.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3959/5000
    500/500 [==============================] - 0s 84us/sample - loss: 29703180.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3960/5000
    500/500 [==============================] - 0s 84us/sample - loss: 96055997.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3961/5000
    500/500 [==============================] - 0s 94us/sample - loss: -23532714.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3962/5000
    500/500 [==============================] - 0s 80us/sample - loss: -96380649.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3963/5000
    500/500 [==============================] - 0s 86us/sample - loss: -80060134.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3964/5000
    500/500 [==============================] - 0s 96us/sample - loss: 147754620.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3965/5000
    500/500 [==============================] - 0s 78us/sample - loss: 101048717.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3966/5000
    500/500 [==============================] - 0s 86us/sample - loss: 89335930.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3967/5000
    500/500 [==============================] - 0s 96us/sample - loss: -123665449.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3968/5000
    500/500 [==============================] - 0s 96us/sample - loss: -134243047.9355 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3969/5000
    500/500 [==============================] - 0s 78us/sample - loss: 27051777.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3970/5000
    500/500 [==============================] - 0s 94us/sample - loss: -17869129.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3971/5000
    500/500 [==============================] - 0s 90us/sample - loss: -33761205.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3972/5000
    500/500 [==============================] - 0s 96us/sample - loss: -23524316.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3973/5000
    500/500 [==============================] - 0s 80us/sample - loss: 48429621.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3974/5000
    500/500 [==============================] - 0s 78us/sample - loss: 32866676.8650 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3975/5000
    500/500 [==============================] - 0s 90us/sample - loss: -23321171.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3976/5000
    500/500 [==============================] - 0s 78us/sample - loss: -100198387.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3977/5000
    500/500 [==============================] - 0s 92us/sample - loss: -104027140.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3978/5000
    500/500 [==============================] - 0s 76us/sample - loss: 113619928.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3979/5000
    500/500 [==============================] - 0s 75us/sample - loss: -66990949.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3980/5000
    500/500 [==============================] - 0s 84us/sample - loss: 356587109.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3981/5000
    500/500 [==============================] - 0s 84us/sample - loss: -20845418.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3982/5000
    500/500 [==============================] - 0s 86us/sample - loss: -118255259.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3983/5000
    500/500 [==============================] - 0s 82us/sample - loss: -66760459.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3984/5000
    500/500 [==============================] - 0s 90us/sample - loss: -6660146.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3985/5000
    500/500 [==============================] - 0s 88us/sample - loss: -79790503.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3986/5000
    500/500 [==============================] - 0s 84us/sample - loss: -80496616.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3987/5000
    500/500 [==============================] - 0s 76us/sample - loss: 21897689.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3988/5000
    500/500 [==============================] - 0s 82us/sample - loss: 17898472.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3989/5000
    500/500 [==============================] - 0s 88us/sample - loss: 130731517.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3990/5000
    500/500 [==============================] - 0s 88us/sample - loss: 147060475.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3991/5000
    500/500 [==============================] - 0s 82us/sample - loss: -2337152.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3992/5000
    500/500 [==============================] - 0s 94us/sample - loss: 87872192.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3993/5000
    500/500 [==============================] - 0s 83us/sample - loss: -42779659.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3994/5000
    500/500 [==============================] - 0s 100us/sample - loss: 14744909.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3995/5000
    500/500 [==============================] - 0s 96us/sample - loss: -80752710.9445 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3996/5000
    500/500 [==============================] - 0s 82us/sample - loss: 125329757.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3997/5000
    500/500 [==============================] - 0s 84us/sample - loss: -15684540.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3998/5000
    500/500 [==============================] - 0s 100us/sample - loss: 42171757.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 3999/5000
    500/500 [==============================] - 0s 90us/sample - loss: 155262931.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4000/5000
    500/500 [==============================] - 0s 84us/sample - loss: 54011321.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4001/5000
    500/500 [==============================] - 0s 84us/sample - loss: -84493645.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4002/5000
    500/500 [==============================] - 0s 88us/sample - loss: -216153097.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4003/5000
    500/500 [==============================] - 0s 90us/sample - loss: 36970538.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4004/5000
    500/500 [==============================] - 0s 84us/sample - loss: 1459299.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4005/5000
    500/500 [==============================] - 0s 84us/sample - loss: -55747601.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4006/5000
    500/500 [==============================] - 0s 90us/sample - loss: 10022722.9720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4007/5000
    500/500 [==============================] - 0s 80us/sample - loss: 12703715.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4008/5000
    500/500 [==============================] - 0s 74us/sample - loss: 18686689.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4009/5000
    500/500 [==============================] - 0s 82us/sample - loss: -34862117.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4010/5000
    500/500 [==============================] - 0s 86us/sample - loss: 204833134.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4011/5000
    500/500 [==============================] - 0s 84us/sample - loss: 144314410.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4012/5000
    500/500 [==============================] - 0s 96us/sample - loss: -35111270.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4013/5000
    500/500 [==============================] - 0s 78us/sample - loss: 25301575.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4014/5000
    500/500 [==============================] - 0s 84us/sample - loss: 131288376.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4015/5000
    500/500 [==============================] - 0s 80us/sample - loss: -67811749.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4016/5000
    500/500 [==============================] - 0s 86us/sample - loss: -18578396.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4017/5000
    500/500 [==============================] - 0s 84us/sample - loss: -69948575.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4018/5000
    500/500 [==============================] - 0s 100us/sample - loss: 77375404.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4019/5000
    500/500 [==============================] - 0s 78us/sample - loss: -83507717.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4020/5000
    500/500 [==============================] - 0s 88us/sample - loss: 54953313.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4021/5000
    500/500 [==============================] - 0s 82us/sample - loss: -79682406.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4022/5000
    500/500 [==============================] - 0s 80us/sample - loss: -45775853.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4023/5000
    500/500 [==============================] - 0s 94us/sample - loss: 268412698.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4024/5000
    500/500 [==============================] - 0s 82us/sample - loss: 50795669.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4025/5000
    500/500 [==============================] - 0s 88us/sample - loss: -65741414.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4026/5000
    500/500 [==============================] - 0s 90us/sample - loss: -176904762.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4027/5000
    500/500 [==============================] - 0s 102us/sample - loss: 82897611.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4028/5000
    500/500 [==============================] - 0s 94us/sample - loss: -43728789.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4029/5000
    500/500 [==============================] - 0s 94us/sample - loss: -12629846.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4030/5000
    500/500 [==============================] - 0s 92us/sample - loss: -7530191.8120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4031/5000
    500/500 [==============================] - 0s 84us/sample - loss: -16271843.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4032/5000
    500/500 [==============================] - 0s 80us/sample - loss: 50352320.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4033/5000
    500/500 [==============================] - ETA: 0s - loss: 961893184.0000 - accuracy: 0.0000e+ - 0s 88us/sample - loss: 169535350.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4034/5000
    500/500 [==============================] - 0s 84us/sample - loss: -69655520.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4035/5000
    500/500 [==============================] - 0s 82us/sample - loss: 183664380.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4036/5000
    500/500 [==============================] - 0s 88us/sample - loss: -27873855.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4037/5000
    500/500 [==============================] - 0s 78us/sample - loss: 86172388.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4038/5000
    500/500 [==============================] - 0s 90us/sample - loss: -56691410.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4039/5000
    500/500 [==============================] - 0s 84us/sample - loss: -37596708.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4040/5000
    500/500 [==============================] - 0s 96us/sample - loss: 84644440.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4041/5000
    500/500 [==============================] - 0s 76us/sample - loss: 67982538.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4042/5000
    500/500 [==============================] - 0s 82us/sample - loss: 1171225.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4043/5000
    500/500 [==============================] - 0s 80us/sample - loss: -74186638.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4044/5000
    500/500 [==============================] - 0s 84us/sample - loss: 41569777.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4045/5000
    500/500 [==============================] - 0s 82us/sample - loss: 76847672.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4046/5000
    500/500 [==============================] - 0s 84us/sample - loss: -105692934.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4047/5000
    500/500 [==============================] - 0s 86us/sample - loss: -70523010.4360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4048/5000
    500/500 [==============================] - 0s 84us/sample - loss: 153098801.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4049/5000
    500/500 [==============================] - 0s 98us/sample - loss: 149327365.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4050/5000
    500/500 [==============================] - 0s 88us/sample - loss: -197727313.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4051/5000
    500/500 [==============================] - 0s 80us/sample - loss: 54683454.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4052/5000
    500/500 [==============================] - 0s 80us/sample - loss: -111519836.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4053/5000
    500/500 [==============================] - 0s 78us/sample - loss: -76051307.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4054/5000
    500/500 [==============================] - 0s 80us/sample - loss: 198154611.8740 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4055/5000
    500/500 [==============================] - 0s 80us/sample - loss: -90186099.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4056/5000
    500/500 [==============================] - 0s 82us/sample - loss: 29656421.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4057/5000
    500/500 [==============================] - 0s 92us/sample - loss: -35236085.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4058/5000
    500/500 [==============================] - 0s 80us/sample - loss: -25409137.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4059/5000
    500/500 [==============================] - 0s 84us/sample - loss: 78253446.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4060/5000
    500/500 [==============================] - 0s 86us/sample - loss: 102935690.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4061/5000
    500/500 [==============================] - 0s 82us/sample - loss: -115290418.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4062/5000
    500/500 [==============================] - 0s 90us/sample - loss: -218416597.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4063/5000
    500/500 [==============================] - 0s 92us/sample - loss: -27607016.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4064/5000
    500/500 [==============================] - 0s 76us/sample - loss: 90468021.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4065/5000
    500/500 [==============================] - 0s 75us/sample - loss: -10999145.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4066/5000
    500/500 [==============================] - 0s 84us/sample - loss: -12149500.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4067/5000
    500/500 [==============================] - 0s 84us/sample - loss: 103079661.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4068/5000
    500/500 [==============================] - 0s 84us/sample - loss: 78774761.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4069/5000
    500/500 [==============================] - 0s 90us/sample - loss: 10120190.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4070/5000
    500/500 [==============================] - 0s 80us/sample - loss: -115639800.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4071/5000
    500/500 [==============================] - 0s 86us/sample - loss: 26196935.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4072/5000
    500/500 [==============================] - 0s 82us/sample - loss: 76973545.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4073/5000
    500/500 [==============================] - 0s 88us/sample - loss: 19322854.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4074/5000
    500/500 [==============================] - 0s 94us/sample - loss: -114229419.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4075/5000
    500/500 [==============================] - 0s 90us/sample - loss: -94452310.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4076/5000
    500/500 [==============================] - 0s 92us/sample - loss: 138734515.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4077/5000
    500/500 [==============================] - 0s 80us/sample - loss: -18675898.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4078/5000
    500/500 [==============================] - 0s 132us/sample - loss: 14472773.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4079/5000
    500/500 [==============================] - 0s 102us/sample - loss: 40321092.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4080/5000
    500/500 [==============================] - 0s 110us/sample - loss: 10519351.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4081/5000
    500/500 [==============================] - 0s 84us/sample - loss: 47364793.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4082/5000
    500/500 [==============================] - 0s 86us/sample - loss: -6080524.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4083/5000
    500/500 [==============================] - 0s 90us/sample - loss: 105481983.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4084/5000
    500/500 [==============================] - 0s 96us/sample - loss: -84916801.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4085/5000
    500/500 [==============================] - 0s 78us/sample - loss: 24389999.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4086/5000
    500/500 [==============================] - 0s 73us/sample - loss: 116732678.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4087/5000
    500/500 [==============================] - 0s 84us/sample - loss: -44416900.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4088/5000
    500/500 [==============================] - 0s 80us/sample - loss: 55312724.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4089/5000
    500/500 [==============================] - 0s 80us/sample - loss: -41014218.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4090/5000
    500/500 [==============================] - 0s 80us/sample - loss: 198620232.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4091/5000
    500/500 [==============================] - 0s 90us/sample - loss: -65923351.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4092/5000
    500/500 [==============================] - 0s 78us/sample - loss: 22829878.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4093/5000
    500/500 [==============================] - 0s 80us/sample - loss: -17561096.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4094/5000
    500/500 [==============================] - 0s 78us/sample - loss: -16046192.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4095/5000
    500/500 [==============================] - 0s 80us/sample - loss: -59503067.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4096/5000
    500/500 [==============================] - 0s 88us/sample - loss: -30929430.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4097/5000
    500/500 [==============================] - 0s 80us/sample - loss: -75858940.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4098/5000
    500/500 [==============================] - 0s 88us/sample - loss: -143666309.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4099/5000
    500/500 [==============================] - 0s 82us/sample - loss: -14123454.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4100/5000
    500/500 [==============================] - 0s 90us/sample - loss: 174087739.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4101/5000
    500/500 [==============================] - 0s 94us/sample - loss: 79773312.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4102/5000
    500/500 [==============================] - 0s 82us/sample - loss: -75783161.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4103/5000
    500/500 [==============================] - 0s 80us/sample - loss: -111167107.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4104/5000
    500/500 [==============================] - 0s 88us/sample - loss: 66044215.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4105/5000
    500/500 [==============================] - 0s 78us/sample - loss: 101658116.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4106/5000
    500/500 [==============================] - 0s 86us/sample - loss: -86767804.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4107/5000
    500/500 [==============================] - 0s 78us/sample - loss: 40600798.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4108/5000
    500/500 [==============================] - 0s 84us/sample - loss: -158368384.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4109/5000
    500/500 [==============================] - 0s 86us/sample - loss: 35783697.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4110/5000
    500/500 [==============================] - 0s 86us/sample - loss: -70047828.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4111/5000
    500/500 [==============================] - 0s 82us/sample - loss: -1678558.5260 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4112/5000
    500/500 [==============================] - 0s 92us/sample - loss: -113047105.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4113/5000
    500/500 [==============================] - 0s 78us/sample - loss: 49870657.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4114/5000
    500/500 [==============================] - 0s 81us/sample - loss: 19367823.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4115/5000
    500/500 [==============================] - 0s 86us/sample - loss: -125916818.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4116/5000
    500/500 [==============================] - 0s 88us/sample - loss: 114552157.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4117/5000
    500/500 [==============================] - 0s 86us/sample - loss: 27803345.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4118/5000
    500/500 [==============================] - 0s 94us/sample - loss: -188019629.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4119/5000
    500/500 [==============================] - 0s 76us/sample - loss: 78044043.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4120/5000
    500/500 [==============================] - 0s 94us/sample - loss: 173134418.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4121/5000
    500/500 [==============================] - 0s 88us/sample - loss: -160173171.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4122/5000
    500/500 [==============================] - 0s 94us/sample - loss: 132294717.9500 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4123/5000
    500/500 [==============================] - 0s 74us/sample - loss: 136752548.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4124/5000
    500/500 [==============================] - 0s 88us/sample - loss: 10716355.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4125/5000
    500/500 [==============================] - 0s 92us/sample - loss: 88562860.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4126/5000
    500/500 [==============================] - 0s 76us/sample - loss: -9712090.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4127/5000
    500/500 [==============================] - 0s 90us/sample - loss: 84235767.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4128/5000
    500/500 [==============================] - 0s 90us/sample - loss: -51905312.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4129/5000
    500/500 [==============================] - 0s 90us/sample - loss: 169867443.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4130/5000
    500/500 [==============================] - 0s 92us/sample - loss: -75907111.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4131/5000
    500/500 [==============================] - 0s 80us/sample - loss: 35582106.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4132/5000
    500/500 [==============================] - 0s 90us/sample - loss: -85901119.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4133/5000
    500/500 [==============================] - 0s 76us/sample - loss: 64349519.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4134/5000
    500/500 [==============================] - 0s 77us/sample - loss: -16492312.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4135/5000
    500/500 [==============================] - 0s 82us/sample - loss: -96192382.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4136/5000
    500/500 [==============================] - 0s 84us/sample - loss: 177127465.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4137/5000
    500/500 [==============================] - 0s 80us/sample - loss: -67639363.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4138/5000
    500/500 [==============================] - 0s 82us/sample - loss: -179928699.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4139/5000
    500/500 [==============================] - 0s 92us/sample - loss: -12750311.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4140/5000
    500/500 [==============================] - 0s 74us/sample - loss: -61570922.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4141/5000
    500/500 [==============================] - 0s 82us/sample - loss: -4263689.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4142/5000
    500/500 [==============================] - 0s 82us/sample - loss: 151187126.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4143/5000
    500/500 [==============================] - 0s 86us/sample - loss: 79713136.9640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4144/5000
    500/500 [==============================] - 0s 88us/sample - loss: 73482959.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4145/5000
    500/500 [==============================] - 0s 88us/sample - loss: 38221791.6120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4146/5000
    500/500 [==============================] - 0s 94us/sample - loss: 84278518.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4147/5000
    500/500 [==============================] - 0s 88us/sample - loss: 169897836.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4148/5000
    500/500 [==============================] - 0s 84us/sample - loss: -45583214.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4149/5000
    500/500 [==============================] - 0s 80us/sample - loss: 119764222.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4150/5000
    500/500 [==============================] - 0s 84us/sample - loss: 103943741.6940 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4151/5000
    500/500 [==============================] - 0s 98us/sample - loss: -249047547.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4152/5000
    500/500 [==============================] - 0s 92us/sample - loss: -160151415.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4153/5000
    500/500 [==============================] - 0s 94us/sample - loss: -19967590.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4154/5000
    500/500 [==============================] - 0s 78us/sample - loss: -153800139.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4155/5000
    500/500 [==============================] - 0s 76us/sample - loss: 71718852.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4156/5000
    500/500 [==============================] - 0s 90us/sample - loss: 26588335.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4157/5000
    500/500 [==============================] - 0s 78us/sample - loss: 23274617.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4158/5000
    500/500 [==============================] - 0s 82us/sample - loss: 66409542.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4159/5000
    500/500 [==============================] - 0s 82us/sample - loss: -198133162.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4160/5000
    500/500 [==============================] - 0s 96us/sample - loss: 120921511.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4161/5000
    500/500 [==============================] - 0s 88us/sample - loss: -46453220.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4162/5000
    500/500 [==============================] - 0s 84us/sample - loss: -137979663.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4163/5000
    500/500 [==============================] - 0s 86us/sample - loss: -743950.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4164/5000
    500/500 [==============================] - 0s 88us/sample - loss: -65398684.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4165/5000
    500/500 [==============================] - 0s 88us/sample - loss: 42219125.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4166/5000
    500/500 [==============================] - 0s 84us/sample - loss: 157039569.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4167/5000
    500/500 [==============================] - 0s 94us/sample - loss: -37399782.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4168/5000
    500/500 [==============================] - 0s 96us/sample - loss: -610913.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4169/5000
    500/500 [==============================] - 0s 96us/sample - loss: 19371372.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4170/5000
    500/500 [==============================] - 0s 92us/sample - loss: 141624452.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4171/5000
    500/500 [==============================] - 0s 96us/sample - loss: -23250210.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4172/5000
    500/500 [==============================] - 0s 92us/sample - loss: 26559215.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4173/5000
    500/500 [==============================] - 0s 94us/sample - loss: 112530424.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4174/5000
    500/500 [==============================] - 0s 80us/sample - loss: -12517646.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4175/5000
    500/500 [==============================] - 0s 79us/sample - loss: -32197497.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4176/5000
    500/500 [==============================] - 0s 82us/sample - loss: -110663425.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4177/5000
    500/500 [==============================] - 0s 88us/sample - loss: -6411165.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4178/5000
    500/500 [==============================] - 0s 84us/sample - loss: 99336406.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4179/5000
    500/500 [==============================] - 0s 84us/sample - loss: -85491458.1640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4180/5000
    500/500 [==============================] - 0s 80us/sample - loss: 26606796.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4181/5000
    500/500 [==============================] - 0s 78us/sample - loss: -189029443.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4182/5000
    500/500 [==============================] - 0s 82us/sample - loss: 134159208.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4183/5000
    500/500 [==============================] - 0s 80us/sample - loss: 19538872.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4184/5000
    500/500 [==============================] - 0s 80us/sample - loss: 36656326.6560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4185/5000
    500/500 [==============================] - 0s 84us/sample - loss: 168050017.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4186/5000
    500/500 [==============================] - 0s 82us/sample - loss: 23518001.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4187/5000
    500/500 [==============================] - 0s 92us/sample - loss: 41965969.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4188/5000
    500/500 [==============================] - 0s 86us/sample - loss: 19162071.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4189/5000
    500/500 [==============================] - 0s 82us/sample - loss: 162654100.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4190/5000
    500/500 [==============================] - 0s 86us/sample - loss: 65744666.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4191/5000
    500/500 [==============================] - 0s 82us/sample - loss: 82957098.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4192/5000
    500/500 [==============================] - 0s 84us/sample - loss: -39658649.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4193/5000
    500/500 [==============================] - 0s 96us/sample - loss: -128448134.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4194/5000
    500/500 [==============================] - 0s 84us/sample - loss: 86791654.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4195/5000
    500/500 [==============================] - 0s 98us/sample - loss: 122282635.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4196/5000
    500/500 [==============================] - 0s 82us/sample - loss: 70662120.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4197/5000
    500/500 [==============================] - 0s 96us/sample - loss: 67566859.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4198/5000
    500/500 [==============================] - 0s 88us/sample - loss: 753796.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4199/5000
    500/500 [==============================] - 0s 94us/sample - loss: -55043188.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4200/5000
    500/500 [==============================] - 0s 78us/sample - loss: -60997453.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4201/5000
    500/500 [==============================] - 0s 84us/sample - loss: 55645675.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4202/5000
    500/500 [==============================] - 0s 92us/sample - loss: 30407139.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4203/5000
    500/500 [==============================] - 0s 78us/sample - loss: 25854656.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4204/5000
    500/500 [==============================] - 0s 92us/sample - loss: -29402981.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4205/5000
    500/500 [==============================] - 0s 76us/sample - loss: -37432375.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4206/5000
    500/500 [==============================] - 0s 77us/sample - loss: 28181921.0280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4207/5000
    500/500 [==============================] - 0s 84us/sample - loss: -21858467.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4208/5000
    500/500 [==============================] - 0s 88us/sample - loss: 94008591.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4209/5000
    500/500 [==============================] - 0s 86us/sample - loss: 105037180.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4210/5000
    500/500 [==============================] - 0s 82us/sample - loss: -126577441.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4211/5000
    500/500 [==============================] - 0s 86us/sample - loss: -5450266.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4212/5000
    500/500 [==============================] - ETA: 0s - loss: 544907200.0000 - accuracy: 0.0000e+ - 0s 88us/sample - loss: 65077516.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4213/5000
    500/500 [==============================] - 0s 94us/sample - loss: 137010525.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4214/5000
    500/500 [==============================] - 0s 80us/sample - loss: 272599195.6440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4215/5000
    500/500 [==============================] - 0s 98us/sample - loss: -64411559.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4216/5000
    500/500 [==============================] - 0s 84us/sample - loss: -84073496.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4217/5000
    500/500 [==============================] - 0s 84us/sample - loss: 93739881.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4218/5000
    500/500 [==============================] - 0s 88us/sample - loss: -125439077.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4219/5000
    500/500 [==============================] - 0s 88us/sample - loss: -59466938.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4220/5000
    500/500 [==============================] - 0s 84us/sample - loss: 74163153.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4221/5000
    500/500 [==============================] - 0s 86us/sample - loss: -77681042.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4222/5000
    500/500 [==============================] - 0s 86us/sample - loss: 63357511.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4223/5000
    500/500 [==============================] - 0s 86us/sample - loss: 11787122.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4224/5000
    500/500 [==============================] - 0s 90us/sample - loss: 89227875.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4225/5000
    500/500 [==============================] - 0s 84us/sample - loss: -28133996.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4226/5000
    500/500 [==============================] - 0s 82us/sample - loss: -17769265.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4227/5000
    500/500 [==============================] - 0s 82us/sample - loss: 38613830.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4228/5000
    500/500 [==============================] - 0s 90us/sample - loss: -21384344.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4229/5000
    500/500 [==============================] - 0s 78us/sample - loss: 45451552.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4230/5000
    500/500 [==============================] - 0s 76us/sample - loss: -48018264.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4231/5000
    500/500 [==============================] - 0s 86us/sample - loss: 173956265.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4232/5000
    500/500 [==============================] - 0s 88us/sample - loss: -42282380.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4233/5000
    500/500 [==============================] - 0s 82us/sample - loss: 32089067.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4234/5000
    500/500 [==============================] - 0s 84us/sample - loss: -56657572.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4235/5000
    500/500 [==============================] - 0s 110us/sample - loss: -73950348.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4236/5000
    500/500 [==============================] - 0s 105us/sample - loss: -21264856.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4237/5000
    500/500 [==============================] - 0s 102us/sample - loss: -14986576.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4238/5000
    500/500 [==============================] - 0s 92us/sample - loss: 62082713.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4239/5000
    500/500 [==============================] - 0s 96us/sample - loss: 49624783.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4240/5000
    500/500 [==============================] - 0s 98us/sample - loss: -106609791.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4241/5000
    500/500 [==============================] - 0s 102us/sample - loss: 11645143.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4242/5000
    500/500 [==============================] - 0s 94us/sample - loss: -4756905.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4243/5000
    500/500 [==============================] - 0s 102us/sample - loss: -26669873.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4244/5000
    500/500 [==============================] - 0s 98us/sample - loss: -22417548.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4245/5000
    500/500 [==============================] - 0s 102us/sample - loss: 57043196.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4246/5000
    500/500 [==============================] - 0s 92us/sample - loss: -100060412.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4247/5000
    500/500 [==============================] - 0s 99us/sample - loss: -62773504.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4248/5000
    500/500 [==============================] - 0s 104us/sample - loss: -1535912.8280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4249/5000
    500/500 [==============================] - 0s 102us/sample - loss: 54462580.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4250/5000
    500/500 [==============================] - 0s 90us/sample - loss: 101716591.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4251/5000
    500/500 [==============================] - 0s 98us/sample - loss: 37984127.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4252/5000
    500/500 [==============================] - 0s 95us/sample - loss: 155101543.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4253/5000
    500/500 [==============================] - 0s 94us/sample - loss: 90214718.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4254/5000
    500/500 [==============================] - 0s 94us/sample - loss: 135836731.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4255/5000
    500/500 [==============================] - 0s 116us/sample - loss: 23362705.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4256/5000
    500/500 [==============================] - 0s 108us/sample - loss: -5009809.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4257/5000
    500/500 [==============================] - 0s 106us/sample - loss: -44179151.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4258/5000
    500/500 [==============================] - 0s 100us/sample - loss: -34421554.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4259/5000
    500/500 [==============================] - 0s 100us/sample - loss: -90166311.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4260/5000
    500/500 [==============================] - 0s 98us/sample - loss: 19041767.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4261/5000
    500/500 [==============================] - 0s 102us/sample - loss: 62296008.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4262/5000
    500/500 [==============================] - 0s 90us/sample - loss: 12149820.4660 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4263/5000
    500/500 [==============================] - 0s 92us/sample - loss: -124844267.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4264/5000
    500/500 [==============================] - 0s 90us/sample - loss: -78865639.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4265/5000
    500/500 [==============================] - 0s 90us/sample - loss: 6992038.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4266/5000
    500/500 [==============================] - 0s 90us/sample - loss: 102191823.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4267/5000
    500/500 [==============================] - 0s 102us/sample - loss: -162437169.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4268/5000
    500/500 [==============================] - 0s 92us/sample - loss: -33190378.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4269/5000
    500/500 [==============================] - 0s 93us/sample - loss: -32657316.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4270/5000
    500/500 [==============================] - 0s 94us/sample - loss: 7024996.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4271/5000
    500/500 [==============================] - 0s 106us/sample - loss: 243197944.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4272/5000
    500/500 [==============================] - 0s 98us/sample - loss: -10241329.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4273/5000
    500/500 [==============================] - 0s 99us/sample - loss: 47569763.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4274/5000
    500/500 [==============================] - 0s 112us/sample - loss: 25844767.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4275/5000
    500/500 [==============================] - 0s 84us/sample - loss: 8109250.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4276/5000
    500/500 [==============================] - 0s 82us/sample - loss: -10007105.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4277/5000
    500/500 [==============================] - 0s 80us/sample - loss: 35171398.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4278/5000
    500/500 [==============================] - 0s 92us/sample - loss: 125389123.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4279/5000
    500/500 [==============================] - 0s 82us/sample - loss: -123891470.3380 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4280/5000
    500/500 [==============================] - 0s 76us/sample - loss: 67539347.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4281/5000
    500/500 [==============================] - 0s 82us/sample - loss: 158218944.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4282/5000
    500/500 [==============================] - 0s 84us/sample - loss: 24649488.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4283/5000
    500/500 [==============================] - 0s 86us/sample - loss: 154453560.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4284/5000
    500/500 [==============================] - 0s 90us/sample - loss: -126458165.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4285/5000
    500/500 [==============================] - 0s 78us/sample - loss: 5179498.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4286/5000
    500/500 [==============================] - 0s 86us/sample - loss: 64963517.2840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4287/5000
    500/500 [==============================] - 0s 100us/sample - loss: -1133536.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4288/5000
    500/500 [==============================] - 0s 84us/sample - loss: 60290902.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4289/5000
    500/500 [==============================] - 0s 82us/sample - loss: -18063439.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4290/5000
    500/500 [==============================] - 0s 92us/sample - loss: 39605012.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4291/5000
    500/500 [==============================] - 0s 92us/sample - loss: -63396633.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4292/5000
    500/500 [==============================] - 0s 82us/sample - loss: 86751915.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4293/5000
    500/500 [==============================] - 0s 78us/sample - loss: -131050036.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4294/5000
    500/500 [==============================] - 0s 80us/sample - loss: 88743656.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4295/5000
    500/500 [==============================] - 0s 76us/sample - loss: 63773440.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4296/5000
    500/500 [==============================] - 0s 84us/sample - loss: -109104608.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4297/5000
    500/500 [==============================] - 0s 86us/sample - loss: 33324920.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4298/5000
    500/500 [==============================] - 0s 96us/sample - loss: -47168992.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4299/5000
    500/500 [==============================] - 0s 80us/sample - loss: 97639595.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4300/5000
    500/500 [==============================] - 0s 77us/sample - loss: 170091888.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4301/5000
    500/500 [==============================] - 0s 112us/sample - loss: -48665630.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4302/5000
    500/500 [==============================] - 0s 98us/sample - loss: -11615705.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4303/5000
    500/500 [==============================] - 0s 96us/sample - loss: 100334995.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4304/5000
    500/500 [==============================] - 0s 76us/sample - loss: 35667150.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4305/5000
    500/500 [==============================] - 0s 84us/sample - loss: 83192874.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4306/5000
    500/500 [==============================] - 0s 84us/sample - loss: -43961917.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4307/5000
    500/500 [==============================] - 0s 90us/sample - loss: 63717563.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4308/5000
    500/500 [==============================] - 0s 84us/sample - loss: -29781990.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4309/5000
    500/500 [==============================] - 0s 86us/sample - loss: 137479844.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4310/5000
    500/500 [==============================] - 0s 84us/sample - loss: 87199765.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4311/5000
    500/500 [==============================] - 0s 94us/sample - loss: 87364394.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4312/5000
    500/500 [==============================] - 0s 90us/sample - loss: -107678610.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4313/5000
    500/500 [==============================] - 0s 90us/sample - loss: 45773711.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4314/5000
    500/500 [==============================] - 0s 84us/sample - loss: 50552747.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4315/5000
    500/500 [==============================] - 0s 78us/sample - loss: 155547720.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4316/5000
    500/500 [==============================] - 0s 92us/sample - loss: 67546019.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4317/5000
    500/500 [==============================] - 0s 80us/sample - loss: 224796191.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4318/5000
    500/500 [==============================] - 0s 86us/sample - loss: 103418846.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4319/5000
    500/500 [==============================] - 0s 84us/sample - loss: 65685081.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4320/5000
    500/500 [==============================] - 0s 84us/sample - loss: -262019546.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4321/5000
    500/500 [==============================] - 0s 84us/sample - loss: 89005165.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4322/5000
    500/500 [==============================] - 0s 86us/sample - loss: 66207766.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4323/5000
    500/500 [==============================] - 0s 82us/sample - loss: -100484530.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4324/5000
    500/500 [==============================] - 0s 80us/sample - loss: -80970285.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4325/5000
    500/500 [==============================] - 0s 86us/sample - loss: -206811840.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4326/5000
    500/500 [==============================] - 0s 94us/sample - loss: -101320495.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4327/5000
    500/500 [==============================] - 0s 84us/sample - loss: 74762505.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4328/5000
    500/500 [==============================] - 0s 90us/sample - loss: -192850278.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4329/5000
    500/500 [==============================] - 0s 96us/sample - loss: 63489734.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4330/5000
    500/500 [==============================] - 0s 82us/sample - loss: 100725149.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4331/5000
    500/500 [==============================] - 0s 84us/sample - loss: 60598039.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4332/5000
    500/500 [==============================] - 0s 88us/sample - loss: 48424766.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4333/5000
    500/500 [==============================] - 0s 80us/sample - loss: -9060275.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4334/5000
    500/500 [==============================] - 0s 84us/sample - loss: -136781320.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4335/5000
    500/500 [==============================] - ETA: 0s - loss: -18207912.0000 - accuracy: 0.0000e+ - 0s 88us/sample - loss: -9542665.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4336/5000
    500/500 [==============================] - 0s 86us/sample - loss: 28907610.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4337/5000
    500/500 [==============================] - 0s 86us/sample - loss: -47979119.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4338/5000
    500/500 [==============================] - ETA: 0s - loss: -43765084.0000 - accuracy: 0.0000e+ - 0s 96us/sample - loss: 107728788.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4339/5000
    500/500 [==============================] - 0s 76us/sample - loss: 110737392.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4340/5000
    500/500 [==============================] - 0s 94us/sample - loss: 43315086.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4341/5000
    500/500 [==============================] - 0s 86us/sample - loss: 55623782.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4342/5000
    500/500 [==============================] - 0s 94us/sample - loss: 19935336.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4343/5000
    500/500 [==============================] - 0s 78us/sample - loss: -34222825.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4344/5000
    500/500 [==============================] - 0s 94us/sample - loss: -48343328.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4345/5000
    500/500 [==============================] - 0s 76us/sample - loss: 163285867.0100 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4346/5000
    500/500 [==============================] - 0s 77us/sample - loss: -15805844.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4347/5000
    500/500 [==============================] - 0s 86us/sample - loss: 94278244.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4348/5000
    500/500 [==============================] - 0s 86us/sample - loss: -50373310.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4349/5000
    500/500 [==============================] - 0s 78us/sample - loss: 71506265.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4350/5000
    500/500 [==============================] - 0s 94us/sample - loss: -18336557.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4351/5000
    500/500 [==============================] - 0s 78us/sample - loss: -51764042.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4352/5000
    500/500 [==============================] - 0s 84us/sample - loss: 104535380.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4353/5000
    500/500 [==============================] - 0s 86us/sample - loss: -75007322.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4354/5000
    500/500 [==============================] - 0s 88us/sample - loss: -55604085.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4355/5000
    500/500 [==============================] - 0s 82us/sample - loss: 129148467.8700 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4356/5000
    500/500 [==============================] - 0s 96us/sample - loss: 7338773.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4357/5000
    500/500 [==============================] - 0s 76us/sample - loss: -47113971.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4358/5000
    500/500 [==============================] - 0s 94us/sample - loss: 160369965.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4359/5000
    500/500 [==============================] - 0s 88us/sample - loss: 186986438.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4360/5000
    500/500 [==============================] - 0s 86us/sample - loss: 9795515.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4361/5000
    500/500 [==============================] - 0s 80us/sample - loss: -29280034.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4362/5000
    500/500 [==============================] - 0s 92us/sample - loss: 188197473.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4363/5000
    500/500 [==============================] - 0s 80us/sample - loss: 44876279.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4364/5000
    500/500 [==============================] - 0s 75us/sample - loss: 131951999.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4365/5000
    500/500 [==============================] - 0s 84us/sample - loss: 51072118.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4366/5000
    500/500 [==============================] - 0s 84us/sample - loss: 37033218.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4367/5000
    500/500 [==============================] - 0s 84us/sample - loss: -169121225.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4368/5000
    500/500 [==============================] - 0s 80us/sample - loss: 80152364.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4369/5000
    500/500 [==============================] - 0s 88us/sample - loss: -36050045.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4370/5000
    500/500 [==============================] - 0s 80us/sample - loss: 2866273.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4371/5000
    500/500 [==============================] - ETA: 0s - loss: -570843904.0000 - accuracy: 0.0000e+0 - 0s 80us/sample - loss: -47068188.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4372/5000
    500/500 [==============================] - ETA: 0s - loss: -346322784.0000 - accuracy: 0.0000e+0 - 0s 90us/sample - loss: 91643894.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4373/5000
    500/500 [==============================] - 0s 76us/sample - loss: -135945191.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4374/5000
    500/500 [==============================] - 0s 84us/sample - loss: 98382685.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4375/5000
    500/500 [==============================] - 0s 82us/sample - loss: 50271250.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4376/5000
    500/500 [==============================] - 0s 83us/sample - loss: -50317977.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4377/5000
    500/500 [==============================] - 0s 96us/sample - loss: -83550706.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4378/5000
    500/500 [==============================] - 0s 76us/sample - loss: -2350406.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4379/5000
    500/500 [==============================] - 0s 75us/sample - loss: -95700014.9480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4380/5000
    500/500 [==============================] - 0s 82us/sample - loss: 113538685.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4381/5000
    500/500 [==============================] - 0s 86us/sample - loss: 23660578.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4382/5000
    500/500 [==============================] - 0s 82us/sample - loss: -144876767.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4383/5000
    500/500 [==============================] - 0s 84us/sample - loss: -100440332.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4384/5000
    500/500 [==============================] - 0s 86us/sample - loss: -2644894.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4385/5000
    500/500 [==============================] - 0s 82us/sample - loss: -7707182.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4386/5000
    500/500 [==============================] - 0s 88us/sample - loss: 134293078.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4387/5000
    500/500 [==============================] - 0s 86us/sample - loss: 66016687.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4388/5000
    500/500 [==============================] - 0s 84us/sample - loss: 61552309.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4389/5000
    500/500 [==============================] - 0s 84us/sample - loss: -17979172.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4390/5000
    500/500 [==============================] - 0s 96us/sample - loss: 146344418.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4391/5000
    500/500 [==============================] - 0s 78us/sample - loss: -16007623.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4392/5000
    500/500 [==============================] - 0s 79us/sample - loss: 4695128.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4393/5000
    500/500 [==============================] - 0s 82us/sample - loss: -220996291.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4394/5000
    500/500 [==============================] - 0s 84us/sample - loss: 31048749.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4395/5000
    500/500 [==============================] - 0s 86us/sample - loss: 71925554.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4396/5000
    500/500 [==============================] - 0s 96us/sample - loss: 50486182.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4397/5000
    500/500 [==============================] - 0s 80us/sample - loss: 29963508.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4398/5000
    500/500 [==============================] - 0s 92us/sample - loss: 43626399.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4399/5000
    500/500 [==============================] - 0s 88us/sample - loss: -161865600.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4400/5000
    500/500 [==============================] - 0s 90us/sample - loss: -239442954.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4401/5000
    500/500 [==============================] - 0s 78us/sample - loss: 74005332.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4402/5000
    500/500 [==============================] - 0s 94us/sample - loss: 83701235.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4403/5000
    500/500 [==============================] - 0s 92us/sample - loss: -22727150.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4404/5000
    500/500 [==============================] - 0s 98us/sample - loss: 115915126.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4405/5000
    500/500 [==============================] - 0s 88us/sample - loss: 52342850.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4406/5000
    500/500 [==============================] - 0s 90us/sample - loss: -80528403.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4407/5000
    500/500 [==============================] - 0s 82us/sample - loss: 155577482.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4408/5000
    500/500 [==============================] - 0s 79us/sample - loss: 29729307.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4409/5000
    500/500 [==============================] - 0s 86us/sample - loss: -21798203.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4410/5000
    500/500 [==============================] - 0s 86us/sample - loss: 160887636.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4411/5000
    500/500 [==============================] - 0s 84us/sample - loss: 142963149.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4412/5000
    500/500 [==============================] - 0s 84us/sample - loss: 139725338.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4413/5000
    500/500 [==============================] - 0s 86us/sample - loss: 100890747.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4414/5000
    500/500 [==============================] - 0s 86us/sample - loss: -24532757.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4415/5000
    500/500 [==============================] - 0s 84us/sample - loss: -16825648.2280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4416/5000
    500/500 [==============================] - 0s 80us/sample - loss: 38631775.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4417/5000
    500/500 [==============================] - 0s 92us/sample - loss: 84507619.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4418/5000
    500/500 [==============================] - 0s 76us/sample - loss: -57156196.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4419/5000
    500/500 [==============================] - 0s 92us/sample - loss: 142826004.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4420/5000
    500/500 [==============================] - 0s 86us/sample - loss: -65178044.8300 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4421/5000
    500/500 [==============================] - 0s 80us/sample - loss: 191300232.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4422/5000
    500/500 [==============================] - 0s 84us/sample - loss: 34355189.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4423/5000
    500/500 [==============================] - 0s 90us/sample - loss: 137774083.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4424/5000
    500/500 [==============================] - 0s 80us/sample - loss: -3521348.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4425/5000
    500/500 [==============================] - 0s 86us/sample - loss: -118888514.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4426/5000
    500/500 [==============================] - 0s 82us/sample - loss: 15884195.4560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4427/5000
    500/500 [==============================] - 0s 84us/sample - loss: 2803444.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4428/5000
    500/500 [==============================] - 0s 84us/sample - loss: -48445524.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4429/5000
    500/500 [==============================] - 0s 82us/sample - loss: -80393874.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4430/5000
    500/500 [==============================] - 0s 88us/sample - loss: -68916554.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4431/5000
    500/500 [==============================] - 0s 84us/sample - loss: -25146977.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4432/5000
    500/500 [==============================] - 0s 82us/sample - loss: -35920801.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4433/5000
    500/500 [==============================] - 0s 86us/sample - loss: 110389663.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4434/5000
    500/500 [==============================] - 0s 94us/sample - loss: 100009945.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4435/5000
    500/500 [==============================] - 0s 82us/sample - loss: -178488827.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4436/5000
    500/500 [==============================] - 0s 80us/sample - loss: 97430035.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4437/5000
    500/500 [==============================] - 0s 92us/sample - loss: -96417820.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4438/5000
    500/500 [==============================] - 0s 82us/sample - loss: -28479264.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4439/5000
    500/500 [==============================] - 0s 88us/sample - loss: -25683640.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4440/5000
    500/500 [==============================] - 0s 86us/sample - loss: -117659071.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4441/5000
    500/500 [==============================] - 0s 84us/sample - loss: 126005018.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4442/5000
    500/500 [==============================] - 0s 94us/sample - loss: 19022470.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4443/5000
    500/500 [==============================] - 0s 92us/sample - loss: -48144921.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4444/5000
    500/500 [==============================] - 0s 94us/sample - loss: -75457938.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4445/5000
    500/500 [==============================] - 0s 80us/sample - loss: -90954229.7640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4446/5000
    500/500 [==============================] - 0s 90us/sample - loss: 16401715.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4447/5000
    500/500 [==============================] - 0s 108us/sample - loss: 27849053.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4448/5000
    500/500 [==============================] - 0s 82us/sample - loss: 96852978.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4449/5000
    500/500 [==============================] - 0s 100us/sample - loss: 40331999.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4450/5000
    500/500 [==============================] - 0s 98us/sample - loss: -245066273.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4451/5000
    500/500 [==============================] - 0s 82us/sample - loss: -97829754.3660 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4452/5000
    500/500 [==============================] - 0s 92us/sample - loss: -157001808.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4453/5000
    500/500 [==============================] - 0s 82us/sample - loss: -31736213.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4454/5000
    500/500 [==============================] - 0s 90us/sample - loss: 91424782.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4455/5000
    500/500 [==============================] - 0s 96us/sample - loss: -134586567.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4456/5000
    500/500 [==============================] - 0s 90us/sample - loss: 15887966.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4457/5000
    500/500 [==============================] - 0s 78us/sample - loss: -90968813.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4458/5000
    500/500 [==============================] - 0s 79us/sample - loss: 24935284.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4459/5000
    500/500 [==============================] - 0s 86us/sample - loss: 34895030.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4460/5000
    500/500 [==============================] - 0s 84us/sample - loss: 86052392.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4461/5000
    500/500 [==============================] - 0s 88us/sample - loss: 205464906.1960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4462/5000
    500/500 [==============================] - 0s 95us/sample - loss: 38881437.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4463/5000
    500/500 [==============================] - 0s 100us/sample - loss: -19985826.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4464/5000
    500/500 [==============================] - 0s 94us/sample - loss: 58999126.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4465/5000
    500/500 [==============================] - 0s 90us/sample - loss: 91458348.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4466/5000
    500/500 [==============================] - 0s 81us/sample - loss: 70023620.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4467/5000
    500/500 [==============================] - 0s 85us/sample - loss: -55280358.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4468/5000
    500/500 [==============================] - 0s 100us/sample - loss: -659525.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4469/5000
    500/500 [==============================] - 0s 100us/sample - loss: -170873100.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4470/5000
    500/500 [==============================] - 0s 88us/sample - loss: 138524972.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4471/5000
    500/500 [==============================] - 0s 83us/sample - loss: -29016480.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4472/5000
    500/500 [==============================] - 0s 90us/sample - loss: -12509415.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4473/5000
    500/500 [==============================] - 0s 92us/sample - loss: -5417775.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4474/5000
    500/500 [==============================] - 0s 78us/sample - loss: -76653295.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4475/5000
    500/500 [==============================] - 0s 62us/sample - loss: -104979280.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4476/5000
    500/500 [==============================] - 0s 87us/sample - loss: -21550399.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4477/5000
    500/500 [==============================] - 0s 61us/sample - loss: 205906936.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4478/5000
    500/500 [==============================] - 0s 76us/sample - loss: 1859522.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4479/5000
    500/500 [==============================] - 0s 70us/sample - loss: 124180122.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4480/5000
    500/500 [==============================] - 0s 117us/sample - loss: 49941638.3980 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4481/5000
    500/500 [==============================] - 0s 54us/sample - loss: 113292027.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4482/5000
    500/500 [==============================] - 0s 62us/sample - loss: 82334946.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4483/5000
    500/500 [==============================] - 0s 117us/sample - loss: -136534668.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4484/5000
    500/500 [==============================] - 0s 67us/sample - loss: 64791518.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4485/5000
    500/500 [==============================] - 0s 56us/sample - loss: -14611174.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4486/5000
    500/500 [==============================] - 0s 50us/sample - loss: -15548155.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4487/5000
    500/500 [==============================] - 0s 107us/sample - loss: -137037249.7280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4488/5000
    500/500 [==============================] - 0s 98us/sample - loss: 13948191.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4489/5000
    500/500 [==============================] - 0s 101us/sample - loss: -12704473.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4490/5000
    500/500 [==============================] - 0s 116us/sample - loss: 52226171.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4491/5000
    500/500 [==============================] - 0s 84us/sample - loss: 88889593.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4492/5000
    500/500 [==============================] - 0s 96us/sample - loss: 48952618.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4493/5000
    500/500 [==============================] - 0s 98us/sample - loss: -78731588.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4494/5000
    500/500 [==============================] - 0s 96us/sample - loss: -54736942.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4495/5000
    500/500 [==============================] - 0s 98us/sample - loss: -21772935.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4496/5000
    500/500 [==============================] - 0s 92us/sample - loss: -26585231.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4497/5000
    500/500 [==============================] - 0s 102us/sample - loss: 92579795.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4498/5000
    500/500 [==============================] - 0s 88us/sample - loss: 57945138.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4499/5000
    500/500 [==============================] - 0s 98us/sample - loss: -84367561.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4500/5000
    500/500 [==============================] - 0s 90us/sample - loss: -183391414.7840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4501/5000
    500/500 [==============================] - 0s 98us/sample - loss: -136957942.6920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4502/5000
    500/500 [==============================] - 0s 94us/sample - loss: -140333269.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4503/5000
    500/500 [==============================] - 0s 98us/sample - loss: 43481671.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4504/5000
    500/500 [==============================] - 0s 108us/sample - loss: -105170708.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4505/5000
    500/500 [==============================] - 0s 101us/sample - loss: 60346835.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4506/5000
    500/500 [==============================] - 0s 95us/sample - loss: -166674300.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4507/5000
    500/500 [==============================] - 0s 95us/sample - loss: 15066147.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4508/5000
    500/500 [==============================] - 0s 92us/sample - loss: -119236345.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4509/5000
    500/500 [==============================] - 0s 88us/sample - loss: 187361020.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4510/5000
    500/500 [==============================] - 0s 95us/sample - loss: -30674416.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4511/5000
    500/500 [==============================] - 0s 84us/sample - loss: 2546904.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4512/5000
    500/500 [==============================] - 0s 86us/sample - loss: -81991079.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4513/5000
    500/500 [==============================] - 0s 90us/sample - loss: -3393221.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4514/5000
    500/500 [==============================] - 0s 86us/sample - loss: 74883684.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4515/5000
    500/500 [==============================] - 0s 90us/sample - loss: 110678277.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4516/5000
    500/500 [==============================] - 0s 86us/sample - loss: 17073973.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4517/5000
    500/500 [==============================] - 0s 90us/sample - loss: -73601102.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4518/5000
    500/500 [==============================] - 0s 84us/sample - loss: 7558638.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4519/5000
    500/500 [==============================] - 0s 80us/sample - loss: 145792053.3080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4520/5000
    500/500 [==============================] - 0s 94us/sample - loss: 38000065.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4521/5000
    500/500 [==============================] - 0s 88us/sample - loss: 55653804.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4522/5000
    500/500 [==============================] - 0s 84us/sample - loss: 51384873.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4523/5000
    500/500 [==============================] - 0s 86us/sample - loss: 15098764.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4524/5000
    500/500 [==============================] - 0s 90us/sample - loss: 44946861.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4525/5000
    500/500 [==============================] - 0s 82us/sample - loss: -136643337.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4526/5000
    500/500 [==============================] - 0s 80us/sample - loss: -76266794.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4527/5000
    500/500 [==============================] - 0s 80us/sample - loss: 118424992.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4528/5000
    500/500 [==============================] - 0s 82us/sample - loss: -85702149.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4529/5000
    500/500 [==============================] - 0s 90us/sample - loss: 38017034.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4530/5000
    500/500 [==============================] - 0s 102us/sample - loss: -7817008.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4531/5000
    500/500 [==============================] - 0s 82us/sample - loss: 9690591.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4532/5000
    500/500 [==============================] - 0s 103us/sample - loss: 75624486.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4533/5000
    500/500 [==============================] - 0s 80us/sample - loss: -53209982.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4534/5000
    500/500 [==============================] - 0s 92us/sample - loss: 39223421.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4535/5000
    500/500 [==============================] - 0s 94us/sample - loss: 53299879.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4536/5000
    500/500 [==============================] - 0s 82us/sample - loss: 99089773.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4537/5000
    500/500 [==============================] - 0s 92us/sample - loss: -41722113.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4538/5000
    500/500 [==============================] - 0s 82us/sample - loss: -86079109.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4539/5000
    500/500 [==============================] - 0s 90us/sample - loss: -62133206.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4540/5000
    500/500 [==============================] - 0s 94us/sample - loss: 30465440.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4541/5000
    500/500 [==============================] - 0s 90us/sample - loss: 167970848.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4542/5000
    500/500 [==============================] - 0s 98us/sample - loss: -73224732.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4543/5000
    500/500 [==============================] - 0s 82us/sample - loss: -190472159.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4544/5000
    500/500 [==============================] - 0s 98us/sample - loss: -121581924.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4545/5000
    500/500 [==============================] - 0s 90us/sample - loss: 27313382.5640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4546/5000
    500/500 [==============================] - 0s 112us/sample - loss: -117250876.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4547/5000
    500/500 [==============================] - 0s 110us/sample - loss: 159393800.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4548/5000
    500/500 [==============================] - 0s 110us/sample - loss: -44330769.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4549/5000
    500/500 [==============================] - 0s 100us/sample - loss: 61704594.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4550/5000
    500/500 [==============================] - 0s 98us/sample - loss: -37360605.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4551/5000
    500/500 [==============================] - 0s 102us/sample - loss: 2166414.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4552/5000
    500/500 [==============================] - 0s 96us/sample - loss: -10980234.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4553/5000
    500/500 [==============================] - 0s 102us/sample - loss: 124224325.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4554/5000
    500/500 [==============================] - 0s 102us/sample - loss: -19335251.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4555/5000
    500/500 [==============================] - 0s 102us/sample - loss: -70444492.3860 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4556/5000
    500/500 [==============================] - 0s 98us/sample - loss: -180600811.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4557/5000
    500/500 [==============================] - 0s 90us/sample - loss: -50465007.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4558/5000
    500/500 [==============================] - 0s 80us/sample - loss: 22377915.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4559/5000
    500/500 [==============================] - 0s 94us/sample - loss: -94614712.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4560/5000
    500/500 [==============================] - 0s 92us/sample - loss: 81240382.4800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4561/5000
    500/500 [==============================] - 0s 82us/sample - loss: 6945953.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4562/5000
    500/500 [==============================] - 0s 80us/sample - loss: 17060669.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4563/5000
    500/500 [==============================] - 0s 92us/sample - loss: -32292903.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4564/5000
    500/500 [==============================] - 0s 80us/sample - loss: -53614887.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4565/5000
    500/500 [==============================] - 0s 92us/sample - loss: 59821360.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4566/5000
    500/500 [==============================] - 0s 88us/sample - loss: -79417764.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4567/5000
    500/500 [==============================] - 0s 96us/sample - loss: 252597139.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4568/5000
    500/500 [==============================] - 0s 78us/sample - loss: -19739467.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4569/5000
    500/500 [==============================] - 0s 79us/sample - loss: 42748148.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4570/5000
    500/500 [==============================] - 0s 90us/sample - loss: -52280679.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4571/5000
    500/500 [==============================] - 0s 90us/sample - loss: 135927588.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4572/5000
    500/500 [==============================] - 0s 88us/sample - loss: -8230802.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4573/5000
    500/500 [==============================] - 0s 98us/sample - loss: 223649450.6740 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4574/5000
    500/500 [==============================] - 0s 98us/sample - loss: 35967717.3760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4575/5000
    500/500 [==============================] - 0s 98us/sample - loss: -47914224.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4576/5000
    500/500 [==============================] - 0s 98us/sample - loss: -89780467.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4577/5000
    500/500 [==============================] - 0s 122us/sample - loss: 70253311.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4578/5000
    500/500 [==============================] - 0s 104us/sample - loss: 40912380.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4579/5000
    500/500 [==============================] - 0s 96us/sample - loss: -131958241.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4580/5000
    500/500 [==============================] - 0s 92us/sample - loss: 122755321.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4581/5000
    500/500 [==============================] - 0s 97us/sample - loss: -147099068.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4582/5000
    500/500 [==============================] - 0s 98us/sample - loss: -39662287.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4583/5000
    500/500 [==============================] - 0s 98us/sample - loss: -82172859.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4584/5000
    500/500 [==============================] - 0s 203us/sample - loss: 48653164.8280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4585/5000
    500/500 [==============================] - 0s 108us/sample - loss: -13923580.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4586/5000
    500/500 [==============================] - 0s 98us/sample - loss: -47373019.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4587/5000
    500/500 [==============================] - 0s 90us/sample - loss: -123832884.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4588/5000
    500/500 [==============================] - 0s 98us/sample - loss: -73856937.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4589/5000
    500/500 [==============================] - 0s 106us/sample - loss: 78287169.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4590/5000
    500/500 [==============================] - 0s 106us/sample - loss: -72576736.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4591/5000
    500/500 [==============================] - 0s 75us/sample - loss: 32772300.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4592/5000
    500/500 [==============================] - 0s 100us/sample - loss: 77924064.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4593/5000
    500/500 [==============================] - 0s 94us/sample - loss: -61378772.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4594/5000
    500/500 [==============================] - 0s 104us/sample - loss: -78388842.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4595/5000
    500/500 [==============================] - 0s 114us/sample - loss: 195511728.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4596/5000
    500/500 [==============================] - 0s 131us/sample - loss: -10407063.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4597/5000
    500/500 [==============================] - 0s 120us/sample - loss: -138322486.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4598/5000
    500/500 [==============================] - 0s 114us/sample - loss: -76546796.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4599/5000
    500/500 [==============================] - 0s 102us/sample - loss: 79341472.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4600/5000
    500/500 [==============================] - 0s 93us/sample - loss: -50795032.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4601/5000
    500/500 [==============================] - 0s 92us/sample - loss: -89590111.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4602/5000
    500/500 [==============================] - 0s 88us/sample - loss: 28142886.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4603/5000
    500/500 [==============================] - 0s 108us/sample - loss: 149294526.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4604/5000
    500/500 [==============================] - 0s 106us/sample - loss: -64000623.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4605/5000
    500/500 [==============================] - 0s 98us/sample - loss: 9198304.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4606/5000
    500/500 [==============================] - 0s 100us/sample - loss: -64764586.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4607/5000
    500/500 [==============================] - 0s 128us/sample - loss: -82121032.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4608/5000
    500/500 [==============================] - 0s 130us/sample - loss: -77845204.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4609/5000
    500/500 [==============================] - 0s 135us/sample - loss: -50263568.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4610/5000
    500/500 [==============================] - 0s 124us/sample - loss: -31330944.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4611/5000
    500/500 [==============================] - 0s 110us/sample - loss: -17518926.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4612/5000
    500/500 [==============================] - 0s 112us/sample - loss: -126838323.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4613/5000
    500/500 [==============================] - 0s 102us/sample - loss: -106383627.3920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4614/5000
    500/500 [==============================] - 0s 98us/sample - loss: 105813007.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4615/5000
    500/500 [==============================] - 0s 96us/sample - loss: -70595534.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4616/5000
    500/500 [==============================] - 0s 100us/sample - loss: 26209858.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4617/5000
    500/500 [==============================] - 0s 100us/sample - loss: 46253522.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4618/5000
    500/500 [==============================] - 0s 108us/sample - loss: -47799665.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4619/5000
    500/500 [==============================] - 0s 132us/sample - loss: 19236649.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4620/5000
    500/500 [==============================] - 0s 124us/sample - loss: -110096621.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4621/5000
    500/500 [==============================] - 0s 120us/sample - loss: 61150658.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4622/5000
    500/500 [==============================] - 0s 97us/sample - loss: 142612784.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4623/5000
    500/500 [==============================] - 0s 112us/sample - loss: 41030966.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4624/5000
    500/500 [==============================] - 0s 94us/sample - loss: 14286758.4340 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4625/5000
    500/500 [==============================] - 0s 84us/sample - loss: 34491649.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4626/5000
    500/500 [==============================] - 0s 94us/sample - loss: -109110272.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4627/5000
    500/500 [==============================] - 0s 78us/sample - loss: 35559251.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4628/5000
    500/500 [==============================] - 0s 79us/sample - loss: 213860389.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4629/5000
    500/500 [==============================] - 0s 80us/sample - loss: 61028645.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4630/5000
    500/500 [==============================] - 0s 84us/sample - loss: -9889614.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4631/5000
    500/500 [==============================] - 0s 86us/sample - loss: -99265413.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4632/5000
    500/500 [==============================] - 0s 82us/sample - loss: 75776946.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4633/5000
    500/500 [==============================] - 0s 86us/sample - loss: -119873572.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4634/5000
    500/500 [==============================] - 0s 84us/sample - loss: -170728479.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4635/5000
    500/500 [==============================] - 0s 82us/sample - loss: 22938324.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4636/5000
    500/500 [==============================] - 0s 90us/sample - loss: -28658181.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4637/5000
    500/500 [==============================] - 0s 98us/sample - loss: 196504778.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4638/5000
    500/500 [==============================] - 0s 86us/sample - loss: -125206168.4485 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4639/5000
    500/500 [==============================] - 0s 82us/sample - loss: 49416525.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4640/5000
    500/500 [==============================] - 0s 88us/sample - loss: -80560611.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4641/5000
    500/500 [==============================] - 0s 102us/sample - loss: 115960320.5120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4642/5000
    500/500 [==============================] - 0s 89us/sample - loss: -23618181.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4643/5000
    500/500 [==============================] - 0s 84us/sample - loss: 150876647.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4644/5000
    500/500 [==============================] - 0s 90us/sample - loss: 147883124.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4645/5000
    500/500 [==============================] - 0s 102us/sample - loss: 47811072.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4646/5000
    500/500 [==============================] - 0s 98us/sample - loss: -91199145.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4647/5000
    500/500 [==============================] - 0s 92us/sample - loss: -187115863.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4648/5000
    500/500 [==============================] - 0s 84us/sample - loss: 118560506.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4649/5000
    500/500 [==============================] - 0s 94us/sample - loss: -63536053.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4650/5000
    500/500 [==============================] - 0s 90us/sample - loss: 27364424.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4651/5000
    500/500 [==============================] - 0s 94us/sample - loss: 39806304.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4652/5000
    500/500 [==============================] - 0s 98us/sample - loss: 105300740.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4653/5000
    500/500 [==============================] - 0s 92us/sample - loss: 11561814.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4654/5000
    500/500 [==============================] - 0s 84us/sample - loss: 66036300.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4655/5000
    500/500 [==============================] - 0s 88us/sample - loss: 64610003.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4656/5000
    500/500 [==============================] - 0s 94us/sample - loss: 230527340.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4657/5000
    500/500 [==============================] - 0s 92us/sample - loss: 72529968.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4658/5000
    500/500 [==============================] - 0s 92us/sample - loss: -24789670.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4659/5000
    500/500 [==============================] - 0s 98us/sample - loss: 45950457.7880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4660/5000
    500/500 [==============================] - 0s 94us/sample - loss: 93323659.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4661/5000
    500/500 [==============================] - 0s 98us/sample - loss: -12548204.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4662/5000
    500/500 [==============================] - 0s 92us/sample - loss: -83245544.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4663/5000
    500/500 [==============================] - 0s 84us/sample - loss: 82360083.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4664/5000
    500/500 [==============================] - 0s 88us/sample - loss: 101742880.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4665/5000
    500/500 [==============================] - 0s 90us/sample - loss: 62926061.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4666/5000
    500/500 [==============================] - 0s 80us/sample - loss: -19802580.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4667/5000
    500/500 [==============================] - 0s 96us/sample - loss: 186968618.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4668/5000
    500/500 [==============================] - 0s 92us/sample - loss: 29933936.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4669/5000
    500/500 [==============================] - 0s 96us/sample - loss: -252440313.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4670/5000
    500/500 [==============================] - 0s 94us/sample - loss: 56931822.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4671/5000
    500/500 [==============================] - 0s 104us/sample - loss: 141335708.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4672/5000
    500/500 [==============================] - 0s 102us/sample - loss: 27071286.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4673/5000
    500/500 [==============================] - 0s 94us/sample - loss: 82326683.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4674/5000
    500/500 [==============================] - 0s 100us/sample - loss: 75642297.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4675/5000
    500/500 [==============================] - 0s 96us/sample - loss: -69202600.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4676/5000
    500/500 [==============================] - 0s 94us/sample - loss: -39258393.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4677/5000
    500/500 [==============================] - 0s 100us/sample - loss: 46082554.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4678/5000
    500/500 [==============================] - 0s 100us/sample - loss: 145074097.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4679/5000
    500/500 [==============================] - 0s 86us/sample - loss: 24171690.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4680/5000
    500/500 [==============================] - 0s 98us/sample - loss: 14260887.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4681/5000
    500/500 [==============================] - 0s 102us/sample - loss: 14101067.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4682/5000
    500/500 [==============================] - 0s 105us/sample - loss: 36492186.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4683/5000
    500/500 [==============================] - 0s 98us/sample - loss: -59434705.8560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4684/5000
    500/500 [==============================] - 0s 94us/sample - loss: -25411637.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4685/5000
    500/500 [==============================] - 0s 82us/sample - loss: -874276.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4686/5000
    500/500 [==============================] - 0s 78us/sample - loss: -30205372.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4687/5000
    500/500 [==============================] - 0s 78us/sample - loss: -103916338.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4688/5000
    500/500 [==============================] - 0s 80us/sample - loss: 41602817.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4689/5000
    500/500 [==============================] - 0s 78us/sample - loss: 103896844.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4690/5000
    500/500 [==============================] - 0s 94us/sample - loss: -21143775.5520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4691/5000
    500/500 [==============================] - 0s 100us/sample - loss: -85150384.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4692/5000
    500/500 [==============================] - 0s 86us/sample - loss: -135799350.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4693/5000
    500/500 [==============================] - 0s 76us/sample - loss: 29497119.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4694/5000
    500/500 [==============================] - 0s 80us/sample - loss: 99329086.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4695/5000
    500/500 [==============================] - 0s 76us/sample - loss: 145389302.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4696/5000
    500/500 [==============================] - 0s 88us/sample - loss: 68106562.0480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4697/5000
    500/500 [==============================] - 0s 92us/sample - loss: -55725332.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4698/5000
    500/500 [==============================] - 0s 82us/sample - loss: -89379056.1920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4699/5000
    500/500 [==============================] - 0s 100us/sample - loss: 54268672.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4700/5000
    500/500 [==============================] - 0s 80us/sample - loss: 101896045.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4701/5000
    500/500 [==============================] - 0s 94us/sample - loss: 161105869.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4702/5000
    500/500 [==============================] - 0s 94us/sample - loss: 25844987.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4703/5000
    500/500 [==============================] - 0s 96us/sample - loss: -41941104.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4704/5000
    500/500 [==============================] - 0s 80us/sample - loss: 165813329.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4705/5000
    500/500 [==============================] - 0s 92us/sample - loss: 130493600.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4706/5000
    500/500 [==============================] - 0s 88us/sample - loss: -70155557.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4707/5000
    500/500 [==============================] - 0s 92us/sample - loss: 67903554.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4708/5000
    500/500 [==============================] - 0s 94us/sample - loss: -41770302.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4709/5000
    500/500 [==============================] - 0s 80us/sample - loss: 28388215.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4710/5000
    500/500 [==============================] - 0s 92us/sample - loss: -72512025.2800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4711/5000
    500/500 [==============================] - 0s 98us/sample - loss: 115966599.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4712/5000
    500/500 [==============================] - 0s 110us/sample - loss: 154372209.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4713/5000
    500/500 [==============================] - 0s 80us/sample - loss: 16346960.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4714/5000
    500/500 [==============================] - 0s 88us/sample - loss: -29760718.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4715/5000
    500/500 [==============================] - 0s 84us/sample - loss: -66812647.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4716/5000
    500/500 [==============================] - 0s 78us/sample - loss: 122454536.8470 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4717/5000
    500/500 [==============================] - 0s 92us/sample - loss: -51909930.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4718/5000
    500/500 [==============================] - 0s 84us/sample - loss: 53444526.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4719/5000
    500/500 [==============================] - 0s 82us/sample - loss: -42592320.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4720/5000
    500/500 [==============================] - 0s 96us/sample - loss: 91401722.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4721/5000
    500/500 [==============================] - 0s 92us/sample - loss: 153646551.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4722/5000
    500/500 [==============================] - 0s 80us/sample - loss: -86466572.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4723/5000
    500/500 [==============================] - 0s 82us/sample - loss: -53896716.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4724/5000
    500/500 [==============================] - 0s 98us/sample - loss: -76843525.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4725/5000
    500/500 [==============================] - 0s 102us/sample - loss: 114589624.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4726/5000
    500/500 [==============================] - 0s 90us/sample - loss: 867964.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4727/5000
    500/500 [==============================] - 0s 96us/sample - loss: 47734885.6880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4728/5000
    500/500 [==============================] - 0s 84us/sample - loss: 20422319.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4729/5000
    500/500 [==============================] - ETA: 0s - loss: -344038784.0000 - accuracy: 0.0000e+0 - 0s 80us/sample - loss: 4776366.3360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4730/5000
    500/500 [==============================] - 0s 94us/sample - loss: -178077346.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4731/5000
    500/500 [==============================] - 0s 100us/sample - loss: -55694864.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4732/5000
    500/500 [==============================] - 0s 96us/sample - loss: -34227078.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4733/5000
    500/500 [==============================] - 0s 94us/sample - loss: 13575419.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4734/5000
    500/500 [==============================] - 0s 94us/sample - loss: -101008121.0880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4735/5000
    500/500 [==============================] - 0s 94us/sample - loss: -3759956.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4736/5000
    500/500 [==============================] - 0s 94us/sample - loss: -38898679.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4737/5000
    500/500 [==============================] - 0s 98us/sample - loss: 96137475.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4738/5000
    500/500 [==============================] - 0s 96us/sample - loss: 172398205.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4739/5000
    500/500 [==============================] - 0s 81us/sample - loss: 9496513.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4740/5000
    500/500 [==============================] - 0s 88us/sample - loss: 60058534.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4741/5000
    500/500 [==============================] - 0s 78us/sample - loss: 40159309.8880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4742/5000
    500/500 [==============================] - 0s 78us/sample - loss: -37583955.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4743/5000
    500/500 [==============================] - 0s 90us/sample - loss: 83022339.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4744/5000
    500/500 [==============================] - 0s 90us/sample - loss: 50433337.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4745/5000
    500/500 [==============================] - 0s 78us/sample - loss: 22611461.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4746/5000
    500/500 [==============================] - 0s 92us/sample - loss: 106536666.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4747/5000
    500/500 [==============================] - 0s 78us/sample - loss: 79686713.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4748/5000
    500/500 [==============================] - 0s 82us/sample - loss: -88505896.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4749/5000
    500/500 [==============================] - 0s 82us/sample - loss: 134555867.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4750/5000
    500/500 [==============================] - 0s 88us/sample - loss: -31916814.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4751/5000
    500/500 [==============================] - 0s 96us/sample - loss: -144946234.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4752/5000
    500/500 [==============================] - 0s 82us/sample - loss: 95048815.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4753/5000
    500/500 [==============================] - 0s 84us/sample - loss: -66804316.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4754/5000
    500/500 [==============================] - 0s 78us/sample - loss: 35376051.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4755/5000
    500/500 [==============================] - 0s 76us/sample - loss: 202478338.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4756/5000
    500/500 [==============================] - 0s 82us/sample - loss: 180980437.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4757/5000
    500/500 [==============================] - 0s 86us/sample - loss: 189022577.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4758/5000
    500/500 [==============================] - 0s 86us/sample - loss: 4045146.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4759/5000
    500/500 [==============================] - 0s 86us/sample - loss: 70112519.8720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4760/5000
    500/500 [==============================] - 0s 86us/sample - loss: -59827712.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4761/5000
    500/500 [==============================] - 0s 82us/sample - loss: -103519789.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4762/5000
    500/500 [==============================] - 0s 88us/sample - loss: -106925730.5640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4763/5000
    500/500 [==============================] - 0s 80us/sample - loss: -92772312.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4764/5000
    500/500 [==============================] - 0s 74us/sample - loss: -578588.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4765/5000
    500/500 [==============================] - 0s 90us/sample - loss: -146774577.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4766/5000
    500/500 [==============================] - 0s 82us/sample - loss: -49939463.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4767/5000
    500/500 [==============================] - 0s 100us/sample - loss: -54418354.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4768/5000
    500/500 [==============================] - 0s 80us/sample - loss: 7682533.1520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4769/5000
    500/500 [==============================] - 0s 82us/sample - loss: -60574390.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4770/5000
    500/500 [==============================] - 0s 80us/sample - loss: -32632404.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4771/5000
    500/500 [==============================] - 0s 84us/sample - loss: -59075118.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4772/5000
    500/500 [==============================] - 0s 76us/sample - loss: 109633913.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4773/5000
    500/500 [==============================] - 0s 90us/sample - loss: -27723373.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4774/5000
    500/500 [==============================] - 0s 82us/sample - loss: -192994952.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4775/5000
    500/500 [==============================] - 0s 100us/sample - loss: -103908374.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4776/5000
    500/500 [==============================] - 0s 92us/sample - loss: -44354238.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4777/5000
    500/500 [==============================] - 0s 80us/sample - loss: 8916852.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4778/5000
    500/500 [==============================] - 0s 84us/sample - loss: -155465636.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4779/5000
    500/500 [==============================] - 0s 78us/sample - loss: -56308425.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4780/5000
    500/500 [==============================] - 0s 94us/sample - loss: -56356120.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4781/5000
    500/500 [==============================] - 0s 88us/sample - loss: -78634955.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4782/5000
    500/500 [==============================] - 0s 88us/sample - loss: 81349359.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4783/5000
    500/500 [==============================] - 0s 84us/sample - loss: -44712676.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4784/5000
    500/500 [==============================] - 0s 100us/sample - loss: -102265847.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4785/5000
    500/500 [==============================] - 0s 102us/sample - loss: -87198446.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4786/5000
    500/500 [==============================] - 0s 83us/sample - loss: 52046243.3280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4787/5000
    500/500 [==============================] - 0s 90us/sample - loss: -28989134.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4788/5000
    500/500 [==============================] - 0s 94us/sample - loss: -24280377.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4789/5000
    500/500 [==============================] - 0s 82us/sample - loss: -98532849.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4790/5000
    500/500 [==============================] - 0s 72us/sample - loss: -117020737.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4791/5000
    500/500 [==============================] - 0s 92us/sample - loss: -104255169.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4792/5000
    500/500 [==============================] - 0s 82us/sample - loss: -91877056.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4793/5000
    500/500 [==============================] - 0s 94us/sample - loss: 123804814.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4794/5000
    500/500 [==============================] - 0s 92us/sample - loss: 65737969.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4795/5000
    500/500 [==============================] - 0s 90us/sample - loss: 83997439.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4796/5000
    500/500 [==============================] - 0s 78us/sample - loss: 127041183.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4797/5000
    500/500 [==============================] - 0s 88us/sample - loss: 105005689.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4798/5000
    500/500 [==============================] - 0s 88us/sample - loss: -173671032.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4799/5000
    500/500 [==============================] - 0s 80us/sample - loss: -147415642.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4800/5000
    500/500 [==============================] - 0s 100us/sample - loss: -61699687.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4801/5000
    500/500 [==============================] - 0s 90us/sample - loss: 19945153.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4802/5000
    500/500 [==============================] - 0s 96us/sample - loss: -4275223.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4803/5000
    500/500 [==============================] - 0s 90us/sample - loss: -94562919.9360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4804/5000
    500/500 [==============================] - 0s 82us/sample - loss: -46625747.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4805/5000
    500/500 [==============================] - 0s 76us/sample - loss: 72368275.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4806/5000
    500/500 [==============================] - 0s 78us/sample - loss: -7845374.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4807/5000
    500/500 [==============================] - 0s 82us/sample - loss: 76390535.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4808/5000
    500/500 [==============================] - 0s 86us/sample - loss: 81736661.2480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4809/5000
    500/500 [==============================] - 0s 86us/sample - loss: 62311936.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4810/5000
    500/500 [==============================] - 0s 86us/sample - loss: 104309822.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4811/5000
    500/500 [==============================] - 0s 78us/sample - loss: -44210610.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4812/5000
    500/500 [==============================] - 0s 80us/sample - loss: -21052746.8160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4813/5000
    500/500 [==============================] - 0s 94us/sample - loss: 70767648.8320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4814/5000
    500/500 [==============================] - 0s 82us/sample - loss: 97254699.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4815/5000
    500/500 [==============================] - 0s 78us/sample - loss: -88519125.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4816/5000
    500/500 [==============================] - 0s 83us/sample - loss: -43461930.4920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4817/5000
    500/500 [==============================] - 0s 100us/sample - loss: 86148258.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4818/5000
    500/500 [==============================] - 0s 78us/sample - loss: -14456482.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4819/5000
    500/500 [==============================] - 0s 92us/sample - loss: 15508633.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4820/5000
    500/500 [==============================] - 0s 82us/sample - loss: -86278885.0560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4821/5000
    500/500 [==============================] - 0s 76us/sample - loss: 19502350.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4822/5000
    500/500 [==============================] - 0s 78us/sample - loss: -63457329.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4823/5000
    500/500 [==============================] - 0s 88us/sample - loss: -123401791.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4824/5000
    500/500 [==============================] - 0s 96us/sample - loss: 33948116.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4825/5000
    500/500 [==============================] - 0s 98us/sample - loss: -21972309.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4826/5000
    500/500 [==============================] - 0s 100us/sample - loss: -40584287.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4827/5000
    500/500 [==============================] - 0s 86us/sample - loss: -73300174.6565 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4828/5000
    500/500 [==============================] - 0s 94us/sample - loss: -54240548.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4829/5000
    500/500 [==============================] - 0s 92us/sample - loss: -33353138.1760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4830/5000
    500/500 [==============================] - 0s 98us/sample - loss: 7327637.4400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4831/5000
    500/500 [==============================] - 0s 109us/sample - loss: -89393684.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4832/5000
    500/500 [==============================] - 0s 97us/sample - loss: 82266096.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4833/5000
    500/500 [==============================] - 0s 94us/sample - loss: 49776464.0000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4834/5000
    500/500 [==============================] - 0s 84us/sample - loss: -21504648.2720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4835/5000
    500/500 [==============================] - 0s 82us/sample - loss: -15780516.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4836/5000
    500/500 [==============================] - 0s 88us/sample - loss: 89913871.4640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4837/5000
    500/500 [==============================] - 0s 82us/sample - loss: -235636.0240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4838/5000
    500/500 [==============================] - 0s 86us/sample - loss: 81533058.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4839/5000
    500/500 [==============================] - 0s 98us/sample - loss: -148913606.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4840/5000
    500/500 [==============================] - 0s 92us/sample - loss: 32609481.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4841/5000
    500/500 [==============================] - 0s 90us/sample - loss: -231154896.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4842/5000
    500/500 [==============================] - 0s 78us/sample - loss: 129616732.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4843/5000
    500/500 [==============================] - 0s 80us/sample - loss: 26006359.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4844/5000
    500/500 [==============================] - 0s 78us/sample - loss: -93886453.1200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4845/5000
    500/500 [==============================] - 0s 84us/sample - loss: -90852283.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4846/5000
    500/500 [==============================] - 0s 92us/sample - loss: -87766150.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4847/5000
    500/500 [==============================] - 0s 80us/sample - loss: -16889812.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4848/5000
    500/500 [==============================] - 0s 94us/sample - loss: -78931763.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4849/5000
    500/500 [==============================] - 0s 94us/sample - loss: -84440426.1600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4850/5000
    500/500 [==============================] - 0s 90us/sample - loss: 19893412.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4851/5000
    500/500 [==============================] - 0s 80us/sample - loss: 143100148.9920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4852/5000
    500/500 [==============================] - 0s 94us/sample - loss: 124644203.6480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4853/5000
    500/500 [==============================] - 0s 100us/sample - loss: -12550975.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4854/5000
    500/500 [==============================] - 0s 84us/sample - loss: 130452538.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4855/5000
    500/500 [==============================] - 0s 92us/sample - loss: 52960902.5920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4856/5000
    500/500 [==============================] - 0s 82us/sample - loss: -142898331.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4857/5000
    500/500 [==============================] - 0s 83us/sample - loss: -107124767.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4858/5000
    500/500 [==============================] - 0s 80us/sample - loss: 110417732.8960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4859/5000
    500/500 [==============================] - 0s 88us/sample - loss: -157406948.6080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4860/5000
    500/500 [==============================] - 0s 100us/sample - loss: 24693969.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4861/5000
    500/500 [==============================] - 0s 98us/sample - loss: -20705695.9980 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4862/5000
    500/500 [==============================] - 0s 86us/sample - loss: 213316133.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4863/5000
    500/500 [==============================] - 0s 78us/sample - loss: -819741.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4864/5000
    500/500 [==============================] - 0s 76us/sample - loss: -67240020.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4865/5000
    500/500 [==============================] - 0s 78us/sample - loss: -3681759.6160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4866/5000
    500/500 [==============================] - 0s 88us/sample - loss: -179238295.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4867/5000
    500/500 [==============================] - 0s 90us/sample - loss: -96573716.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4868/5000
    500/500 [==============================] - 0s 90us/sample - loss: 62884830.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4869/5000
    500/500 [==============================] - 0s 105us/sample - loss: -163740722.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4870/5000
    500/500 [==============================] - 0s 96us/sample - loss: 103057589.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4871/5000
    500/500 [==============================] - 0s 96us/sample - loss: -106924482.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4872/5000
    500/500 [==============================] - 0s 92us/sample - loss: 81549482.8800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4873/5000
    500/500 [==============================] - 0s 88us/sample - loss: 1553912.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4874/5000
    500/500 [==============================] - 0s 78us/sample - loss: 24009482.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4875/5000
    500/500 [==============================] - 0s 94us/sample - loss: -3106687.2640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4876/5000
    500/500 [==============================] - 0s 82us/sample - loss: -148245516.0320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4877/5000
    500/500 [==============================] - 0s 80us/sample - loss: 19865224.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4878/5000
    500/500 [==============================] - 0s 92us/sample - loss: -76041558.9840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4879/5000
    500/500 [==============================] - 0s 96us/sample - loss: -209598405.8240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4880/5000
    500/500 [==============================] - 0s 74us/sample - loss: -47632509.9520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4881/5000
    500/500 [==============================] - 0s 90us/sample - loss: 204889922.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4882/5000
    500/500 [==============================] - 0s 84us/sample - loss: 34906363.7120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4883/5000
    500/500 [==============================] - 0s 82us/sample - loss: -3225027.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4884/5000
    500/500 [==============================] - 0s 90us/sample - loss: 8145599.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4885/5000
    500/500 [==============================] - 0s 86us/sample - loss: -12719630.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4886/5000
    500/500 [==============================] - 0s 88us/sample - loss: 3116309.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4887/5000
    500/500 [==============================] - 0s 86us/sample - loss: -43238556.0960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4888/5000
    500/500 [==============================] - 0s 90us/sample - loss: 29588704.6250 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4889/5000
    500/500 [==============================] - 0s 92us/sample - loss: 147162524.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4890/5000
    500/500 [==============================] - 0s 86us/sample - loss: -33928119.1020 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4891/5000
    500/500 [==============================] - 0s 92us/sample - loss: -111278145.5360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4892/5000
    500/500 [==============================] - 0s 94us/sample - loss: -96780407.1040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4893/5000
    500/500 [==============================] - 0s 88us/sample - loss: 47587144.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4894/5000
    500/500 [==============================] - 0s 90us/sample - loss: -150635390.2080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4895/5000
    500/500 [==============================] - 0s 76us/sample - loss: 17463315.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4896/5000
    500/500 [==============================] - 0s 94us/sample - loss: -39563946.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4897/5000
    500/500 [==============================] - 0s 96us/sample - loss: -21251933.0400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4898/5000
    500/500 [==============================] - 0s 100us/sample - loss: -90432606.5280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4899/5000
    500/500 [==============================] - 0s 88us/sample - loss: 40410883.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4900/5000
    500/500 [==============================] - 0s 78us/sample - loss: 70428281.4080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4901/5000
    500/500 [==============================] - 0s 80us/sample - loss: 48010414.9760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4902/5000
    500/500 [==============================] - 0s 78us/sample - loss: 234141764.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4903/5000
    500/500 [==============================] - 0s 80us/sample - loss: 38428025.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4904/5000
    500/500 [==============================] - 0s 82us/sample - loss: 138090424.9280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4905/5000
    500/500 [==============================] - 0s 96us/sample - loss: 128815214.9600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4906/5000
    500/500 [==============================] - 0s 82us/sample - loss: 146214717.6320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4907/5000
    500/500 [==============================] - 0s 78us/sample - loss: -105504386.4960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4908/5000
    500/500 [==============================] - 0s 94us/sample - loss: 110300748.6720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4909/5000
    500/500 [==============================] - 0s 96us/sample - loss: -44844149.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4910/5000
    500/500 [==============================] - 0s 96us/sample - loss: 32822045.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4911/5000
    500/500 [==============================] - 0s 88us/sample - loss: 127461273.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4912/5000
    500/500 [==============================] - 0s 82us/sample - loss: 69412761.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4913/5000
    500/500 [==============================] - 0s 86us/sample - loss: -16689144.7680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4914/5000
    500/500 [==============================] - 0s 136us/sample - loss: 25835641.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4915/5000
    500/500 [==============================] - 0s 96us/sample - loss: 6908699.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4916/5000
    500/500 [==============================] - 0s 100us/sample - loss: -51483087.4240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4917/5000
    500/500 [==============================] - 0s 94us/sample - loss: -5235051.7760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4918/5000
    500/500 [==============================] - 0s 108us/sample - loss: -231360378.6240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4919/5000
    500/500 [==============================] - 0s 104us/sample - loss: 43910475.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4920/5000
    500/500 [==============================] - 0s 100us/sample - loss: -86103472.2560 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4921/5000
    500/500 [==============================] - 0s 98us/sample - loss: -132222738.2400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4922/5000
    500/500 [==============================] - 0s 90us/sample - loss: -140763029.1840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4923/5000
    500/500 [==============================] - 0s 94us/sample - loss: -167665952.4480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4924/5000
    500/500 [==============================] - 0s 116us/sample - loss: 108683176.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4925/5000
    500/500 [==============================] - 0s 108us/sample - loss: -15918662.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4926/5000
    500/500 [==============================] - 0s 115us/sample - loss: -11193006.9120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4927/5000
    500/500 [==============================] - 0s 110us/sample - loss: -151317459.0720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4928/5000
    500/500 [==============================] - 0s 106us/sample - loss: -55960581.7600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4929/5000
    500/500 [==============================] - 0s 98us/sample - loss: 82477866.1120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4930/5000
    500/500 [==============================] - 0s 98us/sample - loss: 154174903.9680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4931/5000
    500/500 [==============================] - 0s 96us/sample - loss: -58997194.9440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4932/5000
    500/500 [==============================] - 0s 98us/sample - loss: -128321968.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4933/5000
    500/500 [==============================] - 0s 94us/sample - loss: -85041087.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4934/5000
    500/500 [==============================] - 0s 98us/sample - loss: 6350705.3120 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4935/5000
    500/500 [==============================] - 0s 98us/sample - loss: 107166501.5040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4936/5000
    500/500 [==============================] - 0s 98us/sample - loss: 83511213.3440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4937/5000
    500/500 [==============================] - 0s 104us/sample - loss: 42144142.4000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4938/5000
    500/500 [==============================] - 0s 106us/sample - loss: 23337041.6640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4939/5000
    500/500 [==============================] - 0s 100us/sample - loss: -52872297.4720 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4940/5000
    500/500 [==============================] - 0s 100us/sample - loss: 15168585.9200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4941/5000
    500/500 [==============================] - 0s 89us/sample - loss: 128556304.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4942/5000
    500/500 [==============================] - 0s 86us/sample - loss: -93587863.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4943/5000
    500/500 [==============================] - 0s 92us/sample - loss: -84205247.2960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4944/5000
    500/500 [==============================] - 0s 106us/sample - loss: -159991739.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4945/5000
    500/500 [==============================] - 0s 117us/sample - loss: 51335126.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4946/5000
    500/500 [==============================] - 0s 120us/sample - loss: 2570037.5760 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4947/5000
    500/500 [==============================] - 0s 109us/sample - loss: 190742756.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4948/5000
    500/500 [==============================] - ETA: 0s - loss: 803749632.0000 - accuracy: 0.0000e+ - 0s 114us/sample - loss: -110437606.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4949/5000
    500/500 [==============================] - 0s 107us/sample - loss: -168819112.4880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4950/5000
    500/500 [==============================] - 0s 110us/sample - loss: -20472654.1440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4951/5000
    500/500 [==============================] - 0s 115us/sample - loss: -127854691.8400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4952/5000
    500/500 [==============================] - 0s 124us/sample - loss: 140859974.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4953/5000
    500/500 [==============================] - 0s 117us/sample - loss: -102805562.7520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4954/5000
    500/500 [==============================] - 0s 112us/sample - loss: -1372333.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4955/5000
    500/500 [==============================] - 0s 100us/sample - loss: 101010270.8480 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4956/5000
    500/500 [==============================] - 0s 100us/sample - loss: -66679698.4320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4957/5000
    500/500 [==============================] - 0s 116us/sample - loss: -4338961.5680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4958/5000
    500/500 [==============================] - 0s 92us/sample - loss: 2883519.3600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4959/5000
    500/500 [==============================] - 0s 92us/sample - loss: 52733814.7200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4960/5000
    500/500 [==============================] - 0s 78us/sample - loss: 50888428.2880 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4961/5000
    500/500 [==============================] - 0s 90us/sample - loss: -49688436.1270 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4962/5000
    500/500 [==============================] - 0s 88us/sample - loss: -47811314.5600 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4963/5000
    500/500 [==============================] - 0s 80us/sample - loss: 5416130.3680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4964/5000
    500/500 [==============================] - 0s 94us/sample - loss: -103487580.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4965/5000
    500/500 [==============================] - 0s 90us/sample - loss: 37856692.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4966/5000
    500/500 [==============================] - 0s 82us/sample - loss: 47401431.7440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4967/5000
    500/500 [==============================] - 0s 78us/sample - loss: -54836265.2160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4968/5000
    500/500 [==============================] - 0s 82us/sample - loss: -53152679.8000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4969/5000
    500/500 [==============================] - 0s 76us/sample - loss: -38378131.9040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4970/5000
    500/500 [==============================] - 0s 82us/sample - loss: -31203640.0640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4971/5000
    500/500 [==============================] - 0s 88us/sample - loss: -144504016.6400 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4972/5000
    500/500 [==============================] - 0s 90us/sample - loss: -73875983.8080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4973/5000
    500/500 [==============================] - 0s 96us/sample - loss: -46607180.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4974/5000
    500/500 [==============================] - 0s 90us/sample - loss: -81620791.6800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4975/5000
    500/500 [==============================] - 0s 80us/sample - loss: 17315520.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4976/5000
    500/500 [==============================] - 0s 76us/sample - loss: -148884226.3040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4977/5000
    500/500 [==============================] - 0s 76us/sample - loss: 122707164.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4978/5000
    500/500 [==============================] - 0s 138us/sample - loss: 122839927.2320 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4979/5000
    500/500 [==============================] - 0s 112us/sample - loss: 6464463.1680 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4980/5000
    500/500 [==============================] - 0s 100us/sample - loss: 35788236.5440 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4981/5000
    500/500 [==============================] - 0s 94us/sample - loss: -189863392.3200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4982/5000
    500/500 [==============================] - 0s 88us/sample - loss: 77093670.0800 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4983/5000
    500/500 [==============================] - 0s 78us/sample - loss: 20128776.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4984/5000
    500/500 [==============================] - 0s 96us/sample - loss: -104454171.5200 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4985/5000
    500/500 [==============================] - 0s 78us/sample - loss: -94629852.4160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4986/5000
    500/500 [==============================] - 0s 86us/sample - loss: -1701828.1280 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4987/5000
    500/500 [==============================] - 0s 92us/sample - loss: -63455618.5840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4988/5000
    500/500 [==============================] - 0s 96us/sample - loss: 97611139.2000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4989/5000
    500/500 [==============================] - 0s 92us/sample - loss: -2042168.7040 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4990/5000
    500/500 [==============================] - 0s 80us/sample - loss: 32763248.2240 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4991/5000
    500/500 [==============================] - 0s 92us/sample - loss: -133381073.7920 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4992/5000
    500/500 [==============================] - 0s 100us/sample - loss: -52690331.1360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4993/5000
    500/500 [==============================] - 0s 96us/sample - loss: -42325276.8640 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4994/5000
    500/500 [==============================] - 0s 98us/sample - loss: -209314086.0160 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4995/5000
    500/500 [==============================] - 0s 98us/sample - loss: -48833204.3520 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4996/5000
    500/500 [==============================] - 0s 82us/sample - loss: 43442463.6960 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4997/5000
    500/500 [==============================] - 0s 84us/sample - loss: 182596336.3840 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4998/5000
    500/500 [==============================] - 0s 82us/sample - loss: 32621283.0080 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 4999/5000
    500/500 [==============================] - 0s 92us/sample - loss: 20418324.7360 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    Epoch 5000/5000
    500/500 [==============================] - 0s 86us/sample - loss: 17983257.6000 - accuracy: 0.0000e+00 - val_loss: -213561480.1304 - val_accuracy: 0.0000e+00
    




    <tensorflow.python.keras.callbacks.History at 0x1dfb096ff48>




```python

```
