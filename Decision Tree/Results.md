# Output
Following is the output recieved on running auto-efficiency.py:

```
(Data[cylinders] <= 5.0) :
  |YES - (Data[horsepower] <= 70.0) :
    |YES - (Data[model year] <= 76.5) :
      |YES - (Data[origin] <= 2.0) :
        |YES - (Data[displacement] <= 90.0) :
          |YES - 28.4
          |NO - 25.7
        |NO - (Data[displacement] <= 85.0) :
          |YES - 31.6
          |NO - 33.0
      |NO - (Data[horsepower] <= 50.0) :
        |YES - (Data[model year] <= 79.0) :
          |YES - 43.1
          |NO - 43.849999999999994
        |NO - (Data[model year] <= 79.5) :
          |YES - 34.1
          |NO - 36.550000000000004
    |NO - (Data[model year] <= 79.0) :
      |YES - (Data[weight] <= 2276.5) :
        |YES - (Data[model year] <= 76.0) :
          |YES - 26.043478260869566
          |NO - 30.25714285714286
        |NO - (Data[model year] <= 73.5) :
          |YES - 21.235294117647058
          |NO - 24.48461538461538
      |NO - (Data[origin] <= 1.5) :
        |YES - (Data[weight] <= 2601.5) :
          |YES - 30.24
          |NO - 26.89166666666667
        |NO - (Data[acceleration] <= 13.6) :
          |YES - 23.7
          |NO - 33.513333333333335
  |NO - (Data[horsepower] <= 126.0) :
    |YES - (Data[model year] <= 78.0) :
      |YES - (Data[weight] <= 3116.0) :
        |YES - (Data[horsepower] <= 95.0) :
          |YES - 21.583333333333332
          |NO - 19.416666666666668
        |NO - (Data[model year] <= 76.5) :
          |YES - 17.34090909090909
          |NO - 19.154545454545456
      |NO - (Data[weight] <= 3202.5) :
        |YES - (Data[horsepower] <= 86.5) :
          |YES - 29.5
          |NO - 24.383333333333336
        |NO - (Data[displacement] <= 225.0) :
          |YES - 19.1
          |NO - 21.700000000000003
    |NO - (Data[model year] <= 76.5) :
      |YES - (Data[horsepower] <= 150.0) :
        |YES - (Data[model year] <= 71.5) :
          |YES - 16.333333333333332
          |NO - 14.229166666666666
        |NO - (Data[displacement] <= 334.0) :
          |YES - 10.0
          |NO - 13.403225806451612
      |NO - (Data[weight] <= 4040.0) :
        |YES - (Data[horsepower] <= 138.5) :
          |YES - 17.1
          |NO - 18.657142857142855
        |NO - (Data[weight] <= 4342.5) :
          |YES - 15.571428571428571
          |NO - 16.9

RMSE using our tree =  3.394423452839987
MAE using our tree =  2.5006530256302013

RMSE using tree module from sk learn =  3.432728800094587
MAE using tree module from sk learn =  2.538966653375301 

    y_test   our_tree  sklearn_tree
0     19.0  24.484615     24.484615
1     13.0  14.229167     14.203704
2     25.4  24.484615     24.484615
3     25.4  24.383333     24.600000
4     20.5  19.154545     19.154545
..     ...        ...           ...
74    15.0  14.229167     14.203704
75    18.0  19.416667     19.416667
76    23.8  24.484615     24.484615
77    29.9  36.550000     35.433333
78    11.0  14.229167     14.203704

[79 rows x 3 columns]
```

Here we observe that the RMSE and MAE for our decision tree is slightly lesser than of Sklearn function. This could be because sklearn function may be compromising accuracy to reduce runtime. Our decision tree on the other hand is taking far longer than the sklearn function to fit the data.
