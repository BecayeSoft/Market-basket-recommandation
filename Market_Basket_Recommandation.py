# Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing

"""

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None) #no header
transactions = []

dataset.values

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None) #no header
transactions = []

# Adds the value if each row as a string
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

print(dataset)

"""## Training the Apriori model on the dataset"""

from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

"""## Displaying the results

### Results from the output of the apriori function
"""

results = list(rules)
print(results)

"""### Putting the results well organised into a Pandas DataFrame

Shape of the list (First item)
```
[
RelationRecord(
      items=frozenset({'light cream', 'chicken'}), 
      support=0, 
      ordered_statistics=[ 
          OrderedStatistic(
              items_base=frozenset({'light cream'}),
              items_add=frozenset({'chicken'}),
              confidence=0,
              lift=4
          )
      ])

2nd item
3rd item
...
]
```

To get the confidence of the 1st item -> results[0][2][0][2]
"""


def inspect(results):
    lhs = [result[2][0][0] for result in results]
    rhs = [result[2][0][1] for result in results] 
    support = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lift = [result[2][0][3] for result in results]

    return list(zip(lhs, rhs, support, confidence, lift))


resultsInDataFrame = pd.DataFrame(inspect(results),
                                  columns=["Left hand side", "Right hand side", "Support", "Confidence", "Lift"])

"""### Displaying the results non sorted"""

resultsInDataFrame

"""### Displaying the results sorted by descending lifts"""

resultsInDataFrame.sort_values(by=['Lift'], ascending=False)

resultsInDataFrame.nlargest(n=10, columns='Lift')

"""## Notes

  * min_confidence: I chose 0.8. I had too few rules, so I divided it by 2. And repeat.
  * min_lift: a good lift is 3, below is irrelevant
  * min_length & max_length: min and max number of elements in our rules.
<br/>

**Let's consider the following scenarios:**
  * "Buy 1 toothpaste, Get 1 toothbrush for free". For that, we'd use min = max length = 2.
  * "Buy 10 products A and get 1 product B for free" For that, min_length = 1 and max_length = 11.
"""