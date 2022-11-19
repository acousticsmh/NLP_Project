In this folder there are 4 main files for preprocessing :
sentenceSegmentation.py
stopwordRemoval.py
tokenization.py
inflectionReduction.py

There are four approaches tried (also included the baseline TF-IDF :
LSA
pLSA
LDA
BM25

Each of those have their own informationRetrieval files for document representation and relevance ranking procedures.

To run, for all queries in the dataset,run
> python main.py

A question will be asked,
>Enter the method, pLSA or LSA or LDA or BM25 or TF-IDF

Enter the exact string relevant to the method to be tried.
Then, the results for that method will be printed and the output folder will be populated with the preprocessed docs (at different stages), and the 
plot of evaluation measures for the tried method.






When the -custom flag is passed, the system will take a query from the user as input. For example:
> python main.py -custom
> Enter query below
> Papers on Aerodynamics
This will print the IDs of the five most relevant documents to the query to standard output.

When the flag is not passed, all the queries in the Cranfield dataset are considered and precision@k, recall@k, f-score@k, nDCG@k and the Mean Average Precision are computed.
