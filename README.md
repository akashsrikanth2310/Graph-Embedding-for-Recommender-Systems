# Graph-Embedding-for-Recommender-Systems
Graph Embedding for Recommender Systems

In this project, we will revisit the problem central to recommender systems: predicting a user’s preference for some item they have not yet rated. Like the Spark recommender from the first project, we will use a collaborative filtering model to explore this problem. Recall that in this model, the goal is to find the sentiment of a user about a particular item. 
Unlike the first project that used the ALS method, however, we will perform this task using a graph-based technique called DeepWalk.

The main steps that you will complete are to: 
1. Create a heterogeneous information network with nodes consisting of users, item-ratings, items, and other entities related to those items; 
2. Use DeepWalk to generate random walks over this graph; 
3. Based on these random walks, embed the graph in a low dimensional vector space using word2vec.     

The goal of this project is to evaluate and compare preference propagation algorithms in heterogeneous information networks generated from user-item relationships. Specifically, you will implement and evaluate a word2vec-based method. 
