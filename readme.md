# GraphSage on Pytorch
This is a simple pytorch version of GraphSage.
## models
### aggregator
mean, sum, max aggregator supported.

### GraphSage Layer
every layer corresponds to one sampling operation, in other words, your graphsage layer should be equal to your sampling layer:   
for example:   
if your sampling is [10, 25] ->(10 level-1 neibors and 25 level-2 neibors for each node in level 1 neibor), your graphsage layer should be 2

## sampling 
run the sampling test lines in utils.py to see the data structure defined
## training
the cora dataset is based on a classification background, so we are using GraphSage as the encoder, connected with a single linear layer to predict the node labels
## evaluation
Test Loss: 1.268103003501892, Accuracy: 0.8968957662582397
