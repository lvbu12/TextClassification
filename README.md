# TextClassification
## Solution one - One Hot FeedForwardNet  
* One_Hot_FeedForwardNet folder contains a feedfoward network with one hidden layer to classify chinese news text.  
* And the news text is split into character level tokens, each character is presented with a one hot feature with size (chars_size, 1).  
* Then add all the features which represent every character in the news text and get a sum feature with size (chars, 1), for example, [2, 0, 3, 7, ..., 1, 4].  
* Feed the sum feature into the feedforward net and output prediction probs with size (label_size, 1).  

### Config
The feedforward network's configurations is stored in [One_Hot_FeedForwardNet/Configs/config.json](https://github.com/lvbu12/TextClassification/tree/master/One_Hot_FeedForwardNet/Configs/).

### Train network
* Check the config json file, and push the data according to the config file.
* `python One_Hot_FeedForwardNet/ffn_train.py`

### Test Model
* Pull the test data in correct place and select the location of prediction file according to the config file.
* 'python One_Hot_FeedForwardNet/ffn_test.py'

### Report
* Run function `gen_csv_report` in One_Hot_FeedForwardNet/report.py to get the report csv which contains the confusion matrix.
* Run function `compute_macro_F1` or 'compute_micro_F1' to get the macro F1 score or micro F1 score from the confusion matrix.
