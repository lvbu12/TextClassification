# TextClassification
The first solution for text classification:  
One_Hot_FeedForwardNet folder contains a feedfoward network with one hidden layer to classify chinese news text.  
And the news text is split into character level tokens, each character is presented with a one hot feature with size (chars_size, 1).  
Then add all the features which represent every character in the news text and get a sum feature with size (chars, 1), for example, [2, 0, 3, 7, ..., 1, 4].  
Feed the sum feature into the feedforward net and output prediction probs with size (label_size, 1).  
