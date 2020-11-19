#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import numpy as np
import sklearn
import string

from config import device


"""
This code includes the input processing, neural network and loss function of the 
sentiment analysis program. The input processing functions are used to simplify the 
review data before it reaches the network by removing punctuation and any 
non-alphabetic characters. There is also a hardcoded list of stop words that are 
removed from each review to reduce noise. Each remaining word is converted to a 
GloVe vector of dimension 300. The network receives these sets of GLoVe vectors in 
batches of 32 and they’re processed by an LSTM layer. The LSTM is multilayered and 
bidirectional, meaning that it processes data both forwards and backwards (or left 
to right and right to left). The final hidden state of the LSTM is captured and fed 
forward through a dropout layer, and two separate output layers; one has a single 
output with sigmoid activation so that that the output is between 0 and 1, the other 
has 5 outputs with softmax activation so that it produces a set of probabilities that 
sum to 1. These are the rating and category outputs respectively. To calculate a loss 
for the given output, the binary cross entropy loss for the rating output and the cross 
entropy loss for the category output are summed. Finally, after the network is trained, 
the net output is converted to two numbers, a zero or one for the rating and an integer 
between 1 and 4 for the category. The network uses an Adam optimisation algorithm with 
a learning rate of 0.0006.

We decided early on to start with an LSTM approach rather than other forms of recurrent 
networks. We experimented with a convolutional network; the accuracy for our configuration 
was similar but more computationally expensive, so we defaulted back to an LSTM set up. 
We also had between 2 and 3 hidden layers in our model for a while, but found that the 
accuracy was unaffected when they were removed, so we deemed them unnecessary. We decided 
to switch the optimiser from SGD to Adam because we learnt that Adam was a widely used optimiser 
for natural language processing. After changing to an Adam optimiser, it was easier to experiment 
as we no longer had to modify the momentum parameter manually. We also found that our 
configuration was most accurate with a very small learning rate.
"""




################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    '''
    Our preprocessing steps involve cleaning the input text by:
        - removing punctuation
        - removing non-alphabetic characters
        - normalizing case

    This is important since removing variations in text such 
    as case and punctuation will no longer affect the encoding 
    of words into dense vectors.
    '''

    # Remove punctuation
    punctuation = str.maketrans('','', string.punctuation)
    words = [w.translate(punctuation) for w in sample]

    # Remove non-alphabetic 
    words = [w for w in words if w.isalpha()]
    
    # Lower case everything
    words = [w.lower() for w in words]

    return words

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

def customStopwords():
    words = ["a","about","above","after","again","against","ain","all","am","an","and",
    "any","are","aren","aren't","as","at","be","because","been","before","being",
    "below","between","both","but","by","can","couldn","couldn't","d","did",
    "didn","didn't","do","does","doesn","doesn't","doing","don","don't","down",
    "during","each","few","for","from","further","had","hadn","hadn't","has",
    "hasn","hasn't","have","haven","haven't","having","he","her","here","hers",
    "herself","him","himself","his","how","i","if","in","into","is","isn",
    "isn't","it","it's","its","itself","just","ll","m","ma","me","mightn",
    "mightn't","more","most","mustn","mustn't","my","myself","needn","needn't",
    "no","nor","not","now","o","of","off","on","once","only","or","other","our",
    "ours","ourselves","out","over","own","re","s","same","shan","shan't","she",
    "she's","should","should've","shouldn","shouldn't","so","some","such","t",
    "than","that","that'll","the","their","theirs","them","themselves","then",
    "there","these","they","this","those","through","to","too","under","until",
    "up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what",
    "when","where","which","while","who","whom","why","will","with","won",
    "won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've",
    "your","yours","yourself","yourselves","could","he'd","he'll","he's","here's",
    "how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's",
    "there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've",
    "what's","when's","where's","who's","why's","would","able","abst","accordance",
    "according","accordingly","across","act","actually","added","adj","affected",
    "affecting","affects","afterwards","ah","almost","alone","along","already",
    "also","although","always","among","amongst","announce","another","anybody",
    "anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently",
    "approximately","arent","arise","around","aside","ask","asking","auth","available",
    "away","awfully","b","back","became","become","becomes","becoming","beforehand",
    "begin","beginning","beginnings","begins","behind","believe","beside","besides",
    "beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes",
    "certain","certainly","co","com","come","comes","contain","containing","contains",
    "couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg",
    "eight","eighty","either","else","elsewhere","end","ending","enough","especially",
    "et","etc","even","ever","every","everybody","everyone","everything","everywhere",
    "ex","except","f","far","ff","fifth","first","five","fix","followed","following",
    "follows","former","formerly","forth","found","four","furthermore","g","gave","get",
    "gets","getting","give","given","gives","giving","go","goes","gone","got","gotten",
    "h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon",
    "hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate",
    "immediately","importance","important","inc","indeed","index","information","instead","invention",
    "inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely",
    "last","lately","later","latter","latterly","least","less","lest","let","lets","like",
    "liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly",
    "make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg",
    "might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na",
    "name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither",
    "never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless",
    "noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often",
    "oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside",
    "overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps",
    "placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly",
    "present","previously","primarily","probably","promptly","proud","provides","put","q",
    "que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently",
    "ref","refs","regarding","regardless","regards","related","relatively","research","respectively",
    "resulted","resulting","results","right","run","said","saw","say","saying","says","sec",
    "section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent",
    "seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant",
    "significantly","similar","similarly","since","six","slightly","somebody","somehow","someone",
    "somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically",
    "specified","specify","specifying","still","stop","strongly","sub","substantially","successfully",
    "sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks",
    "thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll",
    "thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though",
    "thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards",
    "tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike",
    "unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using",
    "usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt",
    "way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter",
    "whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod",
    "whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without",
    "wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't",
    "allow","allows","apart","appear","appreciate","appropriate","associated","best","better",
    "c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering",
    "corresponding","course","currently","definitely","described","despite","entirely","exactly",
    "example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated",
    "indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second",
    "secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]
    
    return words

#nltk.download('stopwords')
#stopWords = set(stopwords.words('english'))
stopWords = set(customStopwords())
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """

    return ratingOutput.round().long(), categoryOutput.argmax(axis=1).long()

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()

        inputSize = wordVectors.dim
        hiddenSize = 100
        self.lstmNumLayers = 3
        hiddenLayerOutputSize = 10

        self.lstm = tnn.LSTM(input_size = inputSize,
                                hidden_size = hiddenSize,
                                num_layers = self.lstmNumLayers,
                                bidirectional = True,
                                dropout=0.4,
                                batch_first=True)

        self.ratingOutputLayer = tnn.Sequential(
            tnn.Linear(2*hiddenSize, 1),
            tnn.Sigmoid()
        
        )
        self.categoryOutputLayer = tnn.Sequential(
            tnn.Linear(2*hiddenSize, 5),
            tnn.Softmax(dim=1)
          
        )

    def forward(self, input, length):
        batchSize = len(length)
        output, state = self.lstm(input)
        finalHiddenState = state[0]
        bidirectionalHiddenState = torch.cat(
            (finalHiddenState[-2,:,:], finalHiddenState[-1,:,:]), dim = 1).squeeze()
        
        rating = self.ratingOutputLayer(bidirectionalHiddenState).squeeze()
        category = self.categoryOutputLayer(bidirectionalHiddenState)
        return rating, category

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """
    '''
    For the category, Cross-Entropy Loss is used. This loss function calculates 
    the differences in two probability distrubtions for the set of 5 unique 
    categories. Similarly for ratings, a Binary Cross-Entropy Loss is used to 
    determine if a review is positive or negative.
    '''

    def __init__(self):
        super(loss, self).__init__()
        self.crossEntropyLoss = tnn.CrossEntropyLoss()
        self.binaryCrossEntropyLoss = tnn.BCELoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingLoss = self.binaryCrossEntropyLoss(ratingOutput, ratingTarget.float())
        categoryLoss = self.crossEntropyLoss(categoryOutput, categoryTarget)
        finalLoss = ratingLoss + categoryLoss
        return finalLoss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

'''
The Adam optimisation algorithm is used as it is generally known to work 
effectively for natural language processing. Rather than using SGD 
(Stochastic Gradient Descent) which has a single learning rate for all 
weight updates, Adam calculates learning rate for every parameter.
'''

trainValSplit = 0.85
batchSize = 32
epochs = 15
#optimiser = toptim.SGD(net.parameters(), lr=0.15, momentum=0.5)
#optimiser = toptim.Adam(net.parameters(), lr = 0.0015) # 79% acc
#optimiser = toptim.Adam(net.parameters(), lr = 0.0018) # 79.8%
#optimiser = toptim.Adam(net.parameters(), lr = 0.0014) # 79.66%
#optimiser = toptim.Adam(net.parameters(), lr = 0.0013) # 79.59%
#optimiser = toptim.Adam(net.parameters(), lr = 0.002) # 79.67%
# 10 epochs ^^
#optimiser = toptim.Adam(net.parameters(), lr = 0.001) # 80.46% 
# 30 epochs ^^
#optimiser = toptim.Adam(net.parameters(), lr = 0.001) # 200wv # 79.66
#optimiser = toptim.Adam(net.parameters(), lr = 0.001) # 100wv # 78.03
optimiser = toptim.Adam(net.parameters(), lr = 0.0006) # 

