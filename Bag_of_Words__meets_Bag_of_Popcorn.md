Bag of Words & Live Twitter feed Analysis
================

Project Briefing
================

-   We use the *Bag of Words meets Bag of Popcorn* dataset from kaggle to do a sentiment analysis of movies based on user reviews. We deal with *unstructered data* in the form of written text making use of *Natural Language Processing* libraries such as *tm* and *Rweka*. Then we apply Machine Learning algorithms such as *Logistic Regression* & *Random Forest* to predict viewer sentiment on test dat.
-   To make things interesting, we take *live data from twitter* to do real life sentiment analysis of recently released movies based on twitter posts!

Part 1 - Training Dataset- We use the "Bag of Words meets Bag of Popcorrn" dataset from kaggle to train the algorithm.
======================================================================================================================

-   Data Import

``` r
word_train = read.table("D:/datasets/kaggle/word/labeledTrain.tsv", header = T)
word_test = read.table("D:/datasets/kaggle/word/test.tsv", header = T)


full = bind_rows(word_train,word_test)
dim(word_train)
```

    ## [1] 25000     3

``` r
dim(word_test)
```

    ## [1] 25000     2

``` r
str(full)
```

    ## 'data.frame':    50000 obs. of  3 variables:
    ##  $ id       : chr  "5814_8" "2381_9" "7759_3" "3630_4" ...
    ##  $ sentiment: int  1 1 0 0 1 1 0 0 0 1 ...
    ##  $ review   : chr  "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd docu"| __truncated__ "\"The Classic War of the Worlds\" by Timothy Hines is a very entertaining film that obviously goes to great eff"| __truncated__ "The film starts with a manager (Nicholas Bell) giving welcome investors (Robert Carradine) to Primal Park . A s"| __truncated__ "It must be assumed that those who praised this film (\"the greatest filmed opera ever,\" didn't I read somewher"| __truncated__ ...

Data Pre-Processing
===================

-   We see that the review column in our data set is the variable of interest.
-   sentiment column provides the label for the training algorithm
-   We start cleaning the text

``` r
full$review = gsub(full$review, pattern = '<br />', replacement = ' ')

text = VCorpus(VectorSource(full$review))        # Creating a Corpus of reviews

text = tm_map(text,content_transformer(tolower))  # Converting to lower case

text = tm_map(text,removeNumbers)                # Removing numbers

#as.character(text[[1]])
# We can see that all text is now in lower case and numbers have been removed.

text = tm_map(text,removePunctuation)            # Removing Punctuations
```

Cleaning the text - Step 2
==========================

-   We will use a package *SnowballC* to clean the data further.

``` r
text = tm_map(text,removeWords,stopwords())      #to remove common words


text = tm_map(text,stemDocument)                 #to convert words back to root words.

text = tm_map(text,stripWhitespace)              #to remove white spaces
```

-   stopwords() contains a list of irrelevant words in English which usually doesnot help the algorithm learn about the reviews.

-   stemDocument helps in getting the root in each word so that we can train the model using just one version of the same words leading to much better accuracy.

-   removing white spaces becomes important at this point as we have removed and transformed some of the words. It will bring uniformity and make sure that only the words are retained, no empty spaces before or after them.

N grams Tokenizer
=================

-   We will now tokenize the words from the individual reviews and make coulmn of all the individual unique words. This is done using a Document term Matrix or DTM.
-   But single words in themselves maynot be as powerful in suggeting the review. We also take phrases - 2 word & 3 word phrases for better prominence of the terms indicating the sentiment of reviews. eg.s can be - "best movie ever" , "worst movie" etc.

``` r
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
```

-   We create a BigramTokenizer and a TrigramTokenizer containing 2 word and 3 word phrases respectively using *Rweka* package.

-   Creating the DTMs for 1-grams, 2-grams & 3-grams

``` r
dtm1 = DocumentTermMatrix(text)
dtm1 = removeSparseTerms(dtm1,0.95)

## this is dtm for 1 words. we will find the more important words and just use those

dtm2 = DocumentTermMatrix(text, control = list(tokenize = BigramTokenizer))
dtm2 = removeSparseTerms(dtm2,0.99)

## similarly, dtm of 2 words

dtm3 = DocumentTermMatrix(text, control = list(tokenize = TrigramTokenizer))
dtm3 = removeSparseTerms(dtm3,0.999)

##dtm for 3 words
```

Seeing the final DTMs
=====================

-   We will also see the most frequently occuring 1 word , 2-word & 3-word phrases.

``` r
dataset1 = as.data.frame(as.matrix(dtm1))
dataset2 = as.data.frame(as.matrix(dtm2))
dataset3 = as.data.frame(as.matrix(dtm3))

#for dtm1
dataset_counts1 = as.data.frame(colSums(dataset1))
dataset_counts1$word = rownames(dataset_counts1)

colnames(dataset_counts1) = c("count","word")
dataset_counts1 = dataset_counts1[c(2,1)] 
dataset_counts1 = dataset_counts1 %>% arrange(-count)

#for dtm2
dataset_counts2 = as.data.frame(colSums(dataset2))
dataset_counts2$word = rownames(dataset_counts2)

colnames(dataset_counts2) = c("count","word")
dataset_counts2 = dataset_counts2[c(2,1)] 
dataset_counts2 = dataset_counts2 %>% arrange(-count)

#for dtm3
dataset_counts3 = as.data.frame(colSums(dataset3))
dataset_counts3$word = rownames(dataset_counts3)

colnames(dataset_counts3) = c("count","word")
dataset_counts3 = dataset_counts3[c(2,1)] 
dataset_counts3= dataset_counts3 %>% arrange(-count)

head(dataset_counts1,20)
```

    ##       word  count
    ## 1     movi 100947
    ## 2     film  93927
    ## 3      one  53805
    ## 4     like  44073
    ## 5     just  34905
    ## 6     time  30691
    ## 7     good  29345
    ## 8     make  28671
    ## 9  charact  27980
    ## 10     see  27834
    ## 11     get  27812
    ## 12   watch  27590
    ## 13    even  25082
    ## 14   stori  24738
    ## 15  realli  23013
    ## 16     can  21988
    ## 17   scene  21082
    ## 18    well  19719
    ## 19    show  19658
    ## 20    look  19339

``` r
head(dataset_counts2,20)
```

    ##              word count
    ## 1       look like  3648
    ## 2      watch movi  3058
    ## 3       ever seen  2651
    ## 4  special effect  2234
    ## 5        ive seen  2190
    ## 6       dont know  2073
    ## 7     even though  1940
    ## 8    main charact  1913
    ## 9        see movi  1903
    ## 10       one best  1882
    ## 11      movi like  1824
    ## 12       year old  1763
    ## 13      wast time  1751
    ## 14      make movi  1673
    ## 15     watch film  1624
    ## 16      movi ever  1559
    ## 17    horror movi  1528
    ## 18      movi just  1523
    ## 19      good movi  1521
    ## 20      seem like  1482

``` r
head(dataset_counts3,20)
```

    ##               word count
    ## 1    ive ever seen  1032
    ## 2  worst movi ever   573
    ## 3   movi ever seen   542
    ## 4   dont wast time   371
    ## 5    movi ive ever   361
    ## 6   movi ever made   357
    ## 7   one worst movi   325
    ## 8   film ever made   318
    ## 9    movi ive seen   297
    ## 10 worst film ever   273
    ## 11   new york citi   271
    ## 12  dont get wrong   269
    ## 13   film ive seen   268
    ## 14  film ever seen   259
    ## 15  worst movi ive   235
    ## 16    world war ii   217
    ## 17 wast time money   203
    ## 18   film ive ever   191
    ## 19  one worst film   182
    ## 20 base true stori   176

Model fitting
=============

``` r
#Pre processing
final_dataset_words = bind_rows(dataset_counts1,dataset_counts2,dataset_counts3) 
final_dataset = as.data.frame(cbind(dataset1,dataset2,dataset3))

dataset_train = final_dataset[1:25000,]
dataset_test = final_dataset[25001:50000,]
```

-   We build a random forest classifier
-   Next, we also see which are the most important words accoring to the model. We have a look at top 200 such words.

Model fitting - Logistic Regression
===================================

``` r
dataset_train$y_pred = word_train$sentiment
dataset_train$y_pred = as.factor(dataset_train$y_pred)
m1 = glm(formula = y_pred ~.,
         data = dataset_train,
         family = 'binomial')
pred = predict(m1,newdata = dataset_test)
```
