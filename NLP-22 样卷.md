## Q1
#### a)Estimate emission probabilities P(w|t)
$P(fly|V) = \frac {300}{20000}$
...
#### b)The problem of unigram POS tagger
1. ***Context-insensitivity***: Unigram POS taggers do not consider the context in which a word appears. This can lead to incorrect POS tags for words with multiple meanings (homonyms and polysemous words) depending on the context.
2. ***Inability to handle rare or unknown words***: Unigram POS taggers rely on the frequency of words in the training corpus to assign POS tags. Consequently, they might struggle to handle words that are rare or not present in the training data. This issue can lead to incorrect POS tags for such words.
3. ***Limited accuracy***: Unigram POS taggers typically have lower accuracy compared to more advanced POS tagging methods that consider contextual information, such as Hidden Markov Models (HMM), Conditional Random Fields (CRF), or neural network-based POS taggers.
4. ***Dependence on the training corpus***: The performance of a unigram POS tagger heavily depends on the quality and size of the training corpus. If the training data is not representative of the text to be tagged or if it is too small, the unigram POS tagger's performance may be compromised.
5. ***Lack of sequence information***: Unigram POS taggers do not consider the relationships between consecutive words or POS tags in a sentence, which can provide valuable information for assigning the correct POS tags.
#### d) explain why the viterbi algorithm is computationally efficient compared to a naive approach to inferring the correct sequence using an HMM.

1.  Dynamic programming: The Viterbi algorithm uses dynamic programming to store intermediate results and avoid redundant calculations. It builds a Viterbi table where each cell stores the highest probability for a particular state at a given time step. This allows the algorithm to compute probabilities for new states based on previously computed probabilities, avoiding the need to recompute them.
    
2.  Time complexity: The Viterbi algorithm has a time complexity of O(T * N^2), where T is the length of the observation sequence and N is the number of hidden states. The naïve approach, which explores all possible state sequences, has a time complexity of O(N^T), which grows exponentially with the length of the sequence. This makes the Viterbi algorithm much more efficient, especially for longer sequences.
    
3.  Space complexity: The Viterbi algorithm has a space complexity of O(T * N), as it requires storing the Viterbi table with T rows and N columns. In contrast, the naïve approach may require storing all possible state sequences, which can lead to a much higher space complexity.
    
4.  Optimal substructure: The Viterbi algorithm exploits the optimal substructure property of the HMM, meaning that the optimal solution for a subproblem can be used to construct the optimal solution for the original problem. By solving subproblems optimally and combining their solutions, the Viterbi algorithm can efficiently find the most likely sequence of hidden states.
    
5.  Traceback: After constructing the Viterbi table, the algorithm can efficiently backtrack to find the most likely sequence of hidden states by following the path with the highest probability. This traceback step has a time complexity of O(T), adding to the overall efficiency of the Viterbi algorithm.
    
  The Viterbi algorithm is computationally efficient compared to a naïve approach due to its use of dynamic programming, optimal substructure, and a more favorable time and space complexity. This efficiency enables it to handle longer sequences and larger state spaces, making it a popular choice for inferring the correct sequence in HMMs.

#### e) CRF vs. HMM
1.  Word identity: The identity of the current word itself is an important feature, as certain words may be more likely to be associated with specific POS tags.
    
2.  Surrounding words: The words appearing before and after the current word can provide important context for determining the POS tag. Including a window of surrounding words, such as the previous one or two words and the next one or two words, can help the CRF model capture this information.
    
3.  Word suffixes and prefixes: The suffixes and prefixes of a word can provide hints about its grammatical role. For example, words ending in "-ing" are often verbs, while words ending in "-ly" are often adverbs. Including features that represent common prefixes and suffixes can help the CRF model recognize such patterns.
    
4.  Capitalization: The capitalization of a word can be informative for POS tagging. Proper nouns are typically capitalized, whereas other nouns, verbs, adjectives, etc., are usually not capitalized unless they appear at the beginning of a sentence. Including a feature that captures the capitalization information can help the CRF model distinguish between proper nouns and other POS tags.
## Q2
#### a)Compared with traditional lexical semantics (e.g., dictionary- based semantics), what are the advantages and disadvantages of distributional semantics?
Advantages of distributional semantics over traditional lexical semantics:

1.  Data-driven approach: Distributional semantics is derived from analyzing large amounts of text data, capturing the statistical properties of words and their co-occurrences. This enables the model to learn and adapt to new words and language variations without manual intervention.
    
2.  Capturing semantic relationships: Distributional semantics can uncover semantic relationships between words that may not be explicitly defined in a dictionary, such as synonyms, antonyms, or words that are semantically related.
    
3.  Handling polysemy and context: Distributional semantics can represent words in context, which can help disambiguate words with multiple meanings (polysemy) based on their usage patterns.
    
4.  Continuous representation: Distributional semantics provides a continuous vector representation of words, which allows for more flexible and nuanced comparisons of word similarities.
    

Disadvantages of distributional semantics compared to traditional lexical semantics:

1.  Dependency on large corpora: Distributional semantics relies on the availability of large text corpora to learn meaningful word representations. This can be a limitation for low-resource languages or specialized domains where large text corpora are not available.
    
2.  Lack of interpretability: The high-dimensional vector representations generated by distributional semantics can be difficult to interpret compared to traditional dictionary-based semantics, where words are associated with human-readable definitions and relationships.
    
3.  Sensitivity to corpus quality: The quality of the word representations depends on the quality and diversity of the corpus. Noisy or biased corpora may lead to inaccurate or biased representations.
    
4.  Ambiguity and polysemy: While distributional semantics can handle polysemy to some extent, it may still struggle to accurately represent the meaning of words with multiple senses in different contexts. Some advanced techniques like context-aware embeddings (e.g., BERT) can address this issue, but they come with their own computational challenges.
#### b) Assume we have a toy corpus of 3 documents as follows. "This bank has 5 cash counters”; “Shanghai is the financial center of China”; " There are 5 buildings on the river bank”. Though in the course we only discussed how to use lexical semantics to compare similarities between words in meaning, it is also possible to extend it for comparing similarities between documents. Design a dictionary-based method to compare the similarities among the 3 documents.
A dictionary-based method for comparing the similarities among documents can be achieved using the following steps:

1.  Preprocessing: Tokenize each document into a list of words, and remove any punctuation marks or stop words (common words like 'is', 'the', etc.). Also, convert all words to lowercase and, if desired, apply stemming or lemmatization to reduce words to their root form.
    
2.  Create a dictionary: For each unique word in the corpus, create a dictionary entry with the word as the key and a list of its synonyms, antonyms, hypernyms, and hyponyms as the value. This can be done using resources like WordNet or other lexical databases.
    
3.  Document representation: Convert each document into a set of unique words, including the original words and their related words from the dictionary. For example, if the word 'bank' is in the document, its set would include 'bank' and its synonyms, antonyms, hypernyms, and hyponyms.
    
4.  Compute similarity: Calculate the similarity between pairs of documents using a set-based similarity measure such as Jaccard similarity or Dice coefficient. These measures compute the similarity as the ratio of the size of the intersection of the sets to the size of their union or the average of their sizes, respectively.
    

Example:

Document 1: {bank, financial_institution, cash_counter} Document 2: {Shanghai, financial_center, China} Document 3: {building, river_bank, river}

Jaccard similarity:

-   Similarity(D1, D2) = |Intersection(D1, D2)| / |Union(D1, D2)| = 0 / 8 = 0
-   Similarity(D1, D3) = |Intersection(D1, D3)| / |Union(D1, D3)| = 1 / 6 = 0.1667
-   Similarity(D2, D3) = |Intersection(D2, D3)| / |Union(D2, D3)| = 0 / 7 = 0

The Jaccard similarity values show that Document 1 and Document 3 are more similar to each other than either is to Document 2, as they share the term 'bank' and its related terms in their sets. This dictionary-based method provides a simple approach to comparing document similarity based on lexical semantics. However, it may not be as effective for large or diverse corpora, as it relies on the quality and coverage of the dictionary and does not consider the distributional information of words.

#### c)Assume you are provided with a very large corpus of 10,000,000 documentscovering a wide range of topics. Design a method based on distributional semantics to compare the similarities among the 3 documents. What problem(s) will you encounter? Explain how you can solve these problems.
When designing a method based on distributional semantics to compare the similarities among three documents within a large corpus, you may encounter several problems:

1. Scalability and computational complexity: Training word embeddings on a corpus of 10,000,000 documents can be computationally expensive and time-consuming.

   Solution: Use pre-trained embeddings if available for your domain, or leverage techniques like dimensionality reduction (e.g., PCA) to reduce the size of the embeddings. Alternatively, utilize distributed computing or GPU acceleration to speed up the training process.

2. Ambiguity and polysemy: Words with multiple meanings may have embeddings that do not accurately represent their meaning in the context of a specific document.

   Solution: Use algorithms like sense2vec or context-aware language models like BERT, which can generate context-aware embeddings that consider the meaning of a word in the context of the surrounding words.

3. Sparse or uncommon terms: Some words may not have sufficient co-occurrence data in the corpus to generate accurate embeddings.

   Solution: Employ techniques like subword-level embeddings (e.g., FastText) or incorporate external knowledge sources like dictionaries, thesauri, or knowledge graphs to enhance the representations.

4. Noisy data and preprocessing challenges: In a large and diverse corpus, handling different languages, dialects, or domain-specific jargon might be challenging.

   Solution: Apply thorough preprocessing, including language identification, spell-checking, and domain-specific tokenization. Additionally, consider training separate models for different languages or domains if necessary.

5. Evaluating document similarity: Directly comparing document representations may not capture more nuanced or complex relationships between documents.

   Solution: Instead of just averaging word embeddings, experiment with advanced techniques like Doc2Vec, which learns document representations directly, or use clustering methods (e.g., K-means, hierarchical clustering) to group similar documents together.

By addressing these problems, you can develop a more robust method for comparing document similarities in a large corpus using distributional semantics, accommodating a wide range of topics.

## Q3
#### b) 词汇干扰问题处理方法（使用BoW）
Negation scope marking, By using this negation scope marking method, you can capture the effect of negation in the bag-of-words model. This will allow the model to differentiate between the original words and their negated counterparts, incorporating the relevant sentiment signals in the dataset.

1. Identify negation words: Create a list of common negation words, such as "not", "isn't", "aren't", "doesn't", "don't", "won't", "never", etc. You can also include domain-specific negation words if needed.

2. Tokenize the sentences: Split each sentence in the dataset into individual words or tokens.

3. Mark negation scope: Iterate through the tokens in each sentence. When you encounter a negation word, mark the scope of negation, which typically includes the words that follow the negation word up to the next punctuation mark or end of the sentence. You can also consider using a fixed window size, for example, marking the next N words after the negation word.

4. Modify words in the negation scope: For each word in the negation scope, prepend a prefix like "NOT_" to the word. This will create new tokens that represent the negated meaning of the original words, e.g., "not fast" will be represented as "NOT_fast" and "not good" as "NOT_good".

5. Create the bag-of-words representation: After preprocessing the sentences with the above method, create the bag-of-words representation for the modified sentences.


#### c) data bias问题处理
Handling class imbalance is crucial to avoid biased predictions in a classifier. Here are several strategies to ease the problem of class imbalance in your review dataset:

1.  Resampling the data: a. Oversampling the minority class: Create copies of instances from the positive class (20%) until it reaches a more balanced distribution. This can be done using techniques like random oversampling or Synthetic Minority Over-sampling Technique (SMOTE). b. Undersampling the majority class: Randomly remove instances from the negative class (80%) to achieve a more balanced distribution. However, this may result in loss of information.
    
2.  Assigning class weights: Assign higher weights to the minority class (positive) and lower weights to the majority class (negative) during the training process. This will make the classifier more sensitive to the minority class, thus compensating for the imbalance.
    
3.  Using cost-sensitive learning: Introduce different misclassification costs for the two classes. For example, penalize the model more for misclassifying positive instances than negative instances. This encourages the model to pay more attention to the minority class.
    
4.  Ensemble methods: Use ensemble techniques, such as bagging and boosting, with a focus on handling class imbalance. For example, you can use balanced random under-sampling with bagging, or use boosting algorithms like AdaBoost with cost-sensitive learning.
    
5.  Evaluating model performance with appropriate metrics: Accuracy might not be the best metric to evaluate the performance of a classifier in the presence of class imbalance. Instead, use metrics like precision, recall, F1-score, and Area Under the Receiver Operating Characteristic Curve (AUROC) to assess the performance of the model.
    
6.  Using advanced models: Some advanced machine learning models, such as deep learning models, can be more resilient to class imbalance. However, they often require large amounts of data and computational resources.
    

Applying one or a combination of these strategies can help mitigate the effect of class imbalance on your classifier's predictions and improve the overall performance of the model.

#### d) Evaluation index analysis
Accuracy is not a good metric for evaluating classifier performance on imbalanced datasets because it can be misleading. A classifier that always predicts the majority class would have a high accuracy, but it would fail to identify any instances from the minority class. In such cases, precision, recall, and F1-score are more informative metrics to evaluate the classifier's performance.

To identify as many positive sentences as possible from an unseen test dataset, you should focus on the classifier with a higher recall. Recall (or sensitivity) measures the proportion of actual positive instances that are correctly identified by the classifier. A higher recall means the classifier can identify a larger fraction of positive instances in the dataset.

When comparing the precision-recall curves of c1 and c2, c2 is the better. This indicates a higher recall rate at different levels of precision, which means the classifier can better identify positive instances while maintaining a reasonable level of precision.