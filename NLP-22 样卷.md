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
The Viterbi algorithm is an efficient dynamic programming method used for finding the most likely sequence of hidden states (in this case, POS tags) in a Hidden Markov Model (HMM). It is computationally efficient compared to a naïve approach, such as brute force search, for several reasons:

1.  Dynamic programming: The Viterbi algorithm uses dynamic programming to store intermediate results and avoid redundant calculations. It builds a Viterbi table where each cell stores the highest probability for a particular state at a given time step. This allows the algorithm to compute probabilities for new states based on previously computed probabilities, avoiding the need to recompute them.
    
2.  Time complexity: The Viterbi algorithm has a time complexity of O(T * N^2), where T is the length of the observation sequence and N is the number of hidden states. The naïve approach, which explores all possible state sequences, has a time complexity of O(N^T), which grows exponentially with the length of the sequence. This makes the Viterbi algorithm much more efficient, especially for longer sequences.
    
3.  Space complexity: The Viterbi algorithm has a space complexity of O(T * N), as it requires storing the Viterbi table with T rows and N columns. In contrast, the naïve approach may require storing all possible state sequences, which can lead to a much higher space complexity.
    
4.  Optimal substructure: The Viterbi algorithm exploits the optimal substructure property of the HMM, meaning that the optimal solution for a subproblem can be used to construct the optimal solution for the original problem. By solving subproblems optimally and combining their solutions, the Viterbi algorithm can efficiently find the most likely sequence of hidden states.
    
5.  Traceback: After constructing the Viterbi table, the algorithm can efficiently backtrack to find the most likely sequence of hidden states by following the path with the highest probability. This traceback step has a time complexity of O(T), adding to the overall efficiency of the Viterbi algorithm.
    

In summary, the Viterbi algorithm is computationally efficient compared to a naïve approach due to its use of dynamic programming, optimal substructure, and a more favorable time and space complexity. This efficiency enables it to handle longer sequences and larger state spaces, making it a popular choice for inferring the correct sequence in HMMs.

#### e) 
1.  Word identity: The identity of the current word itself is an important feature, as certain words may be more likely to be associated with specific POS tags.
    
2.  Surrounding words: The words appearing before and after the current word can provide important context for determining the POS tag. Including a window of surrounding words, such as the previous one or two words and the next one or two words, can help the CRF model capture this information.
    
3.  Word suffixes and prefixes: The suffixes and prefixes of a word can provide hints about its grammatical role. For example, words ending in "-ing" are often verbs, while words ending in "-ly" are often adverbs. Including features that represent common prefixes and suffixes can help the CRF model recognize such patterns.
    
4.  Capitalization: The capitalization of a word can be informative for POS tagging. Proper nouns are typically capitalized, whereas other nouns, verbs, adjectives, etc., are usually not capitalized unless they appear at the beginning of a sentence. Including a feature that captures the capitalization information can help the CRF model distinguish between proper nouns and other POS tags.

## Q3
#### b)
To deal with negation appropriately in a bag-of-words representation, you can use a simple preprocessing method called "negation scope marking." This method involves modifying words that appear in the scope of negation, so that the negation effect is captured by the bag-of-words model. Here's a step-by-step outline of the method:

1. Identify negation words: Create a list of common negation words, such as "not", "isn't", "aren't", "doesn't", "don't", "won't", "never", etc. You can also include domain-specific negation words if needed.

2. Tokenize the sentences: Split each sentence in the dataset into individual words or tokens.

3. Mark negation scope: Iterate through the tokens in each sentence. When you encounter a negation word, mark the scope of negation, which typically includes the words that follow the negation word up to the next punctuation mark or end of the sentence. You can also consider using a fixed window size, for example, marking the next N words after the negation word.

4. Modify words in the negation scope: For each word in the negation scope, prepend a prefix like "NOT_" to the word. This will create new tokens that represent the negated meaning of the original words, e.g., "not fast" will be represented as "NOT_fast" and "not good" as "NOT_good".

5. Create the bag-of-words representation: After preprocessing the sentences with the above method, create the bag-of-words representation for the modified sentences.

By using this negation scope marking method, you can capture the effect of negation in the bag-of-words model. This will allow the model to differentiate between the original words and their negated counterparts, incorporating the relevant sentiment signals in the dataset.