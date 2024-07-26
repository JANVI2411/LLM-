# NLP - Hugging Face

## Chapter 5: Datasets
1) How to use datasets library: 
  * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter5/3?fw=pt
  * My colab notebook: https://huggingface.co/learn/nlp-course/en/chapter5/3?fw=pt
  * Key_points:
     * Learn how to use map, apply, filter on datasets
     * fast tokenizer batched=True
     * datsets set_format("pandas") and reset_format() method.

2) Big Data Handling using Streaming :
   * Tutorial: 
   * My colab notebook: https://colab.research.google.com/drive/1plwHO6Af3CDH859grlTxYcVbaQshGXpR?usp=sharing
   * Key_points:
     * Handle big data using streaming=True
     * how hugging face uses memory mapping ( RAM to file system storage , no need to load/copy data to RAM ), implemented using Apache Arrow and Py

4) How to upload dataset to Hugging face :   
   * Tutorial: hub: https://huggingface.co/learn/nlp-course/en/chapter5/5?fw=pt
5) Semantic search with Faiss: 
   * My colab notebook: https://colab.research.google.com/drive/1qnDzN1rXj33z-lmp7P17ZLyzbBs2AIiH?usp=sharing
   *   Key_points:
       *  generate embeddings from text data   
       *  datasets -> add_fais_index
       *  datasets -> get_nearest_examples
       *  pandas explode
7) Quiz: https://huggingface.co/learn/nlp-course/en/chapter5/8?fw=pt

## Chapter 6: Tokenizer

1) Train old tokenizer on your custom data:
  * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter6/2?fw=pt
  * My colab notebook:
  * Key_points:
    * train tokenizer on python language dataset
    * create iterator(generator) to load data 
    * fast tokeinzer method train_new_from_iterator(your_iterator, vocab_size)
    * train_new_from_iterator() only works with fast tokenizers which are written in "Rust"
    * normal tokenizer are written in python
    * tokenizer save_pretrained(), push_to_hub()
    * The output of a tokenizer isn’t a simple Python dictionary; what we get is actually a special BatchEncoding object.
    * The thing that we get after we pass text into tokenizer:
       ```
           tokenized_text = Tokenizer(text_list)
           tokenized_text["input_ids"]
           tokenized_text["attention_mask"]
           tokenized_text.tokens()
           tokenized_text.word_ids() 
           tokenized_text["token_type_ids"] or tokenized_text["sequence_ids"]
           word_to_chars() or token_to_chars() and char_to_word() or char_to_token()
           
       ```
    * return_offsets_mapping=True option in tokenizer() 
     ```
           tokenized_text = Tokenizer(text_list)
           tokenized_text["offset_mapping"]
     ```

2) Fast Tokenizer in QA Pipeline
   * Tutorial : https://huggingface.co/learn/nlp-course/en/chapter6/3b?fw=pt
   * My colab notebook:
   * Key_points:
      * How to compute max score, start and end position of answer from context
      * Deal with very long contexts that end up being truncated
      * truncation=True, return_overflowing_tokens=True in tokenizer
      
3) Normalization and Pre-tokenizer
    * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter6/4?fw=pt
    * Key Points:
       * The normalization step involves some general cleanup, such as removing needless whitespace, lowercasing, and/or removing accents.
         ```tokenizer.backend_tokenizer.normalizer.normalize_str()```
       * Pre tokenizer splits a raw text into words on whitespace and punctuation
         ```tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str()```
       * subword tokenization
         
4) Byte-Pair Encoding Tokenization:
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter6/5?fw=pt
   * Key Points:
        * It’s used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa.
        * Learn its algorithm and implementation using Python
5) WordPiece tokenization:
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter6/6?fw=pt
   * Key Points:
        * WordPiece is the tokenization algorithm Google developed to pretrain BERT
        * It’s very similar to BPE in terms of the training, but the actual tokenization is done differently.
        * difference between byte-pair and wordpiece: formula to compute frequency of pairs
6) WordPiece tokenization:
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter6/7?fw=pt

7) How to build the above 3 tokenizer :
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter6/8?fw=pt

## Chapter 7: Main NLP Task

1) Token Classification:
   * Tutorial:
   * Key Points:
       * Example: Named entity recognition (NER), Part-of-speech tagging(POS)
       * dataset feature columns: features: ['ner_tags', 'pos_tags', 'tokens'], Here input columns is either ["ner_tags"] / ["pos_tags"] and label column is ["tokens"].
       * tokens column: The text is splitted into the words using punctuation or whitespace.
       * ner_tags column: ner_tag for each token in tokens column, same goes for pos_tags.
       * Dataset preparation: pass tokens from model tokenizer, get actual tokens(T2) which can be different from the dataset original tokens (T1).
       * Dataset preparation: we need to update our labels, ner_tags as per new tokens(T1), that we can do using word_ids()
       * Data collation: Here our labels should be padded the exact same way as the inputs so that they stay the same size, using -100 as a value so that the corresponding predictions are ignored in the loss computation.
       * Evaluation Metrics: seqeval, this seqeval takes actual label_names, not the labels, so we have to convert the generated label to its corresponding label_name.
       * Model Config: pass id2label and label2id in config
         
2) Fine-Tuning Masked Language Model:
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter7/3?fw=pt
   * key Points:
       * Domain Adaptation: here are a few cases where you’ll want to first fine-tune the language models on your data, before training a task-specific head. For example, if your dataset contains legal contracts or scientific articles, a vanilla Transformer model like BERT will typically treat the domain-specific words in your corpus as rare tokens, and the resulting performance may be less than satisfactory. By fine-tuning the language model on in-domain data you can boost the performance of many downstream tasks, which means you usually only have to do this step once! This process of fine-tuning a pretrained language model on in-domain data is usually called domain adaptation
       * input: text = "This is a great idea."
       * label: label= "This is a great idea."
       * Here input and labels are same, we will mask some words in input randomly using mask token [MASK]: "This is a great [MASK]."
       * Then we pass this masked input and label to tokenizer, and feed it to the model.
       * whole_word_masking_data_collator : mask whole words together, not just individual tokens
       * Evaluation metrics: Perplexity: one way to measure the quality of our language model is to calculate the probabilities it assigns to the next word in all the sentences of the test set. High probabilities indicates that the model is not “surprised” or “perplexed” by the unseen examples, and suggests it has learned the basic patterns of grammar in the language.

3) Summarization:
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter7/5?fw=pt
   * key Points:
        * input : ["text"]
        * label : ["summary"]
        * just pass input and label in tokenizer
        * Evaluation metrics: For summarization, one of the most commonly used metrics is the ROUGE score (Recall-Oriented Understudy for Gisting Evaluation)
        * Recall=  Number of overlapping words / Total number of words in reference summary
        * Precision=  Number of overlapping words  Total number of words in generated summary
        * diff types of rouge score: rouge1 is what exaplained above, rouge2 measures the overlap between bigrams (think the overlap of pairs of words), while rougeL and rougeLsum measure the longest matching sequences of words by looking for the longest common substrings in the generated and reference summaries.
        * DataCollatorForSeq2Seq : the input_ids and attention_mask of the will be padded on the right with a [PAD] token (whose ID is 0). Similarly, the labels will be  padded with -100s, to make sure the padding tokens are ignored by the loss function.

4) Language Model:
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter7/6?fw=pt
   * key Points:
        * A scaled-down version of a python code generation model: one-line completions instead of full functions or classes
        * Here we will use the tokeinzer that we fine-tuned on python dataset
        * input : ["text"]
        * No labels
        * Model config: vocab_size of fine-tuned tokenizer that we are using, fine-tuned tokenizer's bos_token_id, eos_token_id, pad_token_id
        * we will take head of GPT2 model as we have new config
        * DataCollatorForLanguageModeling
          
5) Extractive question answering:
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter7/7?fw=pt
   * key Points:
        * This involves posing questions about a document and identifying the answers as spans of text in the document itself.
        * Tokenize input and context together using return_overflowing_tokens=True and Truncation=True and return_offsets_mapping=True.
        * label: start index, end index from input context
        * Data collator: we padded all the samples to the maximum length we set, there is no data collator to define
        * Evaluation metrics: Squad
        * Tricky part: custom preprocessing and custom compute_metrics function

6) Translation:
   * Tutorial: https://huggingface.co/learn/nlp-course/en/chapter7/4?fw=pt
   * key Points:
        * Evaluation metrics: Bleu Score
   
​

​
 


 
