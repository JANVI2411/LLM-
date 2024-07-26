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
2) 
 
