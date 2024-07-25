# NLP - Hugging Face

# Chapter 5: Datasets
1) How to use datasets library: 
  * tutirial: https://huggingface.co/learn/nlp-course/en/chapter5/3?fw=pt
  * my colab notebook: https://huggingface.co/learn/nlp-course/en/chapter5/3?fw=pt
  * key_points:
     * Learn how to use map, apply, filter on datasets
     * fast tokenizer batched=True
     * datsets set_format("pandas") and reset_format() method.

2) Big Data Handling using Streaming : https://colab.research.google.com/drive/1plwHO6Af3CDH859grlTxYcVbaQshGXpR?usp=sharing
   * key_points:
     * Handle big data using streaming=True
     * how hugging face uses memory mapping ( RAM to file system storage , no need to load/copy data to RAM ), implemented using Apache Arrow and Py

3) How to upload dataset to Hugging face hub: https://huggingface.co/learn/nlp-course/en/chapter5/5?fw=pt
4) Semantic search with Faiss: https://colab.research.google.com/drive/1qnDzN1rXj33z-lmp7P17ZLyzbBs2AIiH?usp=sharing
   *   key_points:
       *  generate embeddings from text data   
       *  datasets -> add_fais_index
       *  datasets -> get_nearest_examples
       *  pandas explode
5) Quiz: https://huggingface.co/learn/nlp-course/en/chapter5/8?fw=pt

# Chapter 5: Tokenizer
