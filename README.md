# AI-Driven Content Source Identification  

### Overview  
With the rapid advancement of AI tools like ChatGPT, Grammarly, and Perplexity, distinguishing between human-written and AI-generated text has become increasingly difficult. Traditional plagiarism detection methods are no longer sufficient to ensure content integrity, especially in academic and professional settings.  

This project presents a deep learning-based approach to classify text as human-written, AI-generated, or paraphrased using state-of-the-art Large Language Models (LLMs) such as:  

- RoBERTa  
- ELECTRA  
- GPT-Neo  
- Mistral  

### Dataset  
The dataset consists of 46,181 labeled text samples collected from:  
- Human-written text from verified sources like academic papers and educational websites.  
- AI-generated text from ChatGPT-4.  
- Paraphrased text created using tools like QuillBot and Grammarly.  

### Methodology  
1. Data Collection & Preprocessing  
   - Web scraping human-written content using BeautifulSoup.  
   - Generating AI-based content and paraphrased text for balanced dataset creation.  
   - Text normalization, tokenization, and removal of duplicates.  

2. Model Training & Fine-Tuning  
   - Training on Hugging Face Transformers and PyTorch.  
   - Hyperparameter tuning via grid search for optimal learning rates, batch sizes, and epochs.  
   - Using advanced prompting techniques to improve classification accuracy:  
     - Chain of Thought (CoT)  
     - Least-to-Most (L2M)  
     - Self-Consistency  
     - Direct Prompting  

3. Results & Model Performance  
   - GPT-Neo achieved the best accuracy of 95.73% with an F1-score of 96%.  
   - ELECTRA and RoBERTa also performed well, with accuracy above 92%.  
   - Mistral, despite having 7 billion parameters, required further fine-tuning to reach its full potential.  

### Key Features  
✅ Detects whether a given text is human-written, AI-generated, or paraphrased.  
✅ Implements multiple LLMs and compares their performance.  
✅ Uses advanced NLP techniques to refine model predictions.  
✅ Supports explainability with logits to understand decision-making.  

### Technologies Used  
- Python, PyTorch, Hugging Face Transformers  
- BeautifulSoup, Scrapy (Web Scraping)  
- NVIDIA GeForce RTX 4090 GPU (Model Training)  
- Amazon S3, Jupyter Notebook  

### Future Improvements  
🔹 Expand dataset to cover diverse writing styles and domains.  
🔹 Fine-tune Mistral for better performance on paraphrased content.  
🔹 Deploy as a web app/API for real-time text classification.  
🔹 Explainable AI (XAI) techniques for transparency in classification.  

 This project detects whether a given text is human-written, AI-generated, or paraphrased using state-of-the-art NLP models. The model is deployed on Hugging Face Spaces, allowing anyone to use it without installation.
🔗 Try the Live Demo Here:
👉 AI Content Detector on Hugging Face Spaces (https://huggingface.co/spaces/vai0511/ai-content-classifier-app)

