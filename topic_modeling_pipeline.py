# topic_modeling_pipeline.py
# --------------------------
# Pipeline for topic modeling of PDF documents using Latent Dirichlet Allocation (LDA).
# Steps:
# 1. Extract text from PDF
# 2. Preprocess text (tokenization, stopwords, lemmatization)
# 3. Build document-term matrix
# 4. Compute coherence scores for multiple topic numbers
# 5. Select optimal number of topics
# 6. Train LDA model and generate visualization (pyLDAvis)

import PyPDF2
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim
import pyLDAvis
import pickle
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----------------------
# 1. Extract text from PDF
# ----------------------
def extract_text_from_pdf(file_path):
    """Extract all text from a PDF file."""
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# ----------------------
# 2. Preprocess text
# ----------------------
def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize text."""
    tokens = simple_preprocess(text, deacc=True)

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# ----------------------
# 3. Compute coherence for multiple topic numbers
# ----------------------
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1, random_state=100):
    """
    Compute coherence scores for LDA models with different topic counts.
    """
    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=random_state
        )
        model_list.append(model)

        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v'
        )
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# ----------------------
# 4. Main execution
# ----------------------
if __name__ == "__main__":
    # Path to your PDF file
    pdf_file_path = 'data/your_corpus.pdf'

    # Output directory for pyLDAvis files
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Extract and preprocess text
    print("Extracting text from PDF...")
    document_text = extract_text_from_pdf(pdf_file_path)
    print("Preprocessing text...")
    processed_text = preprocess_text(document_text)

    # Create dictionary and corpus
    dictionary = corpora.Dictionary([processed_text])
    corpus = [dictionary.doc2bow(processed_text)]

    # Find the optimal number of topics
    print("Computing coherence values...")
    start, limit, step = 2, 10, 1
    model_list, coherence_values = compute_coherence_values(
        dictionary, corpus, [processed_text], limit, start, step, random_state=100
    )

    optimal_num_topics = start + coherence_values.index(max(coherence_values))
    print(f"Optimal number of topics: {optimal_num_topics}")

    lda_model = model_list[coherence_values.index(max(coherence_values))]

    print("Topics discovered:")
    for topic in lda_model.print_topics(num_words=5):
        print(topic)

    # Visualize using pyLDAvis
    print("Preparing visualization...")
    LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

    lda_vis_file = os.path.join(output_dir, f'ldavis_prepared_{optimal_num_topics}.html')
    with open(os.path.join(output_dir, f'ldavis_prepared_{optimal_num_topics}'), 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
    pyLDAvis.save_html(LDAvis_prepared, lda_vis_file)

    print(f"Visualization saved to {lda_vis_file}")
