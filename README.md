# MyTourismRAG
# 🌍 Darija Tourism Chatbot

A RAG-powered chatbot that answers tourism questions about Morocco in Moroccan Darija (Moroccan Arabic dialect).

## 📋 Overview

This project implements a Retrieval-Augmented Generation (RAG) system using Cohere's language model and LangChain to create an intelligent tourism assistant for Morocco. The chatbot can answer questions in Darija about tourist destinations, travel tips, and Moroccan culture.

## ✨ Features

- **Multilingual Support**: Handles questions in Moroccan Darija
- **RAG Architecture**: Combines retrieval with generation for accurate responses
- **Tourism-Focused**: Specializes in Moroccan tourism information
- **Interactive UI**: Gradio interface for easy interaction
- **Evaluation Metrics**: Built-in accuracy measurement system

## 🛠️ Technology Stack

- **LangChain**: Framework for building LLM applications
- **Cohere API**: State-of-the-art language model (command-a-03-2025)
- **FAISS**: Vector database for efficient similarity search
- **HuggingFace Embeddings**: Multilingual sentence embeddings
- **Gradio**: Web interface for chatbot interaction
- **Datasets**: Moroccan Darija QA dataset from HuggingFace

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Cohere API key (get one at [cohere.com](https://cohere.com))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/darija-tourism-chatbot.git
cd darija-tourism-chatbot
```

2. Install required packages:
```bash
pip install -q langchain langchain-community langchain-cohere cohere faiss-cpu datasets sentence-transformers gradio
```

3. Set up your Cohere API key:
```python
os.environ["COHERE_API_KEY"] = "your_api_key_here"
```

## 🚀 Usage

### Running the Chatbot

Execute the main script:
```bash
python chatbot.py
```

The Gradio interface will launch automatically and provide a shareable link.

### Example Questions

Try asking these questions in Darija:
- "واش كاينة أماكن سياحية فمراكش؟" (Are there tourist places in Marrakech?)
- "شنو هي أحسن المدن للسياحة فالمغرب؟" (What are the best cities for tourism in Morocco?)
- "كيفاش نوصل لطنجة؟" (How do I get to Tangier?)

## 🏗️ Architecture

### System Components

1. **Data Loading**: Loads Moroccan Darija QA dataset from HuggingFace
2. **Text Processing**: Splits documents into manageable chunks
3. **Embedding**: Converts text to vectors using multilingual embeddings
4. **Vector Store**: FAISS index for efficient retrieval
5. **LLM Integration**: Cohere's command-a model for generation
6. **RAG Chain**: Combines retrieval and generation
7. **Evaluation**: Accuracy measurement on test samples
8. **Interface**: Gradio web UI for user interaction

### Data Flow

```
User Query → Embedding → FAISS Search → Top-K Documents → 
Cohere LLM → Generated Answer → User
```

## 📊 Performance

The system includes an evaluation component that:
- Tests on a sample of 20 questions
- Measures accuracy using substring matching
- Calculates estimated loss (1 - accuracy)
- Respects API rate limits with built-in delays

## ⚙️ Configuration

### Key Parameters

- **Chunk Size**: 300 characters with 50 character overlap
- **Retrieval**: Top 3 relevant documents (k=3)
- **Temperature**: 0.2 for more deterministic responses
- **Model**: command-a-03-2025 (Cohere's latest)
- **Embeddings**: paraphrase-multilingual-MiniLM-L12-v2

### Rate Limiting

Trial API keys are limited to 10 calls per minute. The code includes:
- 7-second delays between API calls
- 10-second delays after errors
- Graceful error handling

## 📁 Project Structure

```
darija-tourism-chatbot/
│
├── chatbot.py              # Main application file
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── examples/               # Example queries and outputs
```

## 🔧 Customization

### Filtering Tourism Content

Modify the `filter_tourism` function to adjust content filtering:
```python
def filter_tourism(example):
    text = (example.get("question") or "").lower()
    keywords = ["سياحة", "مراكش", "طنجة", "سفر", "زيارة", "hotel", "travel"]
    return any(x in text for x in keywords)
```

### Adjusting Model Behavior

Change the temperature parameter for different response styles:
```python
llm = ChatCohere(model="command-a-03-2025", temperature=0.2)  # More deterministic
llm = ChatCohere(model="command-a-03-2025", temperature=0.8)  # More creative
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Cohere](https://cohere.com) for the powerful language model
- [HuggingFace](https://huggingface.co) for the Darija QA dataset
- [LangChain](https://langchain.com) for the RAG framework
- Moroccan NLP community for language resources

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

## ⚠️ Important Notes

- **API Key Security**: Never commit your API key to version control
- **Rate Limits**: Be mindful of API rate limits on trial accounts
- **Dataset**: The Moroccan-Darija-QA dataset is used under its respective license
- **Language**: Primary language is Moroccan Darija (Moroccan Arabic)

---

Made with ❤️ for the Moroccan tourism community 🇲🇦
