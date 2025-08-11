# 🛡️ Policy Navigator Multi-Agent (PNA)
> A comprehensive AI-powered privacy policy navigation system that combines multiple specialized agents to provide intelligent insights on data privacy, security policies, and breach analysis.

## What Your Agent Does

The Policy Navigator Multi-Agent is a sophisticated system that combines three specialized AI agents to provide comprehensive privacy policy assistance:

1. **Policy Navigator Agent**: Core RAG-based system that answers questions about privacy policies, data protection regulations, and breach response procedures using embedded knowledge from multiple datasets.

2. **Web Scraper Agent**: Extracts and analyzes live web content from privacy-related websites, enabling real-time policy updates and external resource integration.

3. **Audio-to-Text Policy Agent**: Converts voice queries to text and processes them through the policy navigator, making the system accessible through natural speech interaction.

The system leverages advanced embedding models and vector search to provide contextually relevant answers from a comprehensive corpus of privacy documents, breach notifications, and policy guidelines.

## Project Structure

```
AIXPLAIN_PROJECT_FINAL/
├── README.md                          # Complete project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git exclusions
├── LICENSE                           # MIT License
└── aiXplain_Project/
    ├── app_str.py                    # Main Streamlit application
    ├── PNA_RAG.ipynb                 # Jupyter notebook with RAG implementation
    ├── Agent_Record.wav              # Sample audio recording
    ├── user_audio.wav                # User audio input file
    └── Data/                         # Privacy policy datasets
        ├── CTDAPD Dataset.csv        # Comprehensive privacy and data protection data
        ├── Data Breach Notifications.json    # Breach incident data in JSON format
        ├── Data_Breach_Notifications_Affecting_Washington_Residents__Personal_Information_Breakdown_.csv
        │                              # Washington residents breach analysis
        ├── master_seddataprivacyandsecuritypolicy_final_june-14-2021_0.pdf
        │                              # NYSED privacy and security policy
        └── PAG_Advisory_Committee_Report.pdf # Policy advisory guidelines
```

## How to Set It Up

### Prerequisites
- Python 3.8+
- aiXplain API key
- Required Python packages (see requirements.txt)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd AIXPLAIN_PROJECT_FINAL
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up aiXplain API key**
   - Obtain your API key from [aiXplain](https://aixplain.com)
   - Update the `AIXPLAIN_API_KEY` variable in `app_str.py`

4. **Run the Streamlit application**
   ```bash
   streamlit run app_str.py
   ```

### Configuration
- Update model IDs in `app_str.py` if needed:
  - `EMBEDDING_MODEL_ID`: For text embeddings
  - `LLM_TOOL_ID`: For language model responses
  - `SCRAPER_TOOL_ID`: For web scraping functionality

## Dataset/Source Links

The system utilizes the following datasets for comprehensive privacy policy knowledge:

- **CTDAPD Dataset**: Comprehensive privacy and data protection information
- **Data Breach Notifications**: JSON and CSV files containing breach incident data
- **Washington Residents Data Breach Analysis**: Detailed breakdown of personal information exposure
- **NYSED Privacy Policy**: Master privacy and security policy document
- **PAG Advisory Committee Report**: Policy advisory guidelines and recommendations

All datasets are stored in the `aiXplain_Project/aiXplain_Project/Data/` directory and are automatically processed during system initialization.

## Tool Integration Steps

### 1. aiXplain Platform Integration
- **Embedding Model**: Uses aiXplain's embedding service for vector representations
- **LLM Integration**: GPT-4 Mini through aiXplain's model factory
- **Agent Framework**: Leverages aiXplain's agent creation and management tools

### 2. RAG System Implementation
- **Text Chunking**: Intelligent document segmentation with overlap
- **Vector Search**: FAISS-based similarity search with cosine similarity fallback
- **Context Retrieval**: Top-k relevant document retrieval for enhanced responses

### 3. Multi-Modal Input Processing
- **Audio Recording**: Real-time audio capture using sounddevice
- **Speech Recognition**: Google Speech Recognition API integration
- **Web Scraping**: Dynamic content extraction from privacy policy websites

### 4. Streamlit Interface
- **Responsive UI**: Modern, intuitive web interface
- **Real-time Processing**: Live audio recording and transcription
- **Interactive Elements**: Dynamic query input and response display

## Example Inputs/Outputs

### Policy Navigator Agent
**Input**: "What regulations apply to third-party data sharing?"
**Output**: Comprehensive response citing relevant privacy laws, data protection regulations, and specific policy requirements from the embedded knowledge base.

### Web Scraper Agent
**Input**: "https://gdpr-info.eu/art-5-gdpr/"
**Output**: Extracted and processed content from the GDPR information website, providing current policy details and regulatory requirements.

### Audio-to-Text Policy Agent
**Input**: Voice query: "What should I do if my organization experiences a data breach?"
**Output**: 
1. Transcribed text: "What should I do if my organization experiences a data breach?"
2. Policy response: Step-by-step incident response procedures, notification requirements, and compliance guidelines.

## Future Improvements

### Enhanced Agent Capabilities
- **Summarization Agent**: Add specialized agent for policy document summarization and key point extraction
- **Analytics Agent**: Implement statistical analysis and trend identification from breach data
- **Compliance Agent**: Create agent for automated compliance checking and audit preparation
- **PDF Uploader Agent**: Implement agent for dynamic PDF document uploads and real-time policy integration

### User Interface Enhancements
- **Dashboard Analytics**: Interactive charts and visualizations of privacy trends
- **Policy Comparison Tools**: Side-by-side analysis of different privacy policies
- **Mobile Optimization**: Responsive design for mobile and tablet devices
- **Dark Mode**: User preference for interface appearance

### Data Integration Expansions
- **Real-time Updates**: Integration with privacy policy change notifications
- **External APIs**: Connect with regulatory databases and compliance services
- **Industry-Specific Data**: Sector-specific privacy requirements and best practices
- **International Standards**: Multi-jurisdictional privacy law coverage

### Performance and Memory Features
- **Intelligent Caching**: Cache frequently accessed policy information
- **Memory Persistence**: Maintain conversation context across sessions
- **Batch Processing**: Efficient handling of large document collections
- **Optimized Search**: Enhanced vector search algorithms and indexing

### Security and Compliance
- **Audit Logging**: Comprehensive tracking of all queries and responses
- **Access Control**: Role-based permissions for different user types
- **Data Encryption**: Enhanced security for sensitive policy information
- **Compliance Reporting**: Automated generation of compliance documentation
