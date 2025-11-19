# AutoSchemaKG: A Knowledge Graph Construction Framework with Schema Generation and Knowledge Graph Completion

This repository contains the implementation of AutoSchemaKG, a novel framework for automatic knowledge graph construction that combines schema generation via conceptualization. The framework is designed to address the challenges of constructing high-quality knowledge graphs from unstructured text.

This project uses the following paper and data:

*   **Paper:** [Read the paper](https://arxiv.org/abs/2505.23628)
*   **Full Data:** [Download the dataset](https://hkustconnect-my.sharepoint.com/:f:/g/personal/httsangaj_connect_ust_hk/EgpZiquW1dNCniDY_3O7WbIBMrHkXT1dFis7rhfeHZ_AOQ?e=zXF3Yg) (onedrive)
*   **Neo4j CSV Dumps:** [Download the dataset](https://huggingface.co/datasets/AlexFanWei/AutoSchemaKG) (huggingface dataset)

### Update
- (05/07) Update with batch generation and refactor the codebase. Add PDF-md-json instruction. [See PDF support](#pdf-support)
- (24/06) Add: ToG, Chinese KG construction (refer to example_scripts for KG construction with different language). Separate NV-embed-v2 transformers dependency.

## AutoSchemaKG Overview

AutoSchemaKG introduces a two-stage approach:
1. **Knowledge Graph Triple Extraction**: Extract triples comprising entities and events from text by using LLMs 
2. **Schema Induction**: Automatically generate schema for the knowledge graph by using conceptualization and create semantic bridges between seemingly disparate information to enable zero-shot inferencing across domains


The framework achieves state-of-the-art performance on multiple benchmarks and demonstrates strong generalization capabilities across different domains.

## ATLAS Knowledge Graphs

ATLAS (Automated Triple Linking And Schema induction) is a family of knowledge graphs created through the AutoSchemaKG framework, which enables fully autonomous knowledge graph construction without predefined schemas. Here's a summary of what ATLAS is and how it works:

### Key Features of ATLAS Knowledge Graphs

- **Scale**: Consists of 900+ million nodes connected by 5.9 billion edges
- **Autonomous Construction**: Built without predefined schemas or manual intervention
- **Three Variants**: ATLAS-Wiki (from Wikipedia), ATLAS-Pes2o (from academic papers), and ATLAS-CC (from Common Crawl)




## Project Structure

```
AutoSchemaKG/
├── atlas_rag/                # Main package directory
│   ├── kg_construction/      # Knowledge graph construction modules
│   ├── llm_generator/        # Components for large language model generation
│   ├── retriever/            # Retrieval components for RAG
│   ├── utils/                # Utility functions for various tasks
│   └── vectorstore/          # Components for managing vector storage and embeddings
├── example_data/             # Sample data for testing and examples
├── example_scripts/          # Example scripts for usage demonstrations
├── log/                      # Log files for tracking processes
├── neo4j_api_host/           # API hosting for Neo4j
├── neo4j_scripts/            # Scripts for managing Neo4j databases
├── tests/                    # Unit tests for the project
├── .gitignore                # Git ignore file
├── README.md                 # Main documentation for the project
├── atlas_billion_kg_usage.ipynb   # Example for hosting and RAG with ATLAS
├── atlas_full_pipeline.ipynb       # Full pipeline for constructing knowledge graphs
└── atlas_multihopqa.ipynb          # Example for benchmarking multi-hop QA datasets
```

The project is organized into several key components:
- `atlas_rag/`: Core package containing the main functionality
- Evaluation directories for different aspects of the system
- Database-related scripts and API hosting
- Example notebooks demonstrating usage
- Import and distribution directories for data management

## Install atlas-rag through pip
```bash
pip install atlas-rag
```
To support NV-embed-v2, install the transformers package with the version constraint >=4.42.4,<=4.47.1 by running:
```bash
pip install atlas-rag[nvembed]
```

### KG Construction with ATLAS

```python
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI
from transformers import pipeline
# client = OpenAI(api_key='<your_api_key>',base_url="<your_api_base_url>") 
# model_name = "meta-llama/llama-3.1-8b-instruct"

model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",
)
keyword = 'Dulce'
output_directory = f'import/{keyword}'
triple_generator = LLMGenerator(client, model_name=model_name)
kg_extraction_config = ProcessingConfig(
      model_path=model_name,
      data_directory="example_data",
      filename_pattern=filename_pattern,
      batch_size_triple=3, # batch size for triple extraction
      batch_size_concept=16, # batch size for concept generation
      output_directory=f"{output_directory}",
      max_new_tokens=2048,
      max_workers=3,
      remove_doc_spaces=True, # For removing duplicated spaces in the document text
)
kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)

# Construct entity&event graph
kg_extractor.run_extraction() # Involved LLM Generation
# Convert Triples Json to CSV
kg_extractor.convert_json_to_csv()
# Concept Generation
kg_extractor.generate_concept_csv(batch_size=64) # Involved LLM Generation
# Create Concept CSV
kg_extractor.create_concept_csv()
# Convert csv to graphml for networkx
kg_extractor.convert_to_graphml()
```

### Japanese Knowledge Graph Demo

The repository ships with `example_data/JP_sample.json`, which contains日本語テキストと`\"lang\": \"ja\"`メタデータを付与したテストデータです。日本語プロンプトを利用したパイプラインを試すには、以下の手順に従ってください。

1. `.env` に `OPENAI_API_KEY=<YOUR KEY>` を設定する（`kg_demo.py` は `dotenv` で読み込みます）。
2. `kg_demo.py` の `ProcessingConfig` は日本語ファイル (`filename_pattern=\"JP_sample\"`) を指しています。必要に応じて `data_directory` や `batch_size` を調整してください。
3. `uv run python kg_demo.py` を実行し、`import/JP_sample` 配下に生成される `kg_extraction` → `triples_csv` → `concepts` → `kg_graphml` を確認します。

## Large Knowledge Graph Hosting and Retrieval Augmented Generation

This repository provides support for hosting and implementing Retrieval Augmented Generation (RAG) over our constructed knowledge graphs: `ATLAS-wiki`, `ATLAS-pes2o`, and `ATLAS-cc`. For detailed instructions on hosting and running these knowledge graphs, please refer to the `atlas_billion_kg_usage.ipynb` notebook. 

## Building New Knowledge Graphs and Implementing RAG

The `atlas_full_pipeline.ipynb` notebook demonstrates how to:
- Build new knowledge graphs using AutoschemaKG
- Implement Retrieval Augmented Generation on your custom knowledge graphs


## Multi-hop Question Answering Evaluation

To replicate our multi-hop question answering evaluation results on benchmark datasets:
- `MuSiQue`
- `HotpotQA` 
- `2WikiMultiHopQA`

Please follow the instructions in the `atlas_multihopqa.ipynb` notebook, which contains all necessary code and configuration details.

## General Evaluation


The framework includes comprehensive evaluation metrics across three dimensions:
- Knowledge Graph Quality  (`EvaluateKGC`)
- Factual Consistency on FELM (`EvaluateFactuality`)
- General Performance on MMLU (`EValuateGeneralTask`)

Detailed evaluation procedures can be found in the respective evaluation directories.

## PDF Support
Creator: [swgj](https://github.com/Swgj)

Due to the version requirement of marker-pdf, we suggest you to create a new conda environment for PDF-to-Markdown Transformation.

Git clone PDF transform repo.
``` bash
git clone https://github.com/Swgj/pdf_process
cd pdf_process
conda create --name pdf-marker pip python=3.10
conda activate pdf-marker
pip install 'marker-pdf[full]'
pip install google-genai
```
Modify the config.yaml file.
``` yaml
processing_config:
  llm_service: "marker.services.azure_openai.AzureOpenAIService" # to use Azure OpenAI Service. To use default Gemini server, you can comment this line
  other_config:
    use_llm: true
    extract_images: false  # false means not to extract images and use LLM for text description; true means extract images but not generate descriptions
    page_range: null  # null means process all pages, or use List[int] format like [9, 10, 11, 12]
    max_concurrency: 2 # maximum number of concurrent processes
    #Azure OpenAI API configuration
    azure_endpoint: <your endpoint>
    azure_api_version: "2024-10-21"
    deployment_name: "gpt-4o"

# API configuration
api:
  # api_key_env: "GEMINI_API_KEY"  # Uncomment this line for Gemini API key
  api_key_env: "AZURE_API_KEY"

# Input path configuration - can be a file or folder path
input:
  # Supports relative and absolute paths
  path: "test_data"  # Can be a single file path or folder path
  # path: "data/Apple_Environmental_Progress_Report_2024.pdf"  # Example of a single file
  
  # If it's a folder, you can set file filtering conditions
  file_filters:
    extensions: [".pdf"]  # Only process PDF files
    recursive: true       # Whether to recursively process subfolders
    exclude_patterns:     # Exclude files that match these patterns
      - "*temp*"
      - "*~*"

# Output configuration
output:
  base_dir: "md_output"     # Output directory
  create_subdirs: true   # Whether to create a subdirectory for each input file
  format: "md"           # Output format (md, txt)
  
# Logging configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  show_progress: true
```

Run
```bash
bash run.sh
```
Cheers! You have a Markdown version of your PDF file. You can now change directories back to your parent directory and run the command below to obtain your JSON file for further Atlas-RAG KG construction.
```
python -m atlas_rag.kg_construction.utils.md_processing.markdown_to_json --input example_data/md_data --output example_data
```

## Citation

If you use this code in your research, please cite our paper:

```
@misc{bai2025autoschemakgautonomousknowledgegraph,
      title={AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora}, 
      author={Jiaxin Bai and Wei Fan and Qi Hu and Qing Zong and Chunyang Li and Hong Ting Tsang and Hongyu Luo and Yauwai Yim and Haoyu Huang and Xiao Zhou and Feng Qin and Tianshi Zheng and Xi Peng and Xin Yao and Huiwen Yang and Leijie Wu and Yi Ji and Gong Zhang and Renhai Chen and Yangqiu Song},
      year={2025},
      eprint={2505.23628},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23628}, 
}
```





## Contact

Jiaxin Bai: jbai@connect.ust.hk 

Dennis Hong Ting TSANG : httsangaj@connect.ust.hk

Haoyu Huang: haoyuhuang@link.cuhk.edu.hk
