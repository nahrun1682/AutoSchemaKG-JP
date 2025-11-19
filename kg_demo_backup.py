from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from openai import OpenAI
from transformers import pipeline
import os
from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()

# カスタムLLMGeneratorを作成（extra_bodyを削除）
from atlas_rag.llm_generator import LLMGenerator

class FixedLLMGenerator(LLMGenerator):
    def _api_inference(self, message, max_new_tokens=8192,
                       temperature=0.7,
                       frequency_penalty=None,
                       response_format={"type": "text"},
                       return_text_only=True,
                       return_thinking=False,
                       reasoning_effort=None,
                       **kwargs):
        import time
        from openai import NOT_GIVEN
        
        start_time = time.time()
        
        # extra_bodyを削除してAPI呼び出し
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            max_tokens=max_new_tokens,
            temperature=temperature,
            frequency_penalty=NOT_GIVEN if frequency_penalty is None else frequency_penalty,
            response_format=response_format if response_format is not None else {"type": "text"},
            timeout=120,
            reasoning_effort=NOT_GIVEN if reasoning_effort is None else reasoning_effort,
            # extra_body を削除！
        )
        
        time_cost = time.time() - start_time
        content = response.choices[0].message.content
        
        if content is None and hasattr(response.choices[0].message, 'reasoning_content'):
            content = response.choices[0].message.reasoning_content
            
        validate_function = kwargs.get('validate_function', None)
        content = validate_function(content, **kwargs) if validate_function else content
        
        if '</think>' in content and not return_thinking:
            content = content.split('</think>')[-1].strip()
        else:
            if hasattr(response.choices[0].message, 'reasoning_content') and \
               response.choices[0].message.reasoning_content is not None and return_thinking:
                content = '<think>' + response.choices[0].message.reasoning_content + '</think>' + content
        
        if return_text_only:
            return content
        else:
            completion_usage_dict = response.usage.model_dump()
            completion_usage_dict['time'] = time_cost
            return content, completion_usage_dict
        
# カスタムLLMGeneratorを作成（extra_bodyを削除） 
# .envからAPIキーを自動取得

client = OpenAI()  # 環境変数 OPENAI_API_KEY から自動取得
model_name = "gpt-4o-mini"
# client = pipeline(
#     "text-generation",
#     model=model_name,
#     device_map="auto",
# )

keyword = 'Dulce'
filename_pattern = keyword
output_directory = f'import/{keyword}'

# triple_generator = LLMGenerator(client, model_name=model_name)
# 修正版LLMGeneratorを使用
triple_generator = FixedLLMGenerator(client, model_name=model_name)

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
      include_concept=True, # Whether to include concept generation step
      
)
kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)

# Construct entity&event graph
kg_extractor.run_extraction() # Involved LLM Generation
# Convert Triples Json to CSV
kg_extractor.convert_json_to_csv()
# Concept Generation
kg_extractor.generate_concept_csv_temp(batch_size=64) # Involved LLM Generation
# Create Concept CSV
kg_extractor.create_concept_csv()
# Convert csv to graphml for networkx
kg_extractor.convert_to_graphml()