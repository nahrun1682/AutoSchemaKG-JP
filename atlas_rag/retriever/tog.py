import numpy as np
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from typing import Optional
from atlas_rag.retriever.base import BaseEdgeRetriever
from atlas_rag.retriever.inference_config import InferenceConfig

class TogRetriever(BaseEdgeRetriever):
    def __init__(self, llm_generator, sentence_encoder, data, inference_config: Optional[InferenceConfig] = None):
        self.KG = data["KG"]

        self.node_list = list(self.KG.nodes)
        self.node_index = {node: idx for idx, node in enumerate(self.node_list)}
        self.edge_list = list(self.KG.edges)
        self.edge_list_with_relation = [(edge[0], self.KG.edges[edge]["relation"], edge[1])  for edge in self.edge_list]
        self.edge_list_string = [f"{edge[0]}  {self.KG.edges[edge]['relation']}  {edge[1]}" for edge in self.edge_list]
        
        self.llm_generator:LLMGenerator = llm_generator
        self.sentence_encoder:BaseEmbeddingModel = sentence_encoder        

        self.node_embeddings = data["node_embeddings"]
        self.edge_embeddings = data["edge_embeddings"]

        self.inference_config = inference_config if inference_config is not None else InferenceConfig()
        self.topk_neighbors = getattr(self.inference_config, "topk_neighbors", 20)
        if self.topk_neighbors is None or self.topk_neighbors <= 0:
            self.topk_neighbors = None
        ignored_relations = getattr(self.inference_config, "ignored_relations", ("mention in",))
        self.ignored_relations = set(ignored_relations) if ignored_relations else set()
        ignored_node_types = getattr(self.inference_config, "ignored_node_types", ("passage",))
        self.ignored_node_types = set(ignored_node_types) if ignored_node_types else set()

    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, ..."},
            {"role": "user", "content": f"Extract the named entities from: Are Portland International Airport and Gerald R. Ford International Airport both located in Oregon?"},
            {"role": "system", "content": "Portland International Airport, Gerald R. Ford International Airport, Oregon"},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]

        
        response = self.llm_generator.generate_response(messages)
        generated_text = response
        # print(generated_text)
        return generated_text
    


    def retrieve_topk_nodes(self, query, topN=5, **kwargs):
        # extract entities from the query
        entities = self.ner(query)
        print(f"[TOG] NER raw output: {entities}")
        entities = entities.split(", ")

        if len(entities) == 0:
            # If the NER cannot extract any entities, we 
            # use the query as the entity to do approximate search
            entities = [query]

        # evenly distribute the topk for each entity
        topk_for_each_entity = topN//len(entities)
    
        # retrieve the top k nodes
        topk_nodes = []

        for entity_index, entity in enumerate(entities):
            if entity in self.node_list:
                topk_nodes.append(entity)
    
        for entity_index, entity in enumerate(entities): 
            topk_for_this_entity = topk_for_each_entity + 1
            
            entity_embedding = self.sentence_encoder.encode([entity])
            # Calculate similarity scores using dot product
            scores = self.node_embeddings @ entity_embedding[0].T
            # Get top-k indices
            top_indices = np.argsort(scores)[-topk_for_this_entity:][::-1]
            topk_nodes += [self.node_list[i] for i in top_indices]
            
        topk_nodes = list(set(topk_nodes))

        if len(topk_nodes) > 2*topN:
            topk_nodes = topk_nodes[:2*topN]
        return topk_nodes

    def retrieve(self, query, topN=5, **kwargs):
        """ 
        Retrieve the top N paths that connect the entities in the query.
        Dmax is the maximum depth of the search.
        """
        Dmax = self.inference_config.Dmax
        # in the first step, we retrieve the top k nodes
        initial_nodes = self.retrieve_topk_nodes(query, topN=topN)
        print(f"[TOG] initial candidate nodes: {initial_nodes}")
        E = initial_nodes
        P = [ [e] for e in E]
        D = 0

        query_embedding = self.sentence_encoder.encode([query])

        while D <= Dmax:
            P = self.search(query, P, query_embedding)
            print(f"[TOG] after search depth {D}: {len(P)} paths")
            P = self.prune(query, P, topN)
            print(f"[TOG] after prune depth {D}: {len(P)} paths")
            
            if self.reasoning(query, P):
                generated_text = self.generate(query, P)
                break
            
            D += 1
        
        if D > Dmax:    
            generated_text = self.generate(query, P)
        
        # print(generated_text)
        return generated_text

    def search(self, query, P, query_embedding):
        new_paths = []
        for path in P:
            tail_entity = path[-1]
            sucessors = list(self.KG.successors(tail_entity))
            predecessors = list(self.KG.predecessors(tail_entity))
            print(f"[TOG] expanding {tail_entity}: {len(sucessors)} successors, {len(predecessors)} predecessors")

            # print(f"tail_entity: {tail_entity}")
            # print(f"sucessors: {sucessors}")
            # print(f"predecessors: {predecessors}")

            # # print the attributes of the tail_entity
            # print(f"attributes of the tail_entity: {self.KG.nodes[tail_entity]}")
           
            # remove the entity that is already in the path
            sucessors = [
                neighbour
                for neighbour in sucessors
                if neighbour not in path and self._is_valid_edge(tail_entity, neighbour)
            ]
            predecessors = [
                neighbour
                for neighbour in predecessors
                if neighbour not in path and self._is_valid_edge(neighbour, tail_entity)
            ]

            sucessors = self._select_topk_neighbors(sucessors, query_embedding)
            predecessors = self._select_topk_neighbors(predecessors, query_embedding)

            if len(sucessors) == 0 and len(predecessors) == 0:
                new_paths.append(path)
                continue
            for neighbour in sucessors:
                relation = self.KG.edges[(tail_entity, neighbour)]["relation"]
                new_path = path + [relation, neighbour]
                new_paths.append(new_path)
            
            for neighbour in predecessors:
                relation = self.KG.edges[(neighbour, tail_entity)]["relation"]
                new_path = path + [relation, neighbour]
                new_paths.append(new_path)
        
        return new_paths

    def _is_valid_edge(self, head, tail):
        relation = self.KG.edges[(head, tail)].get("relation", "")
        if relation in self.ignored_relations:
            return False
        tail_type = self.KG.nodes[tail].get("type")
        if tail_type in self.ignored_node_types:
            return False
        return True

    def _select_topk_neighbors(self, neighbors, query_embedding):
        if not neighbors:
            return neighbors
        if self.topk_neighbors is None or len(neighbors) <= self.topk_neighbors:
            return neighbors
        valid_nodes = []
        scores = []
        query_vec = query_embedding[0]
        for neighbour in neighbors:
            idx = self.node_index.get(neighbour)
            if idx is None:
                continue
            # similarity via dot product
            score = float(self.node_embeddings[idx] @ query_vec.T)
            valid_nodes.append(neighbour)
            scores.append(score)
        if not valid_nodes:
            return []
        top_indices = np.argsort(scores)[-self.topk_neighbors:][::-1]
        return [valid_nodes[i] for i in top_indices]
    
    def prune(self, query, P, topN=3):
        rated_paths = []
        batch_messages = []
        path_strings = []

        for path in P:
            path_string = ""
            for index, node_or_relation in enumerate(path):
                if index % 2 == 0:
                    id_path = self.KG.nodes[node_or_relation]["id"]
                else:
                    id_path = node_or_relation
                path_string += f"{id_path} --->"
            path_string = path_string[:-5]
            path_strings.append(path_string)

            prompt = (
                "Please rating the following path based on the relevance to the question. "
                "The ratings should be in the range of 1 to 5. 1 for least relevant and 5 for most relevant. "
                "Only provide the rating, do not provide any other information. "
                "The output should be a single integer number. If you think the path is not relevant, please provide 0. "
                "If you think the path is relevant, please provide a rating between 1 and 5. \n "
                f"Query: {query} \n path: {path_string}"
            )

            batch_messages.append(
                [
                    {"role": "system", "content": "Answer the question following the prompt."},
                    {"role": "user", "content": prompt},
                ]
            )

        if not batch_messages:
            return P[:topN]

        responses = self.llm_generator.generate_response(batch_messages)
        for idx, (response, path, path_string) in enumerate(zip(responses, P, path_strings)):
            print(f"[TOG] prune: rating response for path #{idx} = {response!r}")
            try:
                rating = int(response.strip())
            except (ValueError, AttributeError):
                print(f"[TOG] prune: failed to parse rating for path #{idx} (response={response!r}), skipping.")
                continue
            rated_paths.append((rating, path))
        
        if not rated_paths:
            print("[TOG] prune: no valid ratings returned, falling back to original ranking.")
            return P[:topN]
            
        # sort the paths based on the ratings
        sorted_paths = [path for _, path in sorted(rated_paths, key=lambda x: x[0], reverse=True)]
        
        return sorted_paths[:topN]

    def reasoning(self, query, P):
        triples = []
        for path in P:
            for i in range(0, len(path)-2, 2):

                # triples.append((path[i], path[i+1], path[i+2]))
                triples.append((self.KG.nodes[path[i]]["id"], path[i+1], self.KG.nodes[path[i+2]]["id"]))
        
        triples_string = [f"({triple[0]}, {triple[1]}, {triple[2]})" for triple in triples]
        triples_string = ". ".join(triples_string)

        prompt = f"Given a question and the associated retrieved knowledge graph triples (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triples and your knowledge (Yes or No). Query: {query} \n Knowledge triples: {triples_string}"
        
        messages = [{"role": "system", "content": "Answer the question following the prompt."},
        {"role": "user", "content": f"{prompt}"}]

        response = self.llm_generator.generate_response(messages)
        return "yes" in response.lower()

    def generate(self, query, P):
        triples = []
        for path in P:
            for i in range(0, len(path)-2, 2):
                # triples.append((path[i], path[i+1], path[i+2]))
                triples.append((self.KG.nodes[path[i]]["id"], path[i+1], self.KG.nodes[path[i+2]]["id"]))
        
        triples_string = [f"({triple[0]}, {triple[1]}, {triple[2]})" for triple in triples]
        
        # response = self.llm_generator.generate_with_context_kg(query, triples_string)
        return triples_string, ["N/A" for _ in range(len(triples_string))]
