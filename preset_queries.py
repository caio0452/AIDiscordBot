from dataclasses import dataclass
from ai import OAICompatibleProvider
from abc import ABC, abstractmethod
from abc import abstractmethod
from typing import List, Literal
from dataclasses import dataclass

import numpy as np
import json

class UserUtterance:
    def __init__(self, query: str, embedding: List[float]):
        self.query = query
        self.embedding = embedding

class PresetQueryMatcher(ABC):
    @abstractmethod
    def matches_utterance(self, utterance: UserUtterance) -> bool:
        raise NotImplementedError()

class KeywordMatcher(PresetQueryMatcher):
    def __init__(self, keywords: List[str], ignore_case: bool = True):
        self.keywords = keywords
        self.ignore_case = ignore_case

    def matches_utterance(self, utterance: UserUtterance) -> bool:
        if self.ignore_case:
            return any(keyword.lower() in utterance.query.lower() for keyword in self.keywords)
        else:
            return any(keyword in utterance.query for keyword in self.keywords)

class EmbeddingSimilarityMatcher(PresetQueryMatcher):
    def __init__(self, embedding: List[float], min_thresh: float):
        self.embedding = embedding  # TODO: assumes normalized vector
        self.min_thresh = min_thresh

    def matches_utterance(self, utterance: UserUtterance) -> bool:
        return np.dot(self.embedding, utterance.embedding) >= self.min_thresh

class PresetQuery:
    def __init__(self, *, preset_question: str, embedding: List[float], required_matchers: List[PresetQueryMatcher], min_required_similarity: float, answer:str):
        self.preset_question = preset_question
        self.embedding = embedding
        self.required_matchers = required_matchers
        self.min_required_similarity = 0.5
        self.answer = answer

    def matches(self, utterance: UserUtterance) -> bool:
        return all(matcher.matches_utterance(utterance) for matcher in self.required_matchers)

class PresetQueryManager:
    def __init__(self, embeddings_client: OAICompatibleProvider):
        self.embeddings_client = embeddings_client
        self._all_queries: List[PresetQuery] = []

    async def add_query(self, query: PresetQuery):
        self._all_queries.append(query)

    async def get_all_matching_user_utterance(self, utterance: str) -> List[PresetQuery]:
        ret = []
        utterance_emb = await self.embeddings_client.vectorize(utterance)
        user_utterance = UserUtterance(query=utterance, embedding=utterance_emb)
        for query in self._all_queries:
            if query.matches(user_utterance):
                ret.append(query)
        return ret

    @property
    def all_preset_queries(self):
        return self._all_queries
    
_cached_manager = None
async def manager(embeddings_client: OAICompatibleProvider) -> PresetQueryManager:
    json_file_path = "queries.json"
    global _cached_manager
    if _cached_manager is not None:
        return _cached_manager

    print("Building query db...")
    presets_manager = PresetQueryManager(embeddings_client)

    with open(json_file_path, 'r') as file:
        data = json.load(file)
        queries = data["queries"]

    for query in queries:
        question = query["question"]
        keywords = query["keywords"]
        min_similarity = query["min_similarity"]
        answer = query["answer"]

        embedding = await embeddings_client.vectorize(question)
        await presets_manager.add_query(
            PresetQuery(
                preset_question=question,
                embedding=embedding,
                required_matchers=[
                    KeywordMatcher(keywords),
                    EmbeddingSimilarityMatcher(embedding, min_similarity)
                ],
                min_required_similarity=min_similarity,
                answer=answer
            )
        )

    print("Done")
    _cached_manager = presets_manager
    return presets_manager


# This is a draft, that should be integrated to the system above
# TODO: should have a cooldown to avoid people repeatedly triggering on purpose
import providers
import openai
from ai import Prompt

CLASSIFICATION_PROMPT = Prompt([{
    "role": "system", 
    "content": 
    """
    You classify queries that may possibly be asking for estimates (ETA - estimated time of arrival) or release dates for Paper 1.21, which is currently not.
    Given a query, return only a JSON containing :
    * intent: state the user's intent
    * asking_release_info: will be true if the user is asking when the release will happen, or for estimates. Will be false if the user is simply mentioning a realease
    * project_name: The name of the project if any, may be "none"
    * version: The version mentioned if any, may be "none" 

    Examples:
    User query: hi, do you know when 1.21 will be out?
    JSON: {"intent": "The is asking when 1.21 will be out", "asking_release_info": true, "": project_name: "none", "version": "1.21"}

    User query: I would like to know when Paper will update
    JSON: {"intent": "The is asking when 1.21 will be out", "asking_release_info": true, "": project_name: "Paper", "version": "none"}

    User query: I'm not sure when the Paper 1.21 release will drop
    JSON: {"intent": "The user says they are not sure when Paper 1.21 will release", "asking_release_info": false, "": project_name: "Paper", "version": "1.21"}

    User query: guys can you believe 1.21 is released?????
    JSON: {"intent": "The user is asking others if they can believe 1.21 is already released", "asking_release_info": false, "": project_name: "none", "version": "1.21"} 

    User query: guys give info on when 1.21 release
    JSON: {"intent": "The user is asking for info on the 1.21 release date", "asking_release_info": true, "": project_name: "none", "version": "1.21"}  

    User query: vanilla 1.20 out when????
    JSON: {"intent": "The user is asking when vanilla 1.20 will be out", "asking_release_info": true, "": project_name: "vanilla", "version": "1.20"}

    User query: I hate it when people keep asking if 1.21 will come out
    JSON: {"intent": "The user is expressing hate for people asking when 1.21 will come out", "asking_release_info": false, "": project_name: "none", "version": "1.21"}

    User query: man, i wish Paper would just hard fork so they can update to 1.21
    JSON: {"intent": "The user is expressing their wishes for a Paper hard fork", "asking_release_info": false, "": project_name: "Paper", "version": "1.21"}

    User query: is velocity 1.21 already ready for download??
    JSON: {"intent": "The user is asking if velocity 1.21 is ready for download", "asking_release_info": true, "": project_name: "Velocity", "version": "1.21"}
    
    The query is now Query: ((query)). Reply with just the corresponding JSON.
    JSON: 
    """
}])

@dataclass
class ETAClassificationResult:
    similarity: float | None
    llm_classification_json: str | None
    finish_reason: Literal["failed_keyword_check", "failed_similarity_check", "failed_llm_check", "is_eta_question"]

def str_has_any_keyword(string: str, keywords: list[str]) -> bool:
    return any([kw in string for kw in keywords])

async def classify_eta_question(query: str) -> ETAClassificationResult:
    MIN_SIMILARITY = 0.4 # Magic
    needed_keywords = ["eta", "when", "out", "will", "paper", ".", "release", "updat", "progress", "come"] 
    oai_client = openai.AsyncOpenAI(api_key=providers.get_provider_by_name("EMBEDDINGS_PROVIDER").api_key)

    # Filter step 1
    if not str_has_any_keyword(query.lower(), needed_keywords):
        return ETAClassificationResult(None, None, "failed_keyword_check")

    # Filter step 2
    query_emb = await oai_client.embeddings.create(
        model="text-embedding-3-large", 
        input=query
    )
    reference_emb = await oai_client.embeddings.create(
        model="text-embedding-3-large", 
        input="Is there any ETA / estimate / progress on when 1.21 will release / come out?"
    )

    similarity = np.dot(query_emb.data[0].embedding, reference_emb.data[0].embedding)
    if similarity < MIN_SIMILARITY:
        return ETAClassificationResult(similarity, None, "failed_similarity_check")

    # Filter step 3
    llm_classification_resp = await oai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=CLASSIFICATION_PROMPT
            .replace("((query))", query)
            .to_openai_format()
    )
    llm_resp = llm_classification_resp.choices[0].message.content
    llm_resp_json = json.loads(llm_resp)

    wants_release_info: bool = llm_resp_json["asking_release_info"]
    proj_name: str = llm_resp_json["project_name"]
    version: str = llm_resp_json["version"]

    is_third_party_project = not str_has_any_keyword(proj_name.lower(), ["none", "paper", "velocity"])
    is_non_121_version = not str_has_any_keyword(version.lower(), ["1.21", "none"])

    is_eta_question = True
    
    if any([
        not wants_release_info,
        is_non_121_version,
        is_third_party_project
    ]):
        is_eta_question = False

    if proj_name == "none" and version == "none":
        is_eta_question = False

    if is_eta_question:
        return ETAClassificationResult(similarity, llm_resp, "is_eta_question")
    else:
        return ETAClassificationResult(similarity, llm_resp, "failed_llm_check")