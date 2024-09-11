from typing import Any, Dict, List, Optional
import opensearchpy


class QueryFeatureExtractor:
    """
    A class to extract features from a query for Learning to Rank (LTR) models.

    This class encapsulates the information needed to define a feature extractor,
    which is used to extract relevant features from search queries for LTR models.

    Attributes:
        feature_name (str): The name of the feature being extracted.
        query (Dict[str, Any]): The query template used for feature extraction.
        params (List[str]): List of parameters used in the query template. Defaults to ["query"].

    Methods:
        to_dict(): Converts the feature extractor configuration to a dictionary format.
    """

    def __init__(
        self,
        feature_name: str,
        query: Dict[str, Any],
        params: Optional[List[str]] = None,
    ):
        self.feature_name = feature_name
        self.params = params or ["query"]
        self.query = query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.feature_name,
            "params": self.params,
            "template_language": "mustache",
            "template": self.query,
        }


class LTRModelConfig:
    """
    A class to configure the Learning to Rank (LTR) model.

    This class manages the configuration for an LTR model, including the set of
    feature extractors used by the model.

    Attributes:
        feature_extractors (List[QueryFeatureExtractor]): List of feature extractors used by the model.
        featureset_name (str): A unique name for the feature set, generated based on the feature names.
        feature_names (List[str]): List of names of all features used in the model.

    Methods:
        to_dict(): Converts the LTR model configuration to a dictionary format.
    """

    def __init__(
        self,
        feature_extractors: List[QueryFeatureExtractor],
    ):
        self.feature_extractors = feature_extractors
        self.featureset_name = f"featureset_{abs(hash(tuple(fe.feature_name for fe in feature_extractors)))}"
        self.feature_names = [fe.feature_name for fe in feature_extractors]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "featureset": {
                "name": self.featureset_name,
                "features": [fe.to_dict() for fe in self.feature_extractors],
            }
        }


class FeatureLogger:
    """
    A class for logging features in the Learning to Rank (LTR) process.

    This class is responsible for logging and managing features used in the LTR model.
    It provides functionality to log features, retrieve logged features, and clear the feature log.

    Attributes:
        os_client (opensearchpy.OpenSearch): The OpenSearch client instance.
        index_name (str): The name of the index to log features to.
        ltr_config (LTRModelConfig): The configuration for the LTR model.

    Methods:
        _initialize_plugin(): Initializes the LTR plugin.
        _register_featureset(): Registers the feature set with the LTR plugin.
        extract_features(query_params: Dict[str, Any], doc_ids: List[str]) -> Dict[str, List[float]]:
            Extracts features for the given document IDs using the specified query parameters.
        clear_ltr_index(): Clears the LTR index.
    """

    def __init__(
        self,
        os_client: opensearchpy.OpenSearch,
        index_name: str,
        ltr_config: LTRModelConfig,
    ):
        self.os_client = os_client
        self.index_name = index_name
        self.ltr_config = ltr_config
        self._initialize_plugin()
        self._register_featureset()

    def _initialize_plugin(self):
        try:
            self.os_client.indices.create(
                index=".ltrstore",
                body={},
                ignore=400,  # Ignore 400 Index Already Exists exception
            )
        except opensearchpy.OpenSearchException as error:
            print(f"Failed to initialize LTR plugin: {str(error)}")

    def _register_featureset(self):
        try:
            self.os_client.transport.perform_request(
                "POST",
                f"/_ltr/_featureset/{self.ltr_config.featureset_name}",
                body=self.ltr_config.to_dict(),
            )
        except opensearchpy.RequestError as error:
            print(f"Failed to register featureset: {str(error)}")

    def extract_features(
        self, query_params: Dict[str, Any], doc_ids: List[str]
    ) -> Dict[str, List[float]]:
        body = {
            "_source": {"includes": []},
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"_id": doc_ids}},
                        {
                            "sltr": {
                                "_name": "logged_featureset",
                                "featureset": self.ltr_config.featureset_name,
                                "params": query_params,
                            }
                        },
                    ]
                }
            },
            "ext": {
                "ltr_log": {
                    "log_specs": {
                        "name": "log_entry1",
                        "named_query": "logged_featureset",
                    }
                }
            },
        }

        try:
            response = self.os_client.search(index=self.index_name, body=body)
        except opensearchpy.ConnectionError as error:
            print(f"Failed to connect to OpenSearch: {str(error)}")
            return {}
        except opensearchpy.RequestError as error:
            print(f"Failed to execute search: {str(error)}")
            return {}

        feature_values = {}
        for hit in response.get("hits", {}).get("hits", []):
            doc_id = hit.get("_id")
            ltr_log = hit.get("fields", {}).get("_ltrlog", [])
            if ltr_log and isinstance(ltr_log[0], dict):
                feature_values[doc_id] = ltr_log[0].get("log_entry1", [])

        return feature_values
