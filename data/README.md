# GOV.UK Chat Evaluation

## Evaluation data sets

This directory should contain input data for evaluation. The contents are purposefully git ignored as this data can be sensitive.

We have the data stored on [Google Drive](https://docs.google.com/document/d/1tfgY8-5hCZDqw5zYAVpgYgB-VtjCKlzk1l1HMdOVV8U/edit?tab=t.0#heading=h.9awg20g181xc) which AI Team members can access. It is expected that files from Google Drive will be saved to this directory to run an evaluation.

When creating new evaluation data sets or running evaluation ad-hoc, each data set has to follow a particular format.

The evaluation pipeline enables generating the 'actual' response by an LLM-based component (e.g. jailbreak component, structured_answer component) on the fly by setting 'generate: true' in the config file for an evaluation task. Conversely, when setting 'generate: false', the data set supplied for the evaluation task already has to contain the 'actual' responses.

All data sets must be provided as a .jsonl file, where **each line** represents a single case to be evaluated formatted as a JSON object, i.e. each file is a list of JSON objects.

Below we will define the required fields for all evaluation tasks, and both the 'generate: true' and 'generate: false' scenario.

In all cases: additional fields beyond those listed are permitted but will be ignored during evaluation.


### Configuring the OpenSearch index for Retrieval and RAG Answer evaluations

You can configure which OpenSearch index to use when generating an input by updating the JSON to include an additional key–value pair where the key is `expected_opensearch_index` and the value is the index you would like to use:

```
{"expected_opensearch_index": "<opensearch-index-name>", ...}
```

This value is used to populate the `OPENSEARCH_INDEX` environment variable in Chat, which scopes OpenSearch requests to the specified index.

You will still need to configure the [OPENSEARCH_URL, OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD environment variables](https://github.com/alphagov/govuk-chat/blob/09daea62fa3319f527bf734f40fead2f7c928595/.env.example#L5-L7) in your local Ruby application
so that they point to the correct OpenSearch cluster.

There are more details and examples of how to do this in Retrieval and RAG Answers sections below.

## jailbreak_guardrails

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The question or input to evaluate. |
| `expected_outcome` | `bool` | Expected classification outcome. `false` = "no-jailbreak"; `true` = "jailbreak". |
| `actual_outcome` | `bool` | actual classification outcome produced by the jailbreak component. `false` = "no-jailbreak"; `true` = "jailbreak". |


### Example: 'generate: true'

```json
{
    "question": "Example question to evaluate",
    "expected_outcome": true
}
```

### Example: 'generate: false'

```json
{
    "question": "Example question to evaluate",
    "expected_outcome": true,
    "actual_outcome": false
}
```

## output_guardrails

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | **The answer** (!) we want to evaluate against the output guardrails. |
| `expected_triggered` | `bool` | Whether any guardrails are expected to trigger for the answer (in field: 'question'). |
| `expected_guardrails` | `object` | A dictionary mapping guardrail categories to booleans. `true` = guardrail expected to trigger, `false` = not expected. |
| `actual_triggered` | `bool` | Whether any guardrails were actually triggered by the output_guardrails component. |
| `actual_guardrails` | `object` | A dictionary mapping guardrail categories to booleans. `true` = guardrail triggered, `false` = not triggered. |

`expected_guardrails` and `actual_guardrails` keys:

| Key | Type | Description |
|-----|------|-------------|
| `sensitive_financial_matters` | `bool` | True if answer triggers the sensitive financial matters guardrail. |
| `appropriate_language` | `bool` | True if answer triggers the language appropriateness guardrail. |
| `political` | `bool` | True if answer triggers the political content guardrail. |
| `unsupported_statements` | `bool` | True if answer triggers the unsupported statements guardrail. |
| `contains_pii` | `bool` | True if answer triggers the personally identifiable information guardrail. |
| `illegal` | `bool` | True if answer triggers illegal-content guardrail. |
| `inappropriate_style` | `bool` | True if answer triggers the inappropriate style guardrail. |

### Example: 'generate: true'

```json
{
    "question": "Example answer to test the output guardrails.",
    "expected_triggered": true,
    "expected_guardrails": {
        "sensitive_financial_matters": true,
        "appropriate_language": false,
        "political": false,
        "unsupported_statements": false,
        "contains_pii": false,
        "illegal": true,
        "inappropriate_style": false}
}
```

### Example: 'generate: false'

```json
{
    "question": "Example answer to test the output guardrails.",
    "expected_triggered": true,
    "expected_guardrails": {
        "sensitive_financial_matters": true,
        "appropriate_language": false,
        "political": false,
        "unsupported_statements": false,
        "contains_pii": false,
        "illegal": true,
        "inappropriate_style": false},
        "expected_triggered": true,
    "actual_triggered": true,
    "actual_guardrails": {
        "sensitive_financial_matters": true,
        "appropriate_language": false,
        "political": false,
        "unsupported_statements": false,
        "contains_pii": false,
        "illegal": true,
        "inappropriate_style": false}
}
```

## question_router

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The question or input to evaluate. |
| `expected_outcome` | `string` | Expected question_routing_label. |
| `actual_outcome` | `object` | Actual question_routing_label assigned by question_router component. |

### Example: 'generate: true'

```json
{
    "question": "Example question to eveluate.",
    "expected_outcome": "genuine_rag"
}
```

### Example: 'generate: false'

```json
{
    "question": "Example question to evaluate.",
    "expected_outcome": "genuine_rag",
    "actual_outcome": "unclear_intent"
}
```

## rag_answers

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The question or input to be evaluate. |
| `ideal_answer` | `string` | The ideal/ground-truth answer. |
| `llm_answer` | `string` | Actual answer generated by the structured_answer component. |
| `expected_opensearch_index` | `string` | An optional field that can be used to configure the OpenSearch index used to generate the llm_answer. |
| `actual_opensearch_index` | `string` | The OpenSearch index used to generate the llm_answer. |
| `structured_contexts` | `list[object]` | A list of context chunks used for answering the question. Each element is an object as described below. |
`structured_contexts` object fields ( = chunk as defined in OpenSearch):

| Field | Type | Description |
|-------|------|-------------|
| `exact_path` | `string` | Exact path or URI to the source content chunk. |
| `title` | `string` | Title of the page or source document. |
| `heading_hierarchy` | `list[string]` or `null` | Hierarchical headings leading to this chunk. |
| `description` | `string` | Short description or summary of the source page. |
| `html_content` | `string` | Raw HTML content of the chunk. |

### Example: 'generate: true'

For:
* relevance
* faithfulness
* context_relevancy
* coherence


```json
{
  "question": "Example question to evaluate",
  "expected_opensearch_index": "<your-index-name>",
}
```

For:
* factual_precision
* factual_recall

```json
{
  "question": "Example question to be evaluate",
  "ideal_answer": "Example ideal/ground-truth answer",
  "expected_opensearch_index": "<your-index-name>",
}
```

### Example: 'generate: false'

For:
* relevance
* faithfulness
* context_relevancy
* coherence

```json
{
  "question": "Example question to be evaluate",
  "llm_answer": "Example answer produced by Chat",
  "structured_contexts": [
    {
      "exact_path": "/example-exact-path-to#chunk",
      "title": "Example title of the page",
      "heading_hierarchy": ["Heading1", "Heading2"],
      "description": "Example description of the page",
      "html_content": "Example html_content of the chunk"
    }
  ],
  "expected_opensearch_index": "<your-index-name>",
  "actual_opensearch_index": "govuk_chat_chunked_content",
}
```

For:
* factual_precision
* factual_recall

```json
{
  "question": "Example question to be evaluate",
  "ideal_answer": "Example ideal/ground-truth answer",
  "llm_answer": "Example answer produced by Chat",
  "structured_contexts": [
    {
      "exact_path": "/example-exact-path-to#chunk",
      "title": "Example title of the page",
      "heading_hierarchy": ["Heading1", "Heading2"],
      "description": "Example description of the page",
      "html_content": "Example html_content of the chunk"
    }
  ],
  "expected_opensearch_index": "<your-index-name>",
  "actual_opensearch_index": "govuk_chat_chunked_content",
}
```

## retrieval

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The question or input to evaluate. |
| `expected_exact_paths` | `list[string]` | List of expected exact_paths of chunks. These are exclusively used for rendering purposes since they're human readable.  |
| `expected_chunk_uids` | `list[string]` | List of expected chunk_uids. chunk_uids are used in all the retrieval evaluation calculations to map expected sources to actual sources. |
| `expected_opensearch_index` | `string` | An optional field that can be used to configure the OpenSearch index used to generate the llm_answer. |
| `actual_opensearch_index` | `string` | The OpenSearch index used to generate the llm_answer. |
| `actual_search_results` | `list[SearchResult]` | List of actual exact_paths, chunk_uids and similarity score information of chunks retrieved by the retrieval component. |

 SearchResult attributes:

| Field | Type | Description |
|-------|------|-------------|
| `exact_path` | `string` | The exact_path of a chunk. |
| `chunk_uid` | `string` | The uid associated with the chunk. This is generated in the GOV.UK Chat Ruby application.
| `weighted_score` | `float` | The weighted_score of the chunk. |
| `semantic_score` | `float` | The cosine similarity score of the chunk relative to the question. |

### Example: 'generate: true'

```json
{
    "question": "Example question to eveluate.",
    "expected_exact_paths": ["/example-exact-path-to#chunk"],
    "expected_chunk_uids": ["uid1"],
    "expected_opensearch_index": "<your-index-name>",
}
```

### Example: 'generate: false'

```json
{
    "question": "Example question to evaluate.",
    "expected_exact_paths": ["/example-exact-path-to#chunk"],
    "expected_chunk_uids": ["uid1"],
    "actual_search_results": {
        "exact_path": "/example-exact-path-to#chunk",
        "chunk_uid": "uid1",
        "weighted_score": 0.9,
        "semantic_score": 0.85,
    },
    "expected_opensearch_index": "<your-index-name>",
    "actual_opensearch_index": "govuk_chat_chunked_content",
}
```


## topic_tagger

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The question or input to evaluate. |
| `expected_primary_topic` | `string` | Expected primary topic of the question. |
| `expected_secondary_topic` | `string\|None` | Expected secondary topic of the question. |
| `actual_primary_topic` | `string` | Actual primary topic assigned to the question by topic-tagger component. |
| `actual_secondary_topic` | `string\|None` |  Actual secondary topic assigned to the question by topic-tagger component. |


### Example: 'generate: true'

```json
{
    "question": "Example question to eveluate.",
    "expected_primary_topic": "tax",
    "expected_secondary_topic": "business"
}
```

### Example: 'generate: false'

```json
{
    "question": "Example question to eveluate.",
    "expected_primary_topic": "tax",
    "expected_secondary_topic": "business",
    "actual_primary_topic": "benefits",
    "actual_secondary_topic": "business"
}
```
