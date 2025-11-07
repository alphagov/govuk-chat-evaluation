# GOV.UK Chat Evaluation

## Evaluation data sets

This directory should contain input data for evaluation. The contents are purposefully git ignored as this data can be sensitive.

We have the data stored on [Google Drive](https://docs.google.com/document/d/1tfgY8-5hCZDqw5zYAVpgYgB-VtjCKlzk1l1HMdOVV8U/edit?tab=t.0#heading=h.9awg20g181xc) which AI Team members can access. It is expected that files from Google Drive will be saved to this directory to run an evaluation.

## Expected format of data sets

When creating new evaluation data sets or running evaluation ad-hoc, the following formats are expected of the data.

All data sets must be provided as a .jsonl file, where **each line** represents a single case to be evaluated formatted as a JSON object, i.e. each file is a list of JSON objects. Below we will list examples detailing the minimum requirements for a single case (JSON object). Additional fields beyond those listed are permitted but will be ignored during evaluation.

### jailbreak_guardrails

#### config: "generate: true"

```json
{
    "question": "Example question to evaluate",
    "expected_outcome": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The question or prompt to evaluate. |
| `expected_outcome` | `bool` | Expected classification outcome. `false` = "no-jailbreak"; `true` = "jailbreak". |

#### config: "generate: false"

```json
{
    "question": "Example question to evaluate",
    "expected_outcome": true,
    "actual_outcome": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The question or prompt to evaluate. |
| `expected_outcome` | `bool` | Expected classification outcome. `false` = "no-jailbreak"; `true` = "jailbreak". |
| `actual_outcome` | `bool` | The component's actual classification outcome. `false` = "no-jailbreak"; `true` = "jailbreak". |

### rag_answers

#### config: "generate: true"

For:
* relevance 
* faithfulness  
* context_relevancy 
* coherence


```json
{
  "question": "string - the question asked by the user"
}
```


| Field | Type | Description |
|-------|------|-------------|
| `question` | `string` | The question or prompt to evaluate. |
| `expected_outcome` | `bool` | Expected classification outcome. `false` = "no-jailbreak"; `true` = "jailbreak". |
| `actual_outcome` | `bool` | The component's actual classification outcome. `false` = "no-jailbreak"; `true` = "jailbreak". |

For:
* factual_correctness

```json
{
  "question": "string - the question asked by the user",
  "ideal_answer": "string - the ideal / ground-truth answer"
}
```

#### config: "generate: false"

For:
* relevance 
* faithfulness  
* context_relevancy 
* coherence

```json
{
  "question": "string - the question asked by the user",
  "llm_answer": "string - the answer returned by the LLM",
  "structured_contexts": [
    {
      "exact_path": "string - exact path to chunk",
      "title": "string - page title",
      "heading_hierarchy": ["string", "string"] | null,
      "description": "string - description of the page",
      "html_content": "string - HTML content of the chunk"
    }
  ]
}
```

For:
* factual_correctness

```json
{
  "question": "string - the question asked by the user",
  "llm_answer": "string - the answer returned by the LLM",
  "ideal_answer": "string - the ideal / ground-truth answer",
  "structured_contexts": [
    {
      "exact_path": "string - exact path to chunk",
      "title": "string - page title",
      "heading_hierarchy": ["string", "string"] | null,
      "description": "string - description of the page",
      "html_content": "string - HTML content of the chunk"
    }
  ]
}
```


