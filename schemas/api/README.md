# OpenAI API Schemas

JSON Schema 2020-12 definitions for the OpenAI REST API, extracted from the official [OpenAI OpenAPI specification](https://github.com/openai/openai-openapi).

## Files

### `common.json`

Shared types used across multiple API endpoints:

- Chat completion messages (system, user, assistant, tool, function, developer roles)
- Tools and function calling (ChatCompletionTool, FunctionObject, FunctionParameters)
- Response formats (ResponseFormatText, ResponseFormatJsonObject, ResponseFormatJsonSchema)
- Shared utilities (CompletionUsage, ChatCompletionRole, tokens, logprobs)

### `chat_completion.json`

Chat Completion API (`POST /v1/chat/completions`):

- CreateChatCompletionRequest
- CreateChatCompletionResponse
- CreateChatCompletionStreamResponse (for streaming responses)
- Choice, delta, finish_reason types
- All message and tool call variants

### `embedding.json`

Embeddings API (`POST /v1/embeddings`):

- CreateEmbeddingRequest
- CreateEmbeddingResponse
- Embedding object

### `image.json`

Image Generation API (`POST /v1/images/generations`):

- CreateImageRequest
- CreateImageEditRequest
- CreateImageVariationRequest
- ImagesResponse
- Image object

### `audio.json`

Audio APIs:

- Transcription (`POST /v1/audio/transcriptions`): CreateTranscriptionRequest/Response
- Translation (`POST /v1/audio/translations`): CreateTranslationRequest/Response
- Speech (`POST /v1/audio/speech`): CreateSpeechRequest
- Various response formats (JSON, verbose JSON, diarized JSON, streaming)

### `moderation.json`

Moderation API (`POST /v1/moderations`):

- CreateModerationRequest
- CreateModerationResponse

### `completion.json`

Legacy Text Completion API (`POST /v1/completions`):

- CreateCompletionRequest
- CreateCompletionResponse
- CompletionChoice

### `models.json`

Models API (`GET /v1/models`, `GET /v1/models/{model}`):

- ListModelsResponse
- Model object

### `errors.json`

Error types:

- Error object (code, message, param, type)
- ErrorResponse (wrapper)
- ErrorEvent (for streaming error events)

## Schema References

### Internal References

Within the same file, schemas use internal JSON Pointer references:

```json
{
  "$ref": "#/$defs/ChatCompletionRequestMessage"
}
```

### External References

Cross-file references use relative file URIs:

```json
{
  "$ref": "common.json#/$defs/CompletionUsage"
}
```

## Format

All files follow [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/schema) format:

- Root `$schema` declaration
- `$defs` for schema definitions (not `definitions`)
- Uses `title`, `description` for documentation
- Includes OpenAI metadata (`x-oaiMeta`) where available

## Generation

These schemas are generated from the OpenAI OpenAPI specification using the `bootstrap_api_schemas.py` script:

```bash
cd tools/schema-bootstrap
python bootstrap_api_schemas.py
```

The script:

1. Parses the OpenAPI 3.1.0 YAML specification
2. Extracts schema definitions from `components.schemas`
3. Organizes them into logical categories
4. Resolves inter-schema dependencies
5. Converts OpenAPI references to JSON Schema format
6. Writes organized JSON Schema files with proper cross-references

## License

MIT (inherited from [openai-openapi](https://github.com/openai/openai-openapi))

## Usage

### Python

```python
import json
import jsonschema

with open('schemas/api/chat_completion.json') as f:
    schema = json.load(f)

request = {"model": "gpt-4", "messages": [...]}
jsonschema.validate(request, schema['$defs']['CreateChatCompletionRequest'])
```

### TypeScript

```typescript
import schema from './schemas/api/chat_completion.json';

interface CreateChatCompletionRequest
  extends schema['$defs']['CreateChatCompletionRequest'] {}
```

### Rust (with serde_json + jsonschema)

```rust
use serde_json::json;

let schema = serde_json::from_str(include_str!("../schemas/api/chat_completion.json"))?;
// Use with jsonschema validation crate
```
