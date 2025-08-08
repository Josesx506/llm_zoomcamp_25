
### Interface
The application is hosted on a ***fastapi*** backend server running with multiple endpoints. You 
can start the server with `poetry run uvicorn app:app --reload`. <br>

The application requires a db connection and works with either `postgres`/`sqlite3`. In 
production you can use `gpt-4o-mini`, or `gemma3:1b` in development on the devcontainer.

### API documentation
The application has 5 endpoints. 

| Endpoint | Request | Description |
| :------- | :------ | :---------- |
| `/conversations` | GET | Retrieve all conversation ids |
| `/conversations/{conv_id}/messages` | GET | Retrieve all messages associated with a conversation |
| `/conversations/{conv_id}/{msg_id}/upvote` | GET | Provide positive feedback to a LLM response |
| `/conversations/{conv_id}/{msg_id}/downvote` | GET | Provide negative feedback to a LLM response |
| `/generate` | POST | Retrieve a response from the RAG server. |

For each query, the server response includes a conversation id and message id in the response. 
```bash
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" \
    -d '{"prompt":"Who is Abraham Lincoln?", "conversation_id": null}'
```

An example response is
```json
{
  "response":"Abraham Lincoln was the sixteenth President of the United States, serving from 
  March 1861 until his assassination in April 1865. He is best known for leading the country 
  during the Civil War, preserving the Union, and working to end slavery through the 
  Emancipation Proclamation and the passage of the Thirteenth Amendment. Lincoln is often 
  regarded as one of America's greatest presidents due to his leadership and impact on American 
  history.",
  "conversation_id":6,
  "msg_id":54
}
```
To continue the conversation, include the returned `conversation_id` for subsequent requests. 
The application includes the ***history from the last 3 messages*** in a conversation to retain 
state during conversations while prioritizing context. Without this, every message is stateless, 
and the model cannot remember much. Including the entire conversation history in the prompt is a 
good way to run out of tokens when paying for the api. <br>

User feedback is key to evaluating online RAG performance. The `upvote` and `downvote` endpoints 
can be used to easily provide feedback for each prompt response. Values range between _"1=like, 0/None=neutral, -1=dislike"_.
By default, each prompt is neutral. 
>[!Note]
> The feedback button is not the same as a like button that can have more than 1 like/dislike. The values don't exceed the -1 to 1 range irrespective of how many times a response is upvoted.