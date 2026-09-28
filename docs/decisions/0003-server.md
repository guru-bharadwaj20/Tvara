# Decision 0003: serving stack

Use FastAPI with an ASGI server for HTTP request validation and SSE streaming.
The inference executor remains in C++; HTTP parsing and chat-template rendering
remain in Python. A request owns its cancellation signal and generation state.

Avoid running blocking CPU decode directly on the event loop. The scheduler must
control executor work, while streaming yields validated response events. Test the
wire contract with a deterministic backend, clearly separated from real inference.
Schema support and tool calling will be added only when the grammar engine exists.
